//! Native Vulkan client window: live piano-roll visualization of YufMusicGen
//! generation.  Inference runs on a dedicated compute device in a worker
//! thread; the UI renders the roll with a Vulkan swapchain.

use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use ash::vk;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::Key;
use winit::window::{Window, WindowId};

use crate::sampler::Mt19937;

#[derive(Clone)]
pub struct GuiArgs {
    pub checkpoint: PathBuf,
    pub tokenizer: Option<PathBuf>,
    pub prompt: String,
    pub instrument: Option<String>,
    pub instrument_only: bool,
    pub prompt_midi: Option<PathBuf>,
    pub prompt_max_tokens: usize,
    pub steps: Option<usize>,
    pub seconds: Option<f64>,
    pub temperature: f32,
    pub top_p: f32,
    pub seed: u64,
}

#[derive(Clone, Default)]
struct SharedState {
    notes: Vec<RollNote>,
    status: String,
    progress: f32,
    done: bool,
    started: bool,
}

#[derive(Clone, Copy)]
struct RollNote {
    start: f32,
    duration: f32,
    pitch: f32,
    program: f32,
}

pub fn run(args: GuiArgs) -> Result<()> {
    let event_loop = EventLoop::new().context("cannot create winit event loop")?;
    let mut app = App {
        args,
        window: None,
        renderer: None,
        shared: Arc::new(Mutex::new(SharedState::default())),
        worker: None,
        worker_done: None,
    };
    event_loop.run_app(&mut app)?;
    Ok(())
}

struct App {
    args: GuiArgs,
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    shared: Arc<Mutex<SharedState>>,
    worker: Option<std::thread::JoinHandle<()>>,
    worker_done: Option<Receiver<()>>,
}

impl App {
    fn start_generation(&mut self) {
        let shared = self.shared.clone();
        let args = self.args.clone();
        let (done_tx, done_rx) = mpsc::channel();
        self.worker_done = Some(done_rx);
        self.worker = Some(std::thread::spawn(move || {
            if let Err(error) = generate_into_shared(&args, &shared) {
                if let Ok(mut state) = shared.lock() {
                    state.status = format!("error: {error}");
                    state.done = true;
                }
            }
            let _ = done_tx.send(());
        }));
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("YufMusicGen — Vulkan client")
                        .with_inner_size(LogicalSize::new(1100.0, 720.0)),
                )
                .expect("create window"),
        );
        let renderer = Renderer::new(&window).expect("create Vulkan renderer");
        self.window = Some(window);
        self.renderer = Some(renderer);
        if let Ok(mut state) = self.shared.lock() {
            state.status = "Press SPACE to generate".to_string();
        }
        if std::env::var("YUF_AUTO_START").as_deref() == Ok("1") {
            if let Ok(mut state) = self.shared.lock() {
                state.started = true;
                state.status = "starting…".to_string();
            }
            self.start_generation();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event, .. } => {
                let is_q = matches!(&event.logical_key, Key::Character(ch) if ch.eq_ignore_ascii_case("q"));
                if is_q {
                    event_loop.exit();
                }
                if event.logical_key == Key::Named(winit::keyboard::NamedKey::Space) {
                    let should_start = {
                        let state = self.shared.lock().unwrap();
                        !state.started && !state.done
                    };
                    if should_start {
                        if let Ok(mut state) = self.shared.lock() {
                            state.started = true;
                            state.status = "starting…".to_string();
                        }
                        self.start_generation();
                    }
                }
            }
            WindowEvent::Resized(size) => {
                if let Some(renderer) = &mut self.renderer {
                    renderer.resize(size.width.max(1), size.height.max(1));
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            let state = self.shared.lock().unwrap().clone();
            let title = format!(
                "YufMusicGen — {}{}",
                state.status,
                if state.done { " (done)" } else { "" }
            );
            window.set_title(&title);
            if let Some(renderer) = &mut self.renderer {
                let _ = renderer.draw(&state);
            }
        }
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

// ---------------------------------------------------------------------------
// Generation worker
// ---------------------------------------------------------------------------

fn generate_into_shared(args: &GuiArgs, shared: &Arc<Mutex<SharedState>>) -> Result<()> {
    let debug = std::env::var("YUF_DEBUG_GUI").is_ok();
    if debug {
        eprintln!("[worker] loading checkpoint");
    }
    set_status(shared, "loading checkpoint");
    let checkpoint = crate::checkpoint::Checkpoint::load(&args.checkpoint)?;
    let config = checkpoint.header.model_config;
    let codec = crate::generation::build_codec(&checkpoint, args.tokenizer.as_ref())?;

    if debug {
        eprintln!("[worker] initializing Vulkan compute");
    }
    set_status(shared, "initializing Vulkan compute");
    let mut model = crate::compute::model::Model::new(&checkpoint)?;
    if debug {
        eprintln!("[worker] Vulkan compute ready");
    }
    let mut rng = Mt19937::new(args.seed);

    let mut condition = vec![crate::generation::BOS];
    for byte in args.prompt.as_bytes() {
        condition.push(crate::generation::TEXT_OFFSET + *byte as u32);
    }
    condition.push(crate::generation::SEP);
    if let Some(instrument) = &args.instrument {
        let program = crate::instruments::resolve_program(instrument).map_err(anyhow::Error::msg)?;
        let raw_id = crate::generation::program_token_id(&codec, program)
            .with_context(|| format!("no Program_{program} token in vocabulary"))?;
        condition.push(crate::generation::MIDI_OFFSET + raw_id);
    }

    let mut logits = Vec::new();
    for (index, token) in condition.iter().enumerate() {
        logits = model.step(*token)?;
        if debug && index % 4 == 0 {
            eprintln!("[worker] conditioning {}/{}", index + 1, condition.len());
        }
        if index % 32 == 0 {
            set_status(shared, &format!("conditioning {}/{}", index + 1, condition.len()));
        }
    }

    let target_steps = args.steps.unwrap_or_else(|| {
        args.seconds
            .map(|seconds| (seconds * crate::generation::TOKENS_PER_SECOND) as usize)
            .unwrap_or(512)
    })
    .max(1);
    let midi_offset = crate::generation::MIDI_OFFSET;
    let mut generated: Vec<u32> = Vec::new();
    for index in 0..target_steps {
        let mut allowed = vec![f32::NEG_INFINITY; config.vocab_size];
        allowed[midi_offset as usize..].copy_from_slice(&logits[midi_offset as usize..]);
        if index >= crate::generation::MIN_MIDI_TOKENS_BEFORE_EOS {
            allowed[crate::generation::EOS as usize] = logits[crate::generation::EOS as usize];
        }
        let token =
            crate::sampler::sample_token(&allowed, args.temperature, args.top_p, &mut rng) as u32;
        if token == crate::generation::EOS {
            break;
        }
        generated.push(token);
        logits = model.step(token)?;
        if debug && index % 64 == 0 {
            eprintln!("[worker] token {}/{}", index + 1, target_steps);
        }
        if index % 8 == 0 {
            let mut midi_ids: Vec<u32> = Vec::new();
            let mut started = false;
            for sampled in &generated {
                if (midi_offset..config.vocab_size as u32).contains(sampled) {
                    started = true;
                    midi_ids.push(sampled - midi_offset);
                } else if started {
                    break;
                }
            }
            let score = codec.decode(&midi_ids);
            let mut state = shared.lock().unwrap();
            state.notes = score
                .tracks
                .iter()
                .flat_map(|track| {
                    track.notes.iter().map(move |note| RollNote {
                        start: note.start as f32,
                        duration: note.duration.max(1) as f32,
                        pitch: note.pitch as f32,
                        program: track.program as f32,
                    })
                })
                .collect();
            state.status = format!(
                "sampling {}/{} · {} notes",
                index + 1,
                target_steps,
                state.notes.len()
            );
            state.progress = (index + 1) as f32 / target_steps as f32;
            drop(state);
        }
    }
    {
        let mut state = shared.lock().unwrap();
        state.status = "done".to_string();
        state.progress = 1.0;
        state.done = true;
    }
    Ok(())
}

fn set_status(shared: &Arc<Mutex<SharedState>>, status: &str) {
    if let Ok(mut state) = shared.lock() {
        state.status = status.to_string();
    }
}

// ---------------------------------------------------------------------------
// Vulkan renderer
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    pos: [f32; 2],
    color: [f32; 3],
}

struct Renderer {
    entry: ash::Entry,
    instance: ash::Instance,
    surface: vk::SurfaceKHR,
    physical: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    queue_family: u32,
    swapchain: vk::SwapchainKHR,
    swapchain_loader: ash::khr::swapchain::Device,
    surface_loader: ash::khr::surface::Instance,
    surface_format: vk::SurfaceFormatKHR,
    extent: vk::Extent2D,
    images: Vec<vk::Image>,
    views: Vec<vk::ImageView>,
    framebuffers: Vec<vk::Framebuffer>,
    render_pass: vk::RenderPass,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    vertex_buffer: vk::Buffer,
    vertex_memory: vk::DeviceMemory,
    vertex_capacity: usize,
    vertex_ptr: *mut u8,
    last_vertex_count: usize,
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    image_available: Vec<vk::Semaphore>,
    render_finished: Vec<vk::Semaphore>,
    fences: Vec<vk::Fence>,
    frame: usize,
}

const MAX_NOTES: usize = 4096;
const MAX_VERTICES: usize = MAX_NOTES * 6;

impl Renderer {
    fn new(window: &Window) -> Result<Self> {
        let entry = unsafe { ash::Entry::load().context("cannot load Vulkan loader")? };
        let display_handle = window.display_handle().context("display handle")?.as_raw();
        let window_handle = window.window_handle().context("window handle")?.as_raw();
        let surface_extensions = ash_window::enumerate_required_extensions(display_handle)
            .context("cannot enumerate surface extensions")?;
        let app_info = vk::ApplicationInfo::default()
            .api_version(vk::API_VERSION_1_2)
            .application_name(c"yufmusicgen-vulkan")
            .engine_name(c"yufmusicgen-vulkan");
        let instance = unsafe {
            entry.create_instance(
                &vk::InstanceCreateInfo::default()
                    .application_info(&app_info)
                    .enabled_extension_names(surface_extensions),
                None,
            )
            .context("cannot create Vulkan instance")?
        };

        let surface = unsafe {
            ash_window::create_surface(&entry, &instance, display_handle, window_handle, None)
                .context("cannot create window surface")?
        };
        let physical = unsafe { instance.enumerate_physical_devices() }
            .context("no physical devices")?[0];
        let surface_loader = ash::khr::surface::Instance::new(&entry, &instance);
        let queue_family = unsafe { instance.get_physical_device_queue_family_properties(physical) }
            .iter()
            .enumerate()
            .find(|(index, props)| {
                props.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                    && unsafe {
                        surface_loader
                            .get_physical_device_surface_support(physical, *index as u32, surface)
                            .unwrap_or(false)
                    }
            })
            .map(|(index, _)| index as u32)
            .with_context(|| "no graphics+present queue family")?;
        let device = unsafe {
            instance.create_device(
                physical,
                &vk::DeviceCreateInfo::default().queue_create_infos(&[vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(queue_family)
                    .queue_priorities(&[1.0])])
                .enabled_extension_names(&[c"VK_KHR_swapchain".as_ptr()]),
                None,
            )
            .context("cannot create device")?
        };
        let queue = unsafe { device.get_device_queue(queue_family, 0) };
        let swapchain_loader = ash::khr::swapchain::Device::new(&instance, &device);

        let surface_format = unsafe { surface_loader.get_physical_device_surface_formats(physical, surface) }
            .context("no surface formats")?
            .into_iter()
            .find(|format| format.format == vk::Format::B8G8R8A8_UNORM)
            .unwrap_or(
                unsafe { surface_loader.get_physical_device_surface_formats(physical, surface) }
                    .map(|mut formats| formats.remove(0))
                    .unwrap_or(vk::SurfaceFormatKHR {
                        format: vk::Format::B8G8R8A8_UNORM,
                        color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
                    }),
            );
        let mut renderer = Self {
            entry,
            instance,
            surface,
            physical,
            device,
            queue,
            queue_family,
            swapchain: vk::SwapchainKHR::null(),
            swapchain_loader,
            surface_loader,
            surface_format,
            extent: vk::Extent2D::default(),
            images: Vec::new(),
            views: Vec::new(),
            framebuffers: Vec::new(),
            render_pass: vk::RenderPass::null(),
            pipeline: vk::Pipeline::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            vertex_buffer: vk::Buffer::null(),
            vertex_memory: vk::DeviceMemory::null(),
            vertex_capacity: 0,
            vertex_ptr: std::ptr::null_mut(),
            last_vertex_count: 0,
            command_pool: vk::CommandPool::null(),
            command_buffers: Vec::new(),
            image_available: Vec::new(),
            render_finished: Vec::new(),
            fences: Vec::new(),
            frame: 0,
        };
        renderer.init_swapchain(window);
        renderer.init_render_pass();
        renderer.init_pipeline();
        renderer.init_vertex_buffer();
        renderer.init_command_buffers();
        renderer.init_sync_objects();
        Ok(renderer)
    }

    fn init_swapchain(&mut self, window: &Window) {
        let capabilities = unsafe {
            self.surface_loader
                .get_physical_device_surface_capabilities(self.physical, self.surface)
        }
        .expect("surface capabilities");
        let size = window.inner_size();
        let extent = vk::Extent2D {
            width: size.width.clamp(1, 8192),
            height: size.height.clamp(1, 8192),
        };
        let image_count = capabilities.min_image_count.clamp(2, 8);
        let create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(self.surface)
            .min_image_count(image_count)
            .image_format(self.surface_format.format)
            .image_color_space(self.surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(vk::PresentModeKHR::FIFO)
            .clipped(true);
        self.swapchain = unsafe {
            self.swapchain_loader
                .create_swapchain(&create_info, None)
                .expect("create swapchain")
        };
        self.extent = extent;
        self.images = unsafe { self.swapchain_loader.get_swapchain_images(self.swapchain) }
            .expect("swapchain images");
        self.views = self
            .images
            .iter()
            .map(|&image| {
                let view_info = vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(self.surface_format.format)
                    .subresource_range(vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1).level_count(1));
                unsafe { self.device.create_image_view(&view_info, None) }
                    .expect("create image view")
            })
            .collect();
    }

    fn init_render_pass(&mut self) {
        let attachment = vk::AttachmentDescription::default()
            .format(self.surface_format.format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);
        let color_ref = vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let subpass = vk::SubpassDescription::default()
            .color_attachments(std::slice::from_ref(&color_ref))
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS);
        let dependency = vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);
        let render_pass = unsafe {
            self.device.create_render_pass(
                &vk::RenderPassCreateInfo::default()
                    .attachments(std::slice::from_ref(&attachment))
                    .subpasses(std::slice::from_ref(&subpass))
                    .dependencies(std::slice::from_ref(&dependency)),
                None,
            )
        }
        .expect("create render pass");
        self.render_pass = render_pass;
    }

    fn init_pipeline(&mut self) {
        let vertex_shader = Self::shader_module(
            &self.device,
            include_bytes!(concat!(env!("OUT_DIR"), "/spirv/roll_vert.spv")),
        );
        let fragment_shader = Self::shader_module(
            &self.device,
            include_bytes!(concat!(env!("OUT_DIR"), "/spirv/roll_frag.spv")),
        );
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vertex_shader)
                .name(c"main"),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(fragment_shader)
                .name(c"main"),
        ];
        let binding = vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(std::mem::size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX);
        let attributes = [
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(0),
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(8),
        ];
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(std::slice::from_ref(&binding))
            .vertex_attribute_descriptions(&attributes);
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport = vk::Viewport::default()
            .x(0.0)
            .y(0.0)
            .width(self.extent.width as f32)
            .height(self.extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0);
        let scissor = vk::Rect2D::default().extent(self.extent);
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewports(std::slice::from_ref(&viewport))
            .scissors(std::slice::from_ref(&scissor));
        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::NONE);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let color_blend = vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_write_mask(vk::ColorComponentFlags::R | vk::ColorComponentFlags::G | vk::ColorComponentFlags::B | vk::ColorComponentFlags::A);
        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
            .attachments(std::slice::from_ref(&color_blend));
        let pipeline_layout = unsafe {
            self.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default(),
                None,
            )
        }
        .expect("create pipeline layout");
        let pipeline = unsafe {
            self.device.create_graphics_pipelines(
                vk::PipelineCache::null(),
                &[vk::GraphicsPipelineCreateInfo::default()
                    .stages(&stages)
                    .vertex_input_state(&vertex_input)
                    .input_assembly_state(&input_assembly)
                    .viewport_state(&viewport_state)
                    .rasterization_state(&rasterization)
                    .multisample_state(&multisample)
                    .color_blend_state(&color_blend_state)
                    .layout(pipeline_layout)
                    .render_pass(self.render_pass)
                    .subpass(0)],
                None,
            )
        }
        .expect("create graphics pipeline")[0];
        self.pipeline_layout = pipeline_layout;
        self.pipeline = pipeline;
        unsafe {
            self.device.destroy_shader_module(vertex_shader, None);
            self.device.destroy_shader_module(fragment_shader, None);
        }
        self.framebuffers = self
            .views
            .iter()
            .map(|&view| {
                let view_ref = [view];
                unsafe {
                    self.device.create_framebuffer(
                        &vk::FramebufferCreateInfo::default()
                            .render_pass(self.render_pass)
                            .attachments(&view_ref)
                            .width(self.extent.width)
                            .height(self.extent.height)
                            .layers(1),
                        None,
                    )
                }
                .expect("create framebuffer")
            })
            .collect();
    }

    fn shader_module(device: &ash::Device, bytes: &[u8]) -> vk::ShaderModule {
        let words: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        unsafe {
            device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
        }
        .expect("create shader module")
    }

    fn init_vertex_buffer(&mut self) {
        let size = (MAX_VERTICES * std::mem::size_of::<Vertex>()) as vk::DeviceSize;
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::VERTEX_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        self.vertex_buffer = unsafe { self.device.create_buffer(&buffer_info, None) }
            .expect("create vertex buffer");
        let requirements = unsafe { self.device.get_buffer_memory_requirements(self.vertex_buffer) };
        let memory_type = unsafe { self.instance.get_physical_device_memory_properties(self.physical) }
            .memory_types
            .iter()
            .enumerate()
            .find(|(index, memory_type)| {
                requirements.memory_type_bits & (1 << index) != 0
                    && memory_type
                        .property_flags
                        .contains(
                            vk::MemoryPropertyFlags::HOST_VISIBLE
                                | vk::MemoryPropertyFlags::HOST_COHERENT,
                        )
            })
            .map(|(index, _)| index as u32)
            .expect("host-visible memory type");
        self.vertex_memory = unsafe {
            self.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(requirements.size)
                    .memory_type_index(memory_type),
                None,
            )
        }
        .expect("allocate vertex memory");
        unsafe { self.device.bind_buffer_memory(self.vertex_buffer, self.vertex_memory, 0) }
            .expect("bind vertex memory");
        let mapped = unsafe {
            self.device
                .map_memory(self.vertex_memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
        }
        .expect("map vertex memory");
        self.vertex_ptr = mapped as *mut u8;
        self.vertex_capacity = MAX_VERTICES;
    }

    fn init_command_buffers(&mut self) {
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(self.queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        self.command_pool = unsafe { self.device.create_command_pool(&pool_info, None) }
            .expect("create command pool");
        let count = self.views.len() as u32;
        self.command_buffers = unsafe {
            self.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(count),
            )
        }
        .expect("allocate command buffers");
        for (index, &framebuffer) in self.framebuffers.iter().enumerate() {
            let cmd = self.command_buffers[index];
            unsafe {
                self.device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())
                    .expect("begin command buffer");
                let clear = [vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.045, 0.055, 0.075, 1.0],
                    },
                }];
                let render_pass_info = vk::RenderPassBeginInfo::default()
                    .render_pass(self.render_pass)
                    .framebuffer(framebuffer)
                    .render_area(vk::Rect2D::default().extent(self.extent))
                    .clear_values(&clear);
                self.device.cmd_begin_render_pass(cmd, &render_pass_info, vk::SubpassContents::INLINE);
                self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
                self.device.cmd_bind_vertex_buffers(cmd, 0, &[self.vertex_buffer], &[0]);
                self.device.cmd_draw(cmd, 0, 1, 0, 0);
                self.device.cmd_end_render_pass(cmd);
                self.device.end_command_buffer(cmd).expect("end command buffer");
            }
        }
    }

    fn init_sync_objects(&mut self) {
        let count = self.views.len();
        for _ in 0..count {
            let semaphore_info = vk::SemaphoreCreateInfo::default();
            self.image_available.push(unsafe {
                self.device.create_semaphore(&semaphore_info, None)
            }.expect("create semaphore"));
            self.render_finished.push(unsafe {
                self.device.create_semaphore(&semaphore_info, None)
            }.expect("create semaphore"));
            self.fences.push(unsafe {
                self.device.create_fence(
                    &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                    None,
                )
            }.expect("create fence"));
        }
    }

    fn draw(&mut self, state: &SharedState) -> Result<()> {
        if std::env::var("YUF_DEBUG_GUI").is_ok() && self.frame % 120 == 0 {
            eprintln!(
                "[gui] frame={} notes={} vertices={}",
                self.frame,
                state.notes.len(),
                self.last_vertex_count
            );
        }
        let frame = self.frame % self.fences.len();
        unsafe {
            self.device
                .wait_for_fences(&[self.fences[frame]], true, u64::MAX)?;
            self.device.reset_fences(&[self.fences[frame]])?;
        }
        let image_index = unsafe {
            self.swapchain_loader.acquire_next_image(
                self.swapchain,
                u64::MAX,
                self.image_available[frame],
                vk::Fence::null(),
            )
        }
        .map_err(|error| anyhow::anyhow!("acquire next image failed: {error}"))?
        .0 as usize;

        self.build_vertices(&state.notes);
        unsafe {
            self.device.reset_command_buffer(self.command_buffers[image_index], vk::CommandBufferResetFlags::empty())?;
            let clear = [vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.045, 0.055, 0.075, 1.0],
                },
            }];
            self.device.begin_command_buffer(self.command_buffers[image_index], &vk::CommandBufferBeginInfo::default())?;
            let render_pass_info = vk::RenderPassBeginInfo::default()
                .render_pass(self.render_pass)
                .framebuffer(self.framebuffers[image_index])
                .render_area(vk::Rect2D::default().extent(self.extent))
                .clear_values(&clear);
            self.device.cmd_begin_render_pass(self.command_buffers[image_index], &render_pass_info, vk::SubpassContents::INLINE);
            self.device.cmd_bind_pipeline(self.command_buffers[image_index], vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            self.device.cmd_bind_vertex_buffers(self.command_buffers[image_index], 0, &[self.vertex_buffer], &[0]);
            self.device.cmd_draw(self.command_buffers[image_index], self.vertex_count() as u32, 1, 0, 0);
            self.device.cmd_end_render_pass(self.command_buffers[image_index]);
            self.device.end_command_buffer(self.command_buffers[image_index])?;
        }
        let wait_semaphores = [self.image_available[frame]];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let signal_semaphores = [self.render_finished[frame]];
        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(std::slice::from_ref(&self.command_buffers[image_index]))
            .signal_semaphores(&signal_semaphores);
        unsafe {
            self.device.queue_submit(self.queue, &[submit_info], self.fences[frame])?;
            self.swapchain_loader.queue_present(
                self.queue,
                &vk::PresentInfoKHR::default()
                    .wait_semaphores(&signal_semaphores)
                    .swapchains(std::slice::from_ref(&self.swapchain))
                    .image_indices(std::slice::from_ref(&(image_index as u32))),
            )?;
        }
        self.frame += 1;
        Ok(())
    }

    fn vertex_count(&self) -> usize {
        self.last_vertex_count
    }

    fn build_vertices(&mut self, notes: &[RollNote]) {
        let width = self.extent.width.max(1) as f32;
        let height = self.extent.height.max(1) as f32;
        let max_tick = notes
            .iter()
            .map(|note| note.start + note.duration)
            .fold(256.0f32, f32::max)
            .max(256.0);
        let scale_x = 0.92 * width / max_tick;
        let scale_y = 0.92 * height / 128.0;
        let ox = 0.04 * width;
        let oy = 0.04 * height;
        let to_clip = |x: f32, y: f32| -> [f32; 2] {
            [2.0 * x / width - 1.0, 1.0 - 2.0 * y / height]
        };
        let mut vertices: Vec<Vertex> = Vec::with_capacity(notes.len() * 6 + 64);

        // Bar lines every 32 ticks.
        let mut tick = 32.0f32;
        while tick <= max_tick {
            let x = ox + tick * scale_x;
            push_rect_clip(&mut vertices, to_clip(x - 0.7, oy), to_clip(x + 0.7, oy + 0.92 * height), [0.16, 0.18, 0.22]);
            tick += 32.0;
        }
        for note in notes.iter().take(MAX_NOTES) {
            let x = ox + note.start * scale_x;
            let y = oy + (127.0 - note.pitch) * scale_y;
            let w = (note.duration * scale_x).max(2.0);
            let h = (0.9 * scale_y).max(1.5);
            let color = program_color(note.program);
            push_rect_clip(&mut vertices, to_clip(x, y), to_clip(x + w, y + h), color);
        }
        self.last_vertex_count = vertices.len();
        unsafe {
            std::ptr::copy_nonoverlapping(
                vertices.as_ptr(),
                self.vertex_ptr as *mut Vertex,
                vertices.len(),
            );
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        if self.extent.width == width && self.extent.height == height {
            return;
        }
        unsafe {
            self.device.device_wait_idle().ok();
            self.destroy_swapchain_objects();
        }
        let capabilities = unsafe {
            self.surface_loader
                .get_physical_device_surface_capabilities(self.physical, self.surface)
        }
        .expect("surface capabilities");
        let extent = vk::Extent2D { width, height };
        let create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(self.surface)
            .min_image_count(capabilities.min_image_count.clamp(2, 8))
            .image_format(self.surface_format.format)
            .image_color_space(self.surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(vk::PresentModeKHR::FIFO)
            .clipped(true)
            .old_swapchain(self.swapchain);
        let old = self.swapchain;
        self.swapchain = unsafe {
            self.swapchain_loader
                .create_swapchain(&create_info, None)
                .expect("recreate swapchain")
        };
        unsafe {
            self.swapchain_loader.destroy_swapchain(old, None);
        }
        self.extent = extent;
        self.images = unsafe { self.swapchain_loader.get_swapchain_images(self.swapchain) }
            .expect("swapchain images");
        self.views = self
            .images
            .iter()
            .map(|&image| {
                let view_info = vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(self.surface_format.format)
                    .subresource_range(vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1).level_count(1));
                unsafe { self.device.create_image_view(&view_info, None) }
                    .expect("create image view")
            })
            .collect();
        self.framebuffers = self
            .views
            .iter()
            .map(|&view| {
                let view_ref = [view];
                unsafe {
                    self.device.create_framebuffer(
                        &vk::FramebufferCreateInfo::default()
                            .render_pass(self.render_pass)
                            .attachments(&view_ref)
                            .width(extent.width)
                            .height(extent.height)
                            .layers(1),
                        None,
                    )
                }
                .expect("create framebuffer")
            })
            .collect();
        unsafe {
            self.device.free_command_buffers(self.command_pool, &self.command_buffers);
        }
        self.command_buffers.clear();
        self.init_command_buffers();
    }

    unsafe fn destroy_swapchain_objects(&mut self) {
        for &view in &self.views {
            self.device.destroy_image_view(view, None);
        }
        self.views.clear();
        for &framebuffer in &self.framebuffers {
            self.device.destroy_framebuffer(framebuffer, None);
        }
        self.framebuffers.clear();
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().ok();
            for &fence in &self.fences {
                self.device.destroy_fence(fence, None);
            }
            for &semaphore in &self.image_available {
                self.device.destroy_semaphore(semaphore, None);
            }
            for &semaphore in &self.render_finished {
                self.device.destroy_semaphore(semaphore, None);
            }
            if self.command_pool != vk::CommandPool::null() {
                self.device.destroy_command_pool(self.command_pool, None);
            }
            if !self.vertex_ptr.is_null() {
                self.device.unmap_memory(self.vertex_memory);
            }
            if self.vertex_buffer != vk::Buffer::null() {
                self.device.destroy_buffer(self.vertex_buffer, None);
                self.device.free_memory(self.vertex_memory, None);
            }
            self.destroy_swapchain_objects();
            if self.pipeline != vk::Pipeline::null() {
                self.device.destroy_pipeline(self.pipeline, None);
            }
            if self.pipeline_layout != vk::PipelineLayout::null() {
                self.device.destroy_pipeline_layout(self.pipeline_layout, None);
            }
            if self.render_pass != vk::RenderPass::null() {
                self.device.destroy_render_pass(self.render_pass, None);
            }
            if self.swapchain != vk::SwapchainKHR::null() {
                self.swapchain_loader.destroy_swapchain(self.swapchain, None);
            }
            self.device.destroy_device(None);
            let surface_loader =
                ash::khr::surface::Instance::new(&self.entry, &self.instance);
            surface_loader.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}

fn push_rect_clip(
    vertices: &mut Vec<Vertex>,
    min: [f32; 2],
    max: [f32; 2],
    color: [f32; 3],
) {
    let quad = [
        [min[0], min[1]],
        [max[0], min[1]],
        [max[0], max[1]],
        [min[0], min[1]],
        [max[0], max[1]],
        [min[0], max[1]],
    ];
    for pos in quad {
        vertices.push(Vertex { pos, color });
    }
}

fn program_color(program: f32) -> [f32; 3] {
    let index = (program.max(0.0) as u32) % 12;
    const PALETTE: [[f32; 3]; 12] = [
        [0.96, 0.76, 0.34],
        [0.86, 0.40, 0.45],
        [0.36, 0.78, 0.60],
        [0.42, 0.62, 0.95],
        [0.76, 0.55, 0.88],
        [0.95, 0.55, 0.30],
        [0.50, 0.85, 0.85],
        [0.88, 0.68, 0.55],
        [0.60, 0.80, 0.40],
        [0.90, 0.48, 0.66],
        [0.55, 0.70, 0.95],
        [0.95, 0.82, 0.45],
    ];
    PALETTE[index as usize]
}
