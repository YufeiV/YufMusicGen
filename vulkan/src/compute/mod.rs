//! Vulkan compute inference for the YufMusicGen RWKV-7/ROSA model.
//!
//! One command buffer encodes a full single-token forward pass: embedding,
//! per-layer TimeMix / ROSA / FFN compute dispatches and the final LM head.
//! The buffer is recorded once at startup and re-submitted for every token, so
//! per-step CPU overhead stays tiny.

pub mod model;

use std::collections::HashMap;
use std::ffi::CStr;
use std::path::Path;

use anyhow::{Context, Result, ensure};
use ash::vk;

pub const WORKGROUP: u32 = 256;

// Behavior flags shared with the GLSL kernels.
pub const FLAG_HAS_BIAS: u32 = 1 << 0;
pub const FLAG_SILU: u32 = 1 << 1;
pub const FLAG_TANH: u32 = 1 << 2;
pub const FLAG_SIGMOID: u32 = 1 << 3;
pub const FLAG_ADD_RESIDUAL: u32 = 1 << 4;
pub const FLAG_USE_GATE: u32 = 1 << 5;
pub const FLAG_TRANSPOSE_W: u32 = 1 << 6;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PushConsts {
    pub in_off: u32,
    pub out_off: u32,
    pub weight_off: u32,
    pub bias_off: u32,
    pub gate_off: u32,
    pub residual_off: u32,
    pub rows: u32,
    pub cols: u32,
    pub k: u32,
    pub flags: u32,
    pub token_off: u32,
    pub extra0: u32,
    pub extra1: u32,
    pub extra2: u32,
    pub eps: f32,
}

impl PushConsts {
    pub const NONE: u32 = u32::MAX;

    pub fn new() -> Self {
        Self {
            in_off: Self::NONE,
            out_off: Self::NONE,
            weight_off: Self::NONE,
            bias_off: Self::NONE,
            gate_off: Self::NONE,
            residual_off: Self::NONE,
            rows: 1,
            cols: 0,
            k: 0,
            flags: 0,
            token_off: Self::NONE,
            extra0: Self::NONE,
            extra1: Self::NONE,
            extra2: Self::NONE,
            eps: 1e-5,
        }
    }
}

struct BufferObj {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: vk::DeviceSize,
}

impl BufferObj {
    unsafe fn destroy(&self, device: &ash::Device) {
        device.destroy_buffer(self.buffer, None);
        device.free_memory(self.memory, None);
    }
}

pub struct ComputeContext {
    _entry: ash::Entry,
    instance: ash::Instance,
    physical: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    queue_family: u32,
    weights: BufferObj,
    work: BufferObj,
    state: BufferObj,
    work_map: *mut f32,
    work_size: usize,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    pipelines: HashMap<&'static str, vk::Pipeline>,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    pub weights_count: u32,
    pub state_count: u32,
}

// Shader names -> SPIR-V bytes, embedded at compile time from build.rs output.
const SHADERS: [&str; 11] = [
    "embed",
    "linear",
    "layernorm_reduce",
    "layernorm_apply",
    "mix_inputs",
    "timemix_recurrence",
    "rosa_update",
    "rosa_read",
    "ffn_combine",
    "noop",
    "copy",
];

/// Read a physical device's name as a UTF-8 string.
///
/// Vulkan device names are `char[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE]` and are
/// not guaranteed to be valid UTF-8 (drivers may return raw UTF-8/ASCII, and
/// some overlay/driver stacks return arbitrary bytes).  Always go through
/// `to_string_lossy` so the string printed to the console is valid UTF-8;
/// Windows console mode panics on non-UTF-8 output.
fn device_name(instance: &ash::Instance, device: vk::PhysicalDevice) -> String {
    // Keep the properties in a named local so the C string read cannot outlive
    // a temporary struct, then sanitize invalid bytes for console output.
    let properties = unsafe { instance.get_physical_device_properties(device) };
    unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

impl ComputeContext {
    pub fn new(
        weights_data: &[f32],
        work_float_count: usize,
        state_float_count: usize,
    ) -> Result<Self> {
        let _entry = unsafe { ash::Entry::load().context("cannot load the Vulkan loader")? };
        disable_intrusive_layers(&_entry);
        let app_info = vk::ApplicationInfo::default()
            .api_version(vk::API_VERSION_1_2)
            .application_name(c"yufmusicgen-vulkan")
            .engine_name(c"yufmusicgen-vulkan");
        let instance_create_info =
            vk::InstanceCreateInfo::default().application_info(&app_info);
        let instance = unsafe {
            _entry
                .create_instance(&instance_create_info, None)
                .context("cannot create a Vulkan instance")?
        };

        let devices = unsafe { instance.enumerate_physical_devices() }
            .context("no Vulkan physical device found")?;
        ensure!(!devices.is_empty(), "no Vulkan physical device found");
        let requested = std::env::var("YUF_DEVICE").unwrap_or_default();
        let physical = if requested.is_empty() {
            devices[0]
        } else if let Ok(index) = requested.parse::<usize>() {
            *devices
                .get(index)
                .with_context(|| format!("no physical device at index {index}"))?
        } else {
            let lower = requested.to_lowercase();
            *devices
                .iter()
                .find(|device| {
                    let name = device_name(&instance, **device).to_lowercase();
                    name.contains(&lower)
                })
                .with_context(|| format!("no physical device matching {requested:?}"))?
        };
        for (index, device) in devices.iter().enumerate() {
            eprintln!("[vulkan] device[{index}] {}", device_name(&instance, *device));
        }
        eprintln!("[vulkan] device: {}", device_name(&instance, physical));

        let queue_family = unsafe { instance.get_physical_device_queue_family_properties(physical) }
            .iter()
            .position(|props| props.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .with_context(|| "no compute queue family available")? as u32;

        let priorities = [1.0f32];
        let queue_create_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&priorities);
        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_create_info));
        let device = unsafe {
            instance
                .create_device(physical, &device_create_info, None)
                .context("cannot create a Vulkan logical device")?
        };
        let queue = unsafe { device.get_device_queue(queue_family, 0) };

        let mut context = Self {
            _entry,
            instance,
            physical,
            device,
            queue,
            queue_family,
            weights: BufferObj {
                buffer: vk::Buffer::null(),
                memory: vk::DeviceMemory::null(),
                size: 0,
            },
            work: BufferObj {
                buffer: vk::Buffer::null(),
                memory: vk::DeviceMemory::null(),
                size: 0,
            },
            state: BufferObj {
                buffer: vk::Buffer::null(),
                memory: vk::DeviceMemory::null(),
                size: 0,
            },
            work_map: std::ptr::null_mut(),
            work_size: 0,
            descriptor_set_layout: vk::DescriptorSetLayout::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            descriptor_pool: vk::DescriptorPool::null(),
            descriptor_set: vk::DescriptorSet::null(),
            pipelines: HashMap::new(),
            command_pool: vk::CommandPool::null(),
            command_buffer: vk::CommandBuffer::null(),
            fence: vk::Fence::null(),
            weights_count: weights_data.len() as u32,
            state_count: state_float_count as u32,
        };

        context
            .create_buffers(weights_data, work_float_count, state_float_count)
            .context("cannot allocate Vulkan buffers")?;
        context.create_descriptors().context("cannot create descriptors")?;
        context.create_pipelines().context("cannot create pipelines")?;
        context.create_command_buffer().context("cannot create command buffer")?;
        Ok(context)
    }

    fn create_buffers(
        &mut self,
        weights_data: &[f32],
        work_float_count: usize,
        state_float_count: usize,
    ) -> Result<()> {
        // Weights: device-local, uploaded through a host-visible staging buffer.
        let weights_bytes = (weights_data.len() * 4) as vk::DeviceSize;
        let staging = self.create_buffer(
            weights_bytes,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let staging_ptr = unsafe {
            self.device
                .map_memory(staging.memory, 0, staging.size, vk::MemoryMapFlags::empty())?
        };
        unsafe {
            std::ptr::copy_nonoverlapping(
                weights_data.as_ptr(),
                staging_ptr as *mut f32,
                weights_data.len(),
            );
            self.device.unmap_memory(staging.memory);
        }
        self.weights = self.create_buffer(
            weights_bytes,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let work_bytes = (work_float_count * 4) as vk::DeviceSize;
        self.work = self.create_buffer(
            work_bytes,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        self.work_map = unsafe {
            self.device
                .map_memory(self.work.memory, 0, self.work.size, vk::MemoryMapFlags::empty())?
                as *mut f32
        };
        self.work_size = work_float_count;

        let state_bytes = (state_float_count * 4) as vk::DeviceSize;
        self.state = self.create_buffer(
            state_bytes,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        // One-shot copy: staging -> weights, and zero-fill the state buffer.
        let command_pool = self.create_command_pool()?;
        let command_buffer = self.allocate_command_buffer(command_pool)?;
        let begin_info = vk::CommandBufferBeginInfo::default();
        unsafe {
            self.device
                .begin_command_buffer(command_buffer, &begin_info)?;
            self.device
                .cmd_copy_buffer(command_buffer, staging.buffer, self.weights.buffer, &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: weights_bytes,
                }]);
            self.device.cmd_fill_buffer(
                command_buffer,
                self.state.buffer,
                0,
                state_bytes,
                0,
            );
            // The work buffer holds the per-layer `previous` token vectors
            // that `mix_inputs` reads on the very first step.  The reference
            // model starts those at zero; host-visible allocations are not
            // guaranteed to be zeroed, so fill the whole buffer explicitly.
            self.device.cmd_fill_buffer(
                command_buffer,
                self.work.buffer,
                0,
                work_bytes,
                0,
            );
            self.device.end_command_buffer(command_buffer)?;
            let fence = self.create_fence()?;
            self.submit(&[command_buffer], fence)?;
            self.device.wait_for_fences(&[fence], true, u64::MAX)?;
            self.device.destroy_fence(fence, None);
            self.device.free_command_buffers(command_pool, &[command_buffer]);
            self.device.destroy_command_pool(command_pool, None);
            staging.destroy(&self.device);
        }
        Ok(())
    }

    fn create_buffer(
        &self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        memory_properties: vk::MemoryPropertyFlags,
    ) -> Result<BufferObj> {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { self.device.create_buffer(&buffer_info, None)? };
        let requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        let memory_type = self
            .find_memory_type(requirements.memory_type_bits, memory_properties)
            .with_context(|| "no suitable Vulkan memory type for buffer")?;
        let allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type);
        let memory = unsafe { self.device.allocate_memory(&allocate_info, None)? };
        unsafe {
            self.device.bind_buffer_memory(buffer, memory, 0)?;
        }
        Ok(BufferObj {
            buffer,
            memory,
            size,
        })
    }

    fn find_memory_type(
        &self,
        type_filter: u32,
        properties: vk::MemoryPropertyFlags,
    ) -> Option<u32> {
        let memory_properties =
            unsafe { self.instance.get_physical_device_memory_properties(self.physical) };
        for (index, memory_type) in memory_properties.memory_types.iter().enumerate() {
            if type_filter & (1 << index) != 0
                && memory_type.property_flags.contains(properties)
            {
                return Some(index as u32);
            }
        }
        None
    }

    fn create_descriptors(&mut self) -> Result<()> {
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        self.descriptor_set_layout = unsafe {
            self.device
                .create_descriptor_set_layout(&layout_info, None)?
        };

        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<PushConsts>() as u32);
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&self.descriptor_set_layout))
            .push_constant_ranges(std::slice::from_ref(&push_constant_range));
        self.pipeline_layout = unsafe {
            self.device
                .create_pipeline_layout(&pipeline_layout_info, None)?
        };

        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(3)];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(1);
        self.descriptor_pool = unsafe { self.device.create_descriptor_pool(&pool_info, None)? };

        let layout = self.descriptor_set_layout;
        let allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(std::slice::from_ref(&layout));
        self.descriptor_set = unsafe {
            self.device
                .allocate_descriptor_sets(&allocate_info)?
                .pop()
                .with_context(|| "failed to allocate descriptor set")?
        };

        let weight_descriptor = vk::DescriptorBufferInfo::default()
            .buffer(self.weights.buffer)
            .offset(0)
            .range(self.weights.size);
        let work_descriptor = vk::DescriptorBufferInfo::default()
            .buffer(self.work.buffer)
            .offset(0)
            .range(self.work.size);
        let state_descriptor = vk::DescriptorBufferInfo::default()
            .buffer(self.state.buffer)
            .offset(0)
            .range(self.state.size);
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&weight_descriptor)),
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&work_descriptor)),
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&state_descriptor)),
        ];
        unsafe { self.device.update_descriptor_sets(&writes, &[]) };
        Ok(())
    }

    fn create_pipelines(&mut self) -> Result<()> {
        let mut stages = Vec::new();
        for name in SHADERS {
            let spv_path = Path::new(env!("OUT_DIR")).join("spirv").join(format!("{name}.spv"));
            let bytes = std::fs::read(&spv_path)
                .with_context(|| format!("missing compiled shader {}", spv_path.display()))?;
            let words: Vec<u32> = bytes
                .chunks_exact(4)
                .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
            let module_info = vk::ShaderModuleCreateInfo::default().code(&words);
            let module = unsafe { self.device.create_shader_module(&module_info, None)? };
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(c"main");
            let pipeline_info = vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(self.pipeline_layout);
            let pipeline = unsafe {
                match self.device.create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[pipeline_info],
                    None,
                ) {
                    Ok(pipelines) => pipelines
                        .into_iter()
                        .next()
                        .with_context(|| format!("failed to create pipeline {name}"))?,
                    Err((_, err)) => {
                        anyhow::bail!("failed to create pipeline {name}: {err}")
                    }
                }
            };
            unsafe { self.device.destroy_shader_module(module, None) };
            stages.push((name, pipeline));
        }
        self.pipelines = stages.into_iter().collect();
        Ok(())
    }

    fn create_command_pool(&self) -> Result<vk::CommandPool> {
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(self.queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        Ok(unsafe { self.device.create_command_pool(&pool_info, None)? })
    }

    fn allocate_command_buffer(&self, pool: vk::CommandPool) -> Result<vk::CommandBuffer> {
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        Ok(unsafe {
            self.device
                .allocate_command_buffers(&allocate_info)?
                .pop()
                .with_context(|| "failed to allocate command buffer")?
        })
    }

    fn create_fence(&self) -> Result<vk::Fence> {
        Ok(unsafe { self.device.create_fence(&vk::FenceCreateInfo::default(), None)? })
    }

    fn create_command_buffer(&mut self) -> Result<()> {
        self.command_pool = self.create_command_pool()?;
        self.command_buffer = self.allocate_command_buffer(self.command_pool)?;
        self.fence = self.create_fence()?;
        Ok(())
    }

    fn submit(&self, command_buffers: &[vk::CommandBuffer], fence: vk::Fence) -> Result<()> {
        let submit_info = vk::SubmitInfo::default().command_buffers(command_buffers);
        unsafe {
            self.device
                .queue_submit(self.queue, std::slice::from_ref(&submit_info), fence)
                .context("queue submit failed")?;
        }
        Ok(())
    }

    pub fn record_dispatch(&self, cmd: vk::CommandBuffer, kernel: &str, pc: &PushConsts, x: u32) {
        unsafe {
            let pipeline = self.pipelines[kernel];
            self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                std::slice::from_ref(&self.descriptor_set),
                &[],
            );
            self.device.cmd_push_constants(
                cmd,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(pc),
            );
            self.device.cmd_dispatch(cmd, x, 1, 1);
        }
    }

    /// Debug helper: bind a pipeline and dispatch without binding descriptors
    /// or pushing constants, to isolate driver-side issues.
    pub fn record_bare_dispatch(&self, cmd: vk::CommandBuffer, kernel: &str, x: u32) {
        unsafe {
            let pipeline = self.pipelines[kernel];
            self.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            self.device.cmd_dispatch(cmd, x, 1, 1);
        }
    }

    pub fn record_barrier(&self, cmd: vk::CommandBuffer) {
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
        unsafe {
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&barrier),
                &[],
                &[],
            );
        }
    }

    /// Begin recording the reusable per-token command buffer.
    pub fn begin_step_record(&self) -> Result<vk::CommandBuffer> {
        unsafe {
            self.device
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())?;
            let begin_info = vk::CommandBufferBeginInfo::default();
            self.device.begin_command_buffer(self.command_buffer, &begin_info)?;
        }
        Ok(self.command_buffer)
    }

    pub fn end_step_record(&self, cmd: vk::CommandBuffer) -> Result<()> {
        ensure!(cmd == self.command_buffer, "wrong command buffer");
        unsafe { self.device.end_command_buffer(cmd)? }
        Ok(())
    }

    /// Submit the recorded step and block until the GPU finishes.
    pub fn execute_step(&self) -> Result<()> {
        self.submit(&[self.command_buffer], self.fence)?;
        unsafe {
            self.device
                .wait_for_fences(&[self.fence], true, u64::MAX)?;
            self.device.reset_fences(&[self.fence])?;
        }
        Ok(())
    }

    pub fn write_work(&self, offset: usize, values: &[f32]) {
        assert!(
            offset + values.len() <= self.work_size,
            "work buffer write out of range"
        );
        unsafe {
            std::ptr::copy_nonoverlapping(
                values.as_ptr(),
                self.work_map.add(offset),
                values.len(),
            );
        }
    }

    pub fn read_work(&self, offset: usize, count: usize) -> Vec<f32> {
        assert!(
            offset + count <= self.work_size,
            "work buffer read out of range"
        );
        let mut values = vec![0.0f32; count];
        unsafe {
            std::ptr::copy_nonoverlapping(self.work_map.add(offset), values.as_mut_ptr(), count);
        }
        values
    }

    /// Read back a region of the device-local state buffer through a staging
    /// copy (debugging helper).
    pub fn read_state(&self, offset: usize, count: usize) -> Result<Vec<f32>> {
        assert!(offset + count <= self.state_count as usize, "state read out of range");
        let bytes = (count * 4) as vk::DeviceSize;
        let staging = self.create_buffer(
            bytes,
            vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let pool = self.create_command_pool()?;
        let cmd = self.allocate_command_buffer(pool)?;
        let fence = self.create_fence()?;
        let mut values = vec![0.0f32; count];
        unsafe {
            self.device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())?;
            self.device.cmd_copy_buffer(
                cmd,
                self.state.buffer,
                staging.buffer,
                &[vk::BufferCopy {
                    src_offset: (offset * 4) as vk::DeviceSize,
                    dst_offset: 0,
                    size: bytes,
                }],
            );
            self.device.end_command_buffer(cmd)?;
            self.submit(&[cmd], fence)?;
            self.device.wait_for_fences(&[fence], true, u64::MAX)?;
            let ptr = self
                .device
                .map_memory(staging.memory, 0, staging.size, vk::MemoryMapFlags::empty())?
                as *const f32;
            std::ptr::copy_nonoverlapping(ptr, values.as_mut_ptr(), count);
            self.device.unmap_memory(staging.memory);
            self.device.destroy_fence(fence, None);
            self.device.free_command_buffers(pool, &[cmd]);
            self.device.destroy_command_pool(pool, None);
            staging.destroy(&self.device);
        }
        Ok(values)
    }

    /// Zero every recurrent-state slot so generation can restart cleanly.
    pub fn reset_state(&mut self) -> Result<()> {
        let pool = self.create_command_pool()?;
        let cmd = self.allocate_command_buffer(pool)?;
        let fence = self.create_fence()?;
        unsafe {
            self.device.reset_fences(&[fence])?;
            let begin_info = vk::CommandBufferBeginInfo::default();
            self.device.begin_command_buffer(cmd, &begin_info)?;
            self.device.cmd_fill_buffer(cmd, self.state.buffer, 0, self.state.size, 0);
            self.device.end_command_buffer(cmd)?;
            self.submit(&[cmd], fence)?;
            self.device.wait_for_fences(&[fence], true, u64::MAX)?;
            self.device.destroy_fence(fence, None);
            self.device.free_command_buffers(pool, &[cmd]);
            self.device.destroy_command_pool(pool, None);
        }
        Ok(())
    }
}

/// Implicit layers installed by capture/overlay tools (OBS, Steam, game
/// launchers) frequently cause spurious `VK_ERROR_DEVICE_LOST` and GPU hangs
/// for pure-compute workloads.  Disable every implicit layer except official
/// validation layers before creating the instance; this mirrors what renderer
/// frontends do for headless compute.
fn disable_intrusive_layers(entry: &ash::Entry) {
    // The fence/command-buffer fixes removed the need for this workaround;
    // keep it available for diagnosing overlay-related device loss.
    if std::env::var("YUF_DISABLE_LAYERS").as_deref() != Ok("1") {
        return;
    }
    let Ok(layers) = (unsafe { entry.enumerate_instance_layer_properties() }) else {
        return;
    };
    let mut disable: Vec<String> = Vec::new();
    for layer in layers {
        let name = unsafe { CStr::from_ptr(layer.layer_name.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        let is_validation = name.contains("validation")
            || name.starts_with("VK_LAYER_LUNARG_")
            || name == "VK_LAYER_KHRONOS_validation";
        if !is_validation && name.starts_with("VK_LAYER_") {
            disable.push(name);
        }
    }
    if !disable.is_empty() {
        // SAFETY: single-threaded startup, before any other loader state is
        // touched; the loader reads this variable when creating the instance.
        unsafe {
            std::env::set_var("VK_LOADER_DISABLE_INST_LAYERS", disable.join(","));
        }
        eprintln!("[vulkan] disabled implicit layers: {}", disable.join(", "));
    }
}

impl Drop for ComputeContext {
    fn drop(&mut self) {
        unsafe {
            if self.fence != vk::Fence::null() {
                self.device.destroy_fence(self.fence, None);
            }
            if self.command_pool != vk::CommandPool::null() {
                self.device
                    .destroy_command_pool(self.command_pool, None);
            }
            for pipeline in self.pipelines.values() {
                self.device.destroy_pipeline(*pipeline, None);
            }
            if self.pipeline_layout != vk::PipelineLayout::null() {
                self.device
                    .destroy_pipeline_layout(self.pipeline_layout, None);
            }
            if self.descriptor_pool != vk::DescriptorPool::null() {
                self.device
                    .destroy_descriptor_pool(self.descriptor_pool, None);
            }
            if self.descriptor_set_layout != vk::DescriptorSetLayout::null() {
                self.device
                    .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            }
            if !self.work_map.is_null() {
                self.device.unmap_memory(self.work.memory);
            }
            self.weights.destroy(&self.device);
            self.work.destroy(&self.device);
            self.state.destroy(&self.device);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
