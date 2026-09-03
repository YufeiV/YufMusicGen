//! Standalone Vulkan compute probe that shares NO code with the
//! yufmusicgen client: rspirv-built SPIR-V, plain ash.  Used to prove whether
//! a minimal, guaranteed-valid compute dispatch executes on the machine.

use anyhow::{Context, Result};
use ash::vk;
use rspirv::binary::Assemble;
use rspirv::dr::Builder;
use rspirv::spirv::{AddressingModel, Capability, ExecutionMode, ExecutionModel, MemoryModel};

fn main() -> Result<()> {
    let entry = unsafe { ash::Entry::load()? };
    let instance = unsafe {
        entry.create_instance(
            &vk::InstanceCreateInfo::default().application_info(
                &vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_2),
            ),
            None,
        )?
    };
    let physical = unsafe { instance.enumerate_physical_devices() }?[0];
    let properties = unsafe { instance.get_physical_device_properties(physical) };
    let name = unsafe { std::ffi::CStr::from_ptr(properties.device_name.as_ptr()) }
        .to_string_lossy();
    println!("[probe] device: {name}");

    let queue_family = unsafe { instance.get_physical_device_queue_family_properties(physical) }
        .iter()
        .position(|props| props.queue_flags.contains(vk::QueueFlags::COMPUTE))
        .with_context(|| "no compute queue")? as u32;
    let device = unsafe {
        instance.create_device(
            physical,
            &vk::DeviceCreateInfo::default().queue_create_infos(&[vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family)
                .queue_priorities(&[1.0])]),
            None,
        )?
    };
    let queue = unsafe { device.get_device_queue(queue_family, 0) };

    let words = if let Ok(path) = std::env::var("YUF_PROBE_SPV") {
        let bytes = std::fs::read(path).expect("read spv");
        let words: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        println!("[probe] using external SPIR-V ({} words)", words.len());
        words
    } else {
        // Build a trivial compute shader: empty main, LocalSize 1.
        let mut builder = Builder::new();
        builder.set_version(1, 0);
        builder.capability(Capability::Shader);
        builder.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);
        let void = builder.type_void();
        let fn_type = builder.type_function(void, vec![]);
        let main_id = builder
            .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_type)
            .expect("begin function");
        builder.begin_block(None).unwrap();
        builder.ret().unwrap();
        builder.end_function().unwrap();
        builder.entry_point(ExecutionModel::GLCompute, main_id, "main", &[]);
        builder.execution_mode(main_id, ExecutionMode::LocalSize, &[1, 1, 1]);
        let words = builder.module().assemble();
        println!("[probe] built SPIR-V ({} words)", words.len());
        words
    };
    println!("[probe] spirv words: {}", words.len());

    let module = unsafe {
        device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)?
    };
    let push_constant_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(60);
    let layout = unsafe {
        device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .push_constant_ranges(std::slice::from_ref(&push_constant_range)),
            None,
        )?
    };
    let pipeline = unsafe {
        let create_info = vk::ComputePipelineCreateInfo::default()
            .stage(
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::COMPUTE)
                    .module(module)
                    .name(c"main"),
            )
            .layout(layout);
        match device.create_compute_pipelines(
            vk::PipelineCache::null(),
            &[create_info],
            None,
        ) {
            Ok(pipelines) => pipelines
                .into_iter()
                .next()
                .with_context(|| "no pipeline")?,
            Err((_, err)) => anyhow::bail!("pipeline creation failed: {err}"),
        }
    };
    let pool = unsafe {
        device.create_command_pool(
            &vk::CommandPoolCreateInfo::default().queue_family_index(queue_family),
            None,
        )?
    };
    let cmd = unsafe {
        device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )?[0]
    };
    unsafe {
        device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())?;
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        device.cmd_dispatch(cmd, 1, 1, 1);
        device.end_command_buffer(cmd)?;
    }
    let fence = unsafe {
        device.create_fence(
            &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
            None,
        )?
    };
    unsafe {
        device.wait_for_fences(&[fence], true, u64::MAX)?;
        device.reset_fences(&[fence])?;
        device.queue_submit(
            queue,
            &[vk::SubmitInfo::default().command_buffers(&[cmd])],
            fence,
        )?;
        device
            .wait_for_fences(&[fence], true, u64::MAX)
            .context("wait_for_fences failed (GPU fault?)")?;
    }
    println!("[probe] independent compute dispatch SUCCEEDED");
    unsafe {
        device.destroy_fence(fence, None);
        device.destroy_command_pool(pool, None);
        device.destroy_pipeline(pipeline, None);
        device.destroy_pipeline_layout(layout, None);
        device.destroy_shader_module(module, None);
        device.destroy_device(None);
        instance.destroy_instance(None);
    }
    Ok(())
}
