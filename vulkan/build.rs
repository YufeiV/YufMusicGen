//! Compiles GLSL compute shaders to SPIR-V with the pure-Rust `naga` frontend.
//! This keeps the build independent of the Vulkan SDK / glslc.

use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let shader_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("src/compute/shaders");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap()).join("spirv");
    fs::create_dir_all(&out_dir).unwrap();

    let entries = fs::read_dir(&shader_dir)
        .unwrap_or_else(|e| panic!("cannot read shader dir {}: {e}", shader_dir.display()));

    let mut compiled = 0;
    for entry in entries.flatten() {
        let path = entry.path();
        let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if file_name.starts_with('_')
            || path.extension().and_then(|e| e.to_str()) != Some("glsl")
        {
            continue;
        }
        let source = fs::read_to_string(&path).expect("read shader");
        let name = path.file_stem().unwrap().to_str().unwrap().to_string();
        let stage = if name.ends_with("_vert") {
            naga::ShaderStage::Vertex
        } else if name.ends_with("_frag") {
            naga::ShaderStage::Fragment
        } else {
            naga::ShaderStage::Compute
        };
        let spv = compile_glsl(&source, &name, stage).unwrap_or_else(|e| {
            panic!("failed to compile shader {}: {}", path.display(), e)
        });
        fs::write(out_dir.join(format!("{name}.spv")), &spv).unwrap();
        compiled += 1;
    }

    println!("cargo:rerun-if-changed=src/compute/shaders");
    println!("cargo:warning=compiled {compiled} shaders to SPIR-V");
}

fn compile_glsl(
    source: &str,
    entry: &str,
    stage: naga::ShaderStage,
) -> Result<Vec<u8>, String> {
    let module = naga::front::glsl::Frontend::default()
        .parse(&naga::front::glsl::Options::from(stage), source)
        .map_err(|e| e.to_string())?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|e| e.to_string())?;
    let mut writer = naga::back::spv::Writer::new(&naga::back::spv::Options {
        lang_version: (1, 0),
        ..Default::default()
    })
    .map_err(|e| e.to_string())?;
    let mut spv_words = Vec::new();
    writer
        .write(&module, &info, None, &None, &mut spv_words)
        .map_err(|e| e.to_string())?;
    let mut bytes = Vec::with_capacity(spv_words.len() * 4);
    for word in spv_words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    let _ = entry;
    Ok(bytes)
}
