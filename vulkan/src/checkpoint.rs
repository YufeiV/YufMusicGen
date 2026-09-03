//! Loading of exported YufMusicGen checkpoints (`.yuf` container).

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::Deserialize;

pub const MAGIC: &[u8; 4] = b"YUFM";
pub const FORMAT_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub head_size: usize,
    pub rosa_size: usize,
    pub dropout: f32,
    pub tie_embeddings: bool,
}

impl ModelConfig {
    pub fn low_rank(&self) -> usize {
        let raw = 2.5 * (self.d_model as f32).sqrt();
        ((raw / 32.0).round() as usize).max(1) * 32
    }

    pub fn validate(&self) -> Result<()> {
        if self.d_model != self.n_heads * self.head_size {
            bail!(
                "d_model {} must equal n_heads {} * head_size {}",
                self.d_model,
                self.n_heads,
                self.head_size
            );
        }
        if self.d_model < 32 || self.n_layers < 1 {
            bail!("model is too small to be useful");
        }
        if self.vocab_size < self.d_model {
            bail!("vocab_size {} looks wrong for d_model {}", self.vocab_size, self.d_model);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct TensorDesc {
    pub name: String,
    pub shape: Vec<usize>,
    pub offset: usize,
    pub count: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Header {
    pub format: String,
    pub version: u32,
    pub model_config: ModelConfig,
    pub midi: MidiHeader,
    pub source: Option<serde_json::Value>,
    pub lm_head_alias: Option<String>,
    pub tensors: Vec<TensorDesc>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MidiHeader {
    pub tokenization: String,
    pub midi_offset: usize,
    pub midi_vocab_size: usize,
    pub vocab: Vec<String>,
    /// MidiTok `tokenizer.json` serialization (config only; the REMI vocab is
    /// derived deterministically). Present in checkpoints exported after the
    /// tokenizer-json support was added.
    #[serde(default)]
    pub tokenizer_json: Option<String>,
}

pub struct Checkpoint {
    pub header: Header,
    pub data: Vec<f32>,
    tensor_index: HashMap<String, TensorDesc>,
}

impl Checkpoint {
    pub fn load(path: &Path) -> Result<Self> {
        let mut file = File::open(path)
            .with_context(|| format!("cannot open checkpoint {}", path.display()))?;
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)
            .context("cannot read checkpoint magic")?;
        if &magic != MAGIC {
            bail!(
                "{} is not a .yuf checkpoint (bad magic); convert it first with \
                 scripts/export_checkpoint.py",
                path.display()
            );
        }
        let mut version_bytes = [0u8; 4];
        file.read_exact(&mut version_bytes)?;
        let version = u32::from_le_bytes(version_bytes);
        if version != FORMAT_VERSION {
            bail!("unsupported .yuf version {version}");
        }
        let mut len_bytes = [0u8; 8];
        file.read_exact(&mut len_bytes)?;
        let header_len = u64::from_le_bytes(len_bytes) as usize;
        let mut header_json = vec![0u8; header_len];
        file.read_exact(&mut header_json)?;
        let header: Header = serde_json::from_slice(&header_json)
            .context("invalid .yuf header JSON")?;
        if header.format != "yufmusicgen-checkpoint" {
            bail!("unexpected .yuf payload format {:?}", header.format);
        }
        header.model_config.validate()?;

        let data_start = file.stream_position()?;
        let file_len = file.metadata()?.len();
        let expected = data_start + header.total_elements() as u64 * 4;
        if file_len != expected {
            bail!(
                "checkpoint size mismatch: file has {file_len} bytes, header expects {expected}"
            );
        }
        file.seek(SeekFrom::Start(data_start))?;
        let element_count = header.total_elements();
        let mut data = vec![0.0f32; element_count];
        {
            let byte_view = unsafe {
                std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, element_count * 4)
            };
            file.read_exact(byte_view).context("cannot read tensor payload")?;
        }

        let mut tensor_index = HashMap::new();
        for tensor in &header.tensors {
            tensor_index.insert(tensor.name.clone(), tensor.clone());
        }

        Ok(Self {
            header,
            data,
            tensor_index,
        })
    }

    pub fn tensor(&self, name: &str) -> Result<&[f32]> {
        let desc = self
            .tensor_index
            .get(name)
            .with_context(|| format!("checkpoint has no tensor {name}"))?;
        let start = desc.offset;
        let end = start + desc.count;
        Ok(&self.data[start..end])
    }

    pub fn tensor_dims(&self, name: &str) -> Result<Vec<usize>> {
        Ok(self
            .tensor_index
            .get(name)
            .with_context(|| format!("checkpoint has no tensor {name}"))?
            .shape
            .clone())
    }

    pub fn tensor_offset(&self, name: &str) -> Result<u32> {
        Ok(self
            .tensor_index
            .get(name)
            .with_context(|| format!("checkpoint has no tensor {name}"))?
            .offset as u32)
    }

    pub fn token_embedding(&self) -> Result<&[f32]> {
        self.tensor("token_embedding.weight")
    }

    pub fn head_embedding(&self) -> Result<&[f32]> {
        if self.header.model_config.tie_embeddings {
            self.token_embedding()
        } else {
            self.tensor("lm_head.weight")
        }
    }
}

impl Header {
    pub fn total_elements(&self) -> usize {
        self.tensors.iter().map(|t| t.count).sum()
    }
}
