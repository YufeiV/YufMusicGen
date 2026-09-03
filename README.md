# YufMusicGen

YufMusicGen 是一个可运行的 MIDI 音乐生成基线工程，把 MIDI 数据、MidiTok 离散 token、循环式语言模型训练和推理串成一个完整闭环。

核心模型由两部分组成：

- RWKV-7 风格 `TimeMix`：使用 token shift、内容相关衰减、receptance 和 per-head matrix state。序列长度上的计算是线性的，推理不需要 Transformer KV cache。
- ROSA memory：这里将 ROSA 定义为 `Recurrent Orthogonal State Augmentation`，通过 Householder reflection 对一个小型 recurrent memory 做正交更新，再以门控方式写入和读取。

默认 codec 使用 [MidiTok](https://github.com/Natooz/MidiTok)（v3，基于 symusic），支持 REMI、TSD、MIDILike、CPWord、Structured、Octuple 等 tokenization，并可对数据集训练 BPE 词表。MIDI 文件被编码为单条 token 流，文本条件（caption）以 UTF-8 字节 token 拼接在序列前部。

## RWKV-7 CUDA 算子

仓库内置了从官方 `RWKV-v7/train_temp/cuda` 改造的 `rwkv7_clampw` 训练算子，默认在 CUDA、FP32、无外部 recurrent state 的长序列前向中优先使用。它包含 fused forward/backward，并按 16 token chunk 保存反向所需的状态；单 token 推理和 CPU 会使用同算法的 PyTorch 回退实现。

首次运行会通过 `torch.utils.cpp_extension.load` 编译 `yufmusicgen/cuda/` 下的源码，因此 CUDA 训练环境需要匹配的 CUDA Toolkit、`nvcc` 和 C++ 编译器。编译失败会自动回退并给出 warning，也可以显式关闭：

```bash
python scripts/pretrain.py --dataset data/processed --disable-cuda-kernel --device cpu
```

源码来源和 Apache-2.0 归属见 [third_party/RWKV7_NOTICE.md](third_party/RWKV7_NOTICE.md)。

## 安装

需要 Python 3.10+、PyTorch、NumPy 和 MidiTok（随项目一起安装）：

```bash
uv sync
# 或
pip install -e .
```

如果要启用 CUDA kernel，请先安装与驱动匹配的 CUDA 版 PyTorch；例如使用 uv：

```bash
uv pip install --python .venv\Scripts\python.exe --torch-backend=cu132 torch
```

## 1. MIDI 预处理

目录输入会递归发现 `.mid/.midi/.smf`。如果 MIDI 旁边有同名 `.txt`，文件内容会作为文本条件。也可以输入 JSONL：

```json
{"midi": "song.mid", "text": "instrumental lo-fi hip hop, warm piano"}
```

运行（`--vocab-size` 大于 0 时会先在全部语料上训练 BPE 词表）：

```bash
python scripts/preprocess_midi.py \
  --input data/raw_midi \
  --output data/processed \
  --tokenization REMI \
  --vocab-size 20000 \
  --use-tempos \
  --workers 8
```

目录扫描使用流式遍历，只收集 MIDI 文件，不会为全部文件建列表；编码阶段默认用 `min(8, CPU核数)` 个进程并行，`--workers 1` 可强制单进程。文件非常多且不需要 BPE 时，先用 `--vocab-size 0` 跑通再考虑训练词表。

输出：

```text
data/processed/
  manifest.jsonl
  codec.json
  tokenizer.json
  miditok/tokenizer.json
  tokens/00000000.npy
```

`manifest.jsonl` 是训练入口；`codec.json` 记录 MidiTok 配置，`miditok/tokenizer.json` 保存完整 tokenizer（含 BPE 词表），会被写入 checkpoint，推理时不需要手动重建。

## 2. 预训练

默认模型约 0.1B 参数（14 层、640 hidden、16 heads、ROSA 128），适合 8GB 显存的 GPU 以 batch=2 训练。CPU smoke run 可以使用 `configs/tiny.json` 中的规模对应参数：

```bash
python scripts/pretrain.py \
  --dataset data/processed \
  --output checkpoints/pretrain \
  --device cuda \
  --d-model 640 \
  --n-layers 14 \
  --n-heads 16 \
  --head-size 40 \
  --rosa-size 128 \
  --sequence-length 2048 \
  --batch-size 2 \
  --max-steps 10000
```

每个 batch 都从零初始化 recurrent state；模型在一个 token 序列内部按时间推进 state。checkpoint 同时保存模型、优化器、训练步数、已消费样本数、数据集规模和 codec 配置。

### 断点续训

从 `pretrain-step-00004000.pt` 这样的 checkpoint 续训时，不仅模型/优化器/步数会恢复，数据流也会从训练集的中间继续，而不是从头重放：

```bash
python scripts/pretrain.py \
  --dataset data/processed \
  --init-checkpoint checkpoints/pretrain/pretrain-step-00004000.pt \
  --output checkpoints/pretrain \
  --device cuda \
  --max-steps 5000
```

- checkpoint 记录已消费的样本数 `samples_seen`；数据流按 `--seed` 生成确定性的 epoch shuffle 顺序（每轮一个 permutation），续训时直接跳过已消费样本，继续原来的顺序。
- 随机裁剪按（样本，epoch）确定，因此续训前后看到的窗口完全一致。
- 学习率 schedule（warmup / cosine）按全局步数继续，不会因续训而重新开始。
- 只有同一阶段（pretrain→pretrain、posttrain→posttrain）才续接数据流；用 pretrain checkpoint 启动 posttrain 时，数据从头开始。
- 旧版 checkpoint 没有 `samples_seen` 时会按 `step × batch_size × grad_accumulation` 推算已消费样本数；若续训时改变了 batch size 导致样本数与新 batch 不对齐，会回退到最近的 batch 边界并重放少量样本（打印 warning）。

## 3. 后训练

后训练使用同一份 manifest，但只对 MIDI token 和 `EOS` 计算 loss。这样文本提示词作为条件，训练目标集中在音乐响应部分：

```bash
python scripts/posttrain.py \
  --dataset data/processed \
  --init-checkpoint checkpoints/pretrain/pretrain-step-00010000.pt \
  --output checkpoints/posttrain \
  --device cuda \
  --learning-rate 5e-5 \
  --max-steps 3000
```

`--max-steps` 表示本次后训练新增的步数；上例从 10000 步继续训练 3000 步，最终 checkpoint 名称为 `posttrain-step-00013000.pt`。

更适合后训练的数据是带 caption 的 JSONL 或带同名 `.txt` 的 MIDI 目录。预训练数据没有 caption 时仍然可以运行，但文本条件不会得到有效监督。

## 3.5 PEFT 微调（LoRA / PiSSA / MiSS / State tuning）

> 注：由于微调实现有问题，已经回退删除了

项目参考 [RWKV-PEFT](https://github.com/Joluck/RWKV-PEFT) 内置了自包含的 PEFT 实现（`yufmusicgen/peft.py`），不依赖第三方 `peft` 库，LoRA / MiSS 适配器是可合并回基座权重的普通 `nn.Module`，State tuning 则把每层循环初始状态变成可学习参数。预训练和后续训练入口都支持：

```bash
python scripts/posttrain.py \
  --dataset data/processed \
  --init-checkpoint checkpoints/pretrain/pretrain-step-00010000.pt \
  --output checkpoints/posttrain-lora \
  --device cuda \
  --learning-rate 1e-4 \
  --max-steps 3000 \
  --peft lora \
  --peft-config '{"r":8,"lora_alpha":32,"lora_dropout":0.05}'
```

`--peft` 可选 `lora`、`miss`、`state`（默认 `none` 为全量微调）。`--peft-config` 是 JSON 字符串，各方法常用配置：

| 方法 | 示例 | 说明 |
| --- | --- | --- |
| LoRA | `{"r":8,"lora_alpha":32,"lora_dropout":0.05}` | 低秩适配 `time_mix` 的 receptance/key/value/output 投影 |
| PiSSA | `{"r":8,"lora_alpha":8,"pissa_init":true}` | LoRA 的 SVD 初始化变体；`alpha == r` 时零步输出与原始模型一致 |
| MiSS | `{"r":16}` | 共享分片块（原 Bone/DiSHA），参数约为同秩 LoRA 的一半；`init_weights:"mini"` 可配 `mini_r` 进一步压缩 |
| State tuning | （无需配置） | 只训练每层 TimeMix 的初始 memory 与 ROSA 初始状态，基座权重全部冻结 |

`target_modules` 默认为 `["time_mix.receptance","time_mix.key","time_mix.value","time_mix.output"]`（与 RWKV-PEFT 默认一致）；可用 `"all-linear"` 覆盖 TimeMix / ROSA / FFN 全部投影，也可以按名字精确指定，例如 `{"target_modules":["ffn_in","ffn_gate","ffn_out"]}`。`modules_to_save` 可以让指定模块不套适配器但保持可训练，例如 `{"modules_to_save":["token_embedding"]}`。

训练信息会随 checkpoint 保存：`save_checkpoint` 写入 `peft_config` 字段，续训时自动恢复相同的适配器布局（即使命令行没有重复传 `--peft`），`scripts/generate.py` 加载 PEFT checkpoint 时也会自动重建适配器，无需额外参数。合并适配器到基座权重（推理部署用）：

```python
import torch
from yufmusicgen.config import ModelConfig, dataclass_from_dict
from yufmusicgen.model import YufMusicGen
from yufmusicgen.peft import PeftConfig, apply_peft, merge_adapters

payload = torch.load("checkpoints/posttrain-lora/posttrain-step-00013000.pt", map_location="cpu")
model = YufMusicGen(dataclass_from_dict(ModelConfig, payload["model_config"]))
apply_peft(model, PeftConfig.from_dict(payload["peft_config"]))
model.load_state_dict(payload["model"], strict=True)
merge_adapters(model)                      # LoRA/MiSS 增量写回 base weight
torch.save({"model": model.state_dict()}, "merged.pt")
```

注意：
- State tuning 训练时初始状态是学习参数，TimeMix 走 PyTorch 递推路径（CUDA fused kernel 仅在无初始状态时启用），速度比全量/LoRA 慢，适合小数据量风格微调。
- PiSSA 要求 `lora_alpha == r` 才能在第一步保持原模型输出；否则等效于带缩放的低秩起点。
- MiSS 的 `r` 建议取偶数，且应能整除目标投影的输入维度（默认目标输入维度都是 `d_model`，`--d-model 640` 时 `r=32/64` 均可整除）。

## 4. 推理

```bash
python scripts/generate.py \
  --checkpoint checkpoints/posttrain/posttrain-step-00013000.pt \
  --prompt "cinematic piano, slow tempo, emotional strings" \
  --steps 2048 \
  --temperature 0.9 \
  --top-p 0.95 \
  --output outputs/cinematic.mid
```

输出为标准 `.mid` 文件，可用任意 DAW / 播放器打开。也可以用 `--seconds` 按估算时长生成（内部按约 20 token/秒换算）：

```bash
python scripts/generate.py --checkpoint checkpoints/posttrain/posttrain-step-00013000.pt --prompt "ambient synth" --seconds 20 --output outputs/ambient.mid
```

### WebUI（Gradio）

同样参数也可以在图界面里使用，支持文本提示、乐器选择、Prompt MIDI 上传、采样参数调节，并显示钢琴卷帘预览：

```bash
uv sync --extra webui
python scripts/generate_webui.py            # http://127.0.0.1:7860
# 或
yufmusicgen-webui --port 7860 --share
```

页面会从 `checkpoints/` 自动扫描可选 checkpoint，也可以手动输入路径。

### 指定乐器

`--instrument` 接受 GM 乐器名或程序号（0-127），会在条件序列里追加对应的 `Program` token，让模型以该乐器作为主声部生成：

```bash
python scripts/generate.py --checkpoint checkpoints/posttrain/posttrain-step-00013000.pt --prompt "sad ballad" --instrument violin --output outputs/violin.mid

python scripts/generate.py --checkpoint checkpoints/posttrain/posttrain-step-00013000.pt --prompt "lofi piano" --instrument 0 --output outputs/piano.mid
```

常用别名：`piano`、`violin`、`flute`、`guitar`、`drums`（鼓组，program -1）等；用 `--list-instruments` 可查看全部 128 个 GM 乐器。想强制输出只用该乐器（屏蔽其他 `Program` token），加 `--instrument-only`；注意在 BPE 词表下屏蔽是近似的。

### 用已有 MIDI 作为 Prompt

`--prompt-midi` 把现有 MIDI 编码成 token 作为条件前缀，模型从曲子结尾继续生成，输出会包含 prompt 部分 + 续写内容：

```bash
python scripts/generate.py --checkpoint checkpoints/posttrain/posttrain-step-00013000.pt --prompt-midi seeds/my_melody.mid --steps 1024 --output outputs/continued.mid
```

`--prompt-max-tokens` 控制保留的 prompt token 数（默认 512，超出时保留结尾片段）；可以同时指定 `--instrument` 和 `--prompt-midi`，乐器 `Program` token 会放在 prompt 之后、生成点之前，表示“继续这段音乐，并用该乐器生成后续部分”。

## Token 布局

```text
BOS + UTF-8 text bytes + SEP + [Program token] + [prompt MIDI tokens] + generated MidiTok tokens + EOS
```

词表固定为：`PAD/BOS/EOS/SEP`、256 个 byte token，以及偏移后的 MidiTok 词表（id 从 260 开始）。生成时只允许采样 MIDI token 和 `EOS`，避免模型输出文本字节破坏可解码性。

## 目录结构

```text
yufmusicgen/
  codec.py          # MidiTok codec（编码/解码/BPE/持久化）
  midi_io.py        # symusic 的 MIDI 读写与时长/截断
  tokenizer.py      # 文本/MIDI token 布局
  model.py          # RWKV-7 风格 TimeMix + ROSA
  data.py           # JSONL manifest 与 batch
  training.py       # 预训练/后训练共享循环
  cli/              # 四个命令行入口
scripts/            # 可直接执行的薄包装脚本
configs/            # 示例配置
```

## 工程边界

这是一个从数据到推理都能运行的 baseline。默认 MidiTok 配置是 REMI + 单 token 流（`one_token_stream_for_programs=True`）；要换 tokenization 或 BPE 规模，重新跑预处理即可，不需要改 RWKV/ROSA 的 recurrent model API。

## 验证

安装开发依赖后运行：

```bash
pip install -e .[dev]
python -m pytest
```

要验证 CUDA kernel 的前向和反向数值一致性，在已配置 `nvcc`、MSVC 和 GPU 的环境中运行：

```bash
set YUFMUSICGEN_TEST_CUDA=1
python -m pytest tests/test_rwkv7_cuda.py -q
```
