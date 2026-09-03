# YufMusicGen Vulkan Client

一个用 Rust 实现的、基于 Vulkan 的原生 YufMusicGen 客户端。它用 Vulkan compute
shader 在 GPU 上完整运行 RWKV-7 TimeMix + ROSA + FFN 推理（不需要 PyTorch、
不需要 CUDA），并提供一个带交换链渲染的实时钢琴卷帘窗口。

## 功能

- `generate`：命令行生成 MIDI（文本 prompt、指定乐器、`--prompt-midi` 续写、
  temperature / top-p 采样）。
- `gui`：原生窗口客户端。按 **空格** 开始生成，实时把正在生成的音符画成钢琴
  卷帘（每个程序一个颜色，竖线为小节线），**Esc / Q** 退出。
- 模型推理完全在 Vulkan compute 上进行：约 450 次 dispatch / token，
  单 token 前向与 PyTorch 参考实现逐层精确一致。
- REMI（MidiTok v3）编解码用纯 Rust 实现，与 MidiTok 3.0.6 的输出逐 token 一致
  （编码、量化、解码、MIDI 读写都有对照测试）。

## 构建

需要 Rust 工具链和可用的 Vulkan 驱动（任何支持 compute 的 GPU，NVIDIA /
AMD / Intel 均可）。不需要 Vulkan SDK / glslc：构建脚本用纯 Rust 的
`naga` 把 GLSL 编译成 SPIR-V。

```bash
cd vulkan
cargo build --release
```

## 使用

### 1. 导出 checkpoint

客户端读取自包含的 `.yuf` 格式（PyTorch 权重 + 词表的扁平二进制），需要用仓库
里的 Python 环境把 `.pt` 转换一次：

```bash
python vulkan/scripts/export_checkpoint.py \
  --checkpoint checkpoints/posttrain-step-00037000.pt \
  --output vulkan/model.yuf
```

转换只读一次；之后 Rust 客户端完全不需要 Python。

### 2. 生成 MIDI

```bash
vulkan/target/release/yufmusicgen-vulkan generate \
  --checkpoint vulkan/model.yuf \
  --prompt "cinematic piano, slow tempo, emotional strings" \
  --steps 2048 --temperature 0.9 --top-p 0.95 \
  --output outputs/cinematic.mid
```

常用参数与 Python CLI 一致：`--instrument piano|violin|40`、`--instrument-only`、
`--prompt-midi seed.mid --prompt-max-tokens 512`、`--seconds 20`、`--seed N`。

### 3. 图形客户端

```bash
vulkan/target/release/yufmusicgen-vulkan gui \
  --checkpoint vulkan/model.yuf \
  --prompt "lofi hip hop, warm piano" --steps 1024
```

窗口打开后按空格开始生成；音符实时出现，窗口标题显示进度。

## 数值一致性

推理的数学与 `YufMusicGen` 的**参考递归**（`model._torch_recurrence` /
`_reference_recurrence`，即 CUDA kernel 的实现语义）一致。逐中间量、逐 logits
与 PyTorch 对齐（见 `examples/compare_logits.rs`，贪心采样下整条 token 序列一致）。

> 注意：仓库 Python 侧 `use_rosa_scan=True` 的 `torch.scan` 路径缺少
> `read_gate` 的 sigmoid，与参考递归/CUDA kernel 语义不一致（仓库自身的一个
> bug）。本客户端遵循参考递归，因此与 `use_rosa_scan=False` 的 PyTorch 输出一致。

## 实现结构

```text
vulkan/
  build.rs                 # 用 naga 把 GLSL 编译成 SPIR-V（无需 glslc）
  scripts/export_checkpoint.py   # .pt -> .yuf
  src/
    checkpoint.rs          # .yuf 加载（模型配置、词表、张量索引）
    compute/
      mod.rs               # Vulkan 实例/设备/缓冲/管线/描述符/同步
      model.rs             # 单 token 前向的计算图 + CPU 侧越界校验
      shaders/*.glsl       # embed/linear/layernorm/mix/timemix/rosa/ffn/copy
      shaders/roll_*.glsl  # GUI 顶点/片元着色器
    remi.rs                # REMI (MidiTok v3) 编解码
    midi.rs                # SMF 读写
    sampler.rs             # temperature / top-p 采样
    generation.rs          # 条件构建 + 生成循环
    gui.rs                 # winit + Vulkan 交换链钢琴卷帘
  tests/remi_tests.rs      # 与 MidiTok 参考 fixture 的对照测试
  examples/compare_logits.rs  # 与 PyTorch 的 logits/贪心序列对比工具
```

推理时每个 token 只提交一次命令缓冲（初始化时录制），内部通过 compute dispatch
逐层执行；所有缓冲区偏移在录制时做 CPU 侧校验，任何越界都会在提交 GPU 前报错。

## 调试开关

- `YUF_DEVICE=<名称子串或索引>`：选择物理设备（默认第一个）。
- `YUF_DUMP=<region,...>`：导出中间量（`ln0,r,k,v,w,a,g,o,h1,final,logits,...`）。
- `YUF_DEBUG_MAX_DISPATCH=N`：限制 dispatch 数量（定位问题用）。
- `YUF_AUTO_START=1`：GUI 打开后自动开始生成。
- `YUF_DISABLE_LAYERS=1`：创建实例前禁用隐式 layer（OBS/Steam 注入导致
  device lost 时的排查手段）。

## 已知问题

- 如果显卡驱动被反复触发 TDR（设备丢失），先彻底关机/重启或重装驱动再运行；
  客户端默认只做合法、有界的内存访问，并且所有偏移在 GPU 执行前由 CPU 校验。

