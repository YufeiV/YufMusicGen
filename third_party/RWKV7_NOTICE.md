# RWKV-7 CUDA attribution

YufMusicGen includes a modified training kernel derived from the official
RWKV-7 implementation in:

https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v7/train_temp/cuda

The upstream project is distributed under the Apache License, Version 2.0.
The files in `yufmusicgen/cuda/` are modified for YufMusicGen, including the
operator namespace, FP32 build path, tensor validation contract, and lazy
loading wrapper. The upstream license text is available at:

https://github.com/BlinkDL/RWKV-LM/blob/main/LICENSE

Copyright and license notices from the upstream source are retained here for
the vendored derivative kernel.
