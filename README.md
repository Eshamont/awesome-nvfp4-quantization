# Awesome NVFP4 Quantization

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A curated list of papers, libraries, toolchains, and documentation for **NVFP4 quantization**.

## Contents

- [NVFP4 at a glance](#nvfp4-at-a-glance)
- [NVFP4 PTQ papers](#nvfp4-ptq-papers)
  - [PTQ methods](#ptq-methods)
  - [PTQ systems and applications](#ptq-systems-and-applications)
  - [Analysis and evaluation](#analysis-and-evaluation)
- [NVFP4 QAT and QAD](#nvfp4-qat-and-qad)
  - [QAT](#qat)
  - [QAD](#qad)
- [Libraries and toolchains](#libraries-and-toolchains)
  - [Quantization toolchains](#quantization-toolchains)
  - [Inference and serving](#inference-and-serving)
  - [Kernels and low-level libraries](#kernels-and-low-level-libraries)
- [Documentation and tutorials](#documentation-and-tutorials)

## NVFP4 at a glance

NVFP4 is a hardware-accelerated microscaling format introduced with NVIDIA Blackwell. It represents each value in FP4 E2M1 and uses two-level scaling: one FP8 E4M3 scale per 16-value block and one FP32 scale per tensor.

$$
\hat{x} = q_{\mathrm{E2M1}} \times s_{\mathrm{block}}^{\mathrm{E4M3}} \times s_{\mathrm{tensor}}^{\mathrm{FP32}}.
$$

| Property | NVFP4 specification |
| --- | --- |
| Element format | FP4 E2M1: 1 sign bit, 2 exponent bits, and 1 mantissa bit |
| E2M1 values | ±0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6 |
| Block size | 16 values |
| Block scale | One FP8 E4M3 scale per 16-value block |
| Tensor scale | One FP32 scale per tensor |
| Nominal storage | 4 + 8/16 = 4.5 bits/value, plus the tensor scale, padding, and metadata |
| Native hardware | NVIDIA Blackwell and later; exact support depends on the GPU and software backend |

NVFP4 differs from **MXFP4** and **NF4** in its format design. MXFP4 normally uses an E8M0 power-of-two scale for each 32-value block, while NF4 is a non-uniform codebook format commonly used for weight-only storage. See the NVIDIA [NVFP4 introduction](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/) and [TensorRT quantization schemes](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-schemes.html) for precise inference semantics.

## NVFP4 PTQ papers

### PTQ methods

- **[H-Scale: Hessian-Guided Scale Refinement for NVFP4 Sub-Byte LLM Inference](https://arxiv.org/abs/2608.28113)** [arXiv 2026]
- **[FOCUS: FP4 Optimization via Coupled-Relaxation and Dual-Granularity Scaling](https://arxiv.org/abs/2608.01847)** [arXiv 2026] [[Code](https://github.com/tencent/AngelSlim)]
- **[SharQ: Bridging Activation Sparsity and FP4 Quantization for LLM Inference](https://arxiv.org/abs/2606.26587)** [arXiv 2026] [[Code](https://github.com/actypedef/SharQ)]
- **[ScaleSweep: Accurate NVFP4 Post-Training Quantization of LLMs via Block Scale Initialization](https://arxiv.org/abs/2606.07618)** [arXiv 2026]
- **[OCGQuant: Outlier-Companion Grouping for NVFP4 Quantization](https://arxiv.org/abs/2609.00066)** [EMNLP 2026] [[Code](https://github.com/Eshamont/OCGQuant)]
- **[Grid Games: The Power of Multiple Grids for Quantizing Large Language Models](https://arxiv.org/abs/2605.12327)** [arXiv 2026] [[Code](https://github.com/IST-DASLab/GridGames)]
- **[SOAR: Scale Optimization for Accurate Reconstruction in NVFP4 Quantization](https://arxiv.org/abs/2605.12245)** [arXiv 2026] [[Code](https://github.com/steven-bao1/SOAR)]
- **[FAAR: Format-Aware Adaptive Rounding for NVFP4](https://arxiv.org/abs/2603.22370)** [arXiv 2026]
- **[MixFP4: Enhancing NVFP4 with Adaptive FP4/INT4 Block Representations](https://icml.cc/virtual/2026/poster/62778)** [ICML 2026]
- **[ARCQuant: Boosting NVFP4 Quantization with Augmented Residual Channels for LLMs](https://aclanthology.org/2026.acl-long.388/)** [ACL 2026] [[Code](https://github.com/actypedef/ARCQuant)]
- **[Four Over Six: More Accurate NVFP4 Quantization with Adaptive Block Scaling](https://arxiv.org/abs/2512.02010)** [arXiv 2025] [[Code](https://github.com/mit-han-lab/fouroversix)]
- **[WUSH: Near-Optimal Adaptive Transforms for LLM Quantization](https://icml.cc/virtual/2026/poster/63124)** [ICML 2026] [[Code](https://github.com/IST-DASLab/WUSH)]
- **[Search Your Block Floating Point Scales!](https://proceedings.mlsys.org/paper_files/paper/2026/hash/633b0e871a48d542280c3ad03928e60d-Abstract-Conference.html)** [MLSys 2026]
- **[Bridging the Gap Between Promise and Performance for Microscaling FP4 Quantization](https://proceedings.iclr.cc/paper_files/paper/2026/hash/b87bb4f6346d727b265088235e5bc389-Abstract-Conference.html)** [ICLR 2026] [[Code](https://github.com/IST-DASLab/FP-Quant)]
- **[FGMP: Fine-Grained Mixed-Precision Weight and Activation Quantization for Hardware-Accelerated LLM Inference](https://arxiv.org/abs/2504.14152)** [arXiv 2025]
- **[RaZeR: Pushing the Limits of NVFP4 Quantization with Redundant Zero Remapping](https://arxiv.org/abs/2501.04052)** [arXiv 2025] [[Code](https://github.com/abdelfattah-lab/NVFP4-RaZeR)]

### PTQ systems and applications

- **[ReSET: Accurate Latency-Critical NVFP4 Reasoning via Step-Aware Temperature Scaling](https://arxiv.org/abs/2606.13233)** [arXiv 2026] [[Code](https://github.com/aiha-lab/ReSET)]
- **[Mix-Quant: Quantized Prefilling, Precise Decoding for Agentic LLMs](https://arxiv.org/abs/2605.20315)** [arXiv 2026]
- **[PermuQuant: Lowering Per-Group Quantization Error by Reordering Channels for Diffusion Models](https://arxiv.org/abs/2605.09503)** [arXiv 2026] [[Code](https://github.com/yscheng04/PermuQuant)]
- **[6Bit-Diffusion: Inference-Time Mixed-Precision Quantization for Video Diffusion Models](https://arxiv.org/abs/2603.18742)** [arXiv 2026]
- **[BitDecoding: Unlocking Tensor Cores for Long-Context LLMs with Low-Bit KV Cache](https://2026.hpca-conf.org/details/hpca-2026-main-conference/92/BitDecoding-Unlocking-Tensor-Cores-for-Long-Context-LLMs-with-Low-Bit-KV-Cache)** [HPCA 2026] [[Code](https://github.com/OpenBitSys/BitDecoding)]

### Analysis and evaluation

- **[Characterizing the Impact of NVFP4 Quantization for Low-Power Edge AI Deployment](https://arxiv.org/abs/2606.06527)** [arXiv 2026]
- **[Diagnosing FP4 inference: a layer-wise and block-wise sensitivity analysis of NVFP4 and MXFP4](https://openreview.net/forum?id=xbDPiTgLeU)** [ICLR 2026 Workshop on Sci4DL]
- **[INT vs. FP: A Comprehensive Study of Fine-Grained Low-bit Quantization Formats](https://icml.cc/virtual/2026/poster/66697)** [ICML 2026]

## NVFP4 QAT and QAD

These methods optimize models for NVFP4 deployment through quantization-aware training (QAT) or quantization-aware distillation (QAD).

### QAT

- **[QUASAR: Lowering the Loss Floor of Quantization-Aware Training with Loss-Aware Reconstruction](https://arxiv.org/abs/2608.13966)** [arXiv 2026] [[Models](https://huggingface.co/QUASAR-QAT)]
- **[QUADS: Stabilizing NVFP4 Reinforcement Learning for MoE via QUantization-error Alignment across Dual Sides](https://arxiv.org/abs/2607.15810)** [arXiv 2026]
- **[ReQAT: Achieving Full-Precision Reasoning Accuracy with 4-bit Floating-Point Quantization-Aware Training](https://openreview.net/forum?id=aKY52fnzgc)** [ICML 2026] [[Code](https://github.com/aiha-lab/ReQAT)]
- **[Not All NVFP4 QAT Recipes Are Equal: How Architecture and Scale Shape Model Quality for Anomaly Segmentation](https://arxiv.org/abs/2605.27616)** [CVPR Workshops 2026]
- **[Attn-QAT: 4-Bit Attention With Quantization-Aware Training](https://openreview.net/forum?id=pqOlyvsmU3)** [ICML 2026]

### QAD

- **[Beyond Output Matching: Preserving Internal Geometry in NVFP4 LLM Distillation](https://arxiv.org/abs/2606.05682)** [arXiv 2026]
- **[Star Elastic: Many-in-One Reasoning LLMs with Efficient Budget Control](https://icml.cc/virtual/2026/poster/61782)** [ICML 2026]
- **[Quantization-Aware Distillation for NVFP4 Inference Accuracy Recovery](https://arxiv.org/abs/2601.20088)** [arXiv 2026] [[Project](https://research.nvidia.com/labs/nemotron/nemotron-qad/)] [[Code](https://github.com/NVIDIA/Model-Optimizer)]

## Libraries and toolchains

### Quantization toolchains

- [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) — PTQ, QAT/QAD, calibration, checkpoint packing, and export to TensorRT-LLM and other runtimes.
- [LLM Compressor](https://github.com/vllm-project/llm-compressor) — Hugging Face-compatible NVFP4 PTQ, GPTQ, KV-cache quantization, and compressed-tensors export for vLLM.
- [TorchAO](https://github.com/pytorch/ao) — prototype NVFP4 weight-only and dynamic W4A4 inference, plus QAT workflows in native PyTorch.
- [Intel Neural Compressor](https://github.com/intel/neural-compressor) — NVFP4 PTQ recipes with LLM Compressor-compatible export.
- [FP-Quant](https://github.com/IST-DASLab/FP-Quant) — format-aware PTQ and Micro-Rotated GPTQ for NVFP4/MXFP4.

### Inference and serving

- [NVIDIA TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) — optimized NVFP4 LLM/MoE execution and NVFP4 KV cache on supported Blackwell GPUs.
- [vLLM](https://github.com/vllm-project/vllm) — ModelOpt and compressed-tensors NVFP4 checkpoints, online NVFP4 paths, and multiple FP4 backends.
- [SGLang](https://github.com/sgl-project/sglang) — NVFP4 serving with selectable FlashInfer, CUTLASS, cuDNN, and TensorRT-LLM GEMM backends.

### Kernels and low-level libraries

- [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass) — CUDA C++ and CuTe DSL building blocks for Blackwell NVFP4 dense, grouped, and block-scaled GEMM.
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer) — NVFP4/MXFP4 GEMM, fused MoE, quantization, and serving kernels.
- [QuTLASS](https://github.com/IST-DASLab/qutlass) — low-bit CUDA kernels used by FP-Quant and related microscaling research.

## Documentation and tutorials

- [Introducing NVFP4 for Efficient and Accurate Low-Precision Inference](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/) — format overview and comparison with FP4/MXFP4.
- [TensorRT Quantization Schemes](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-schemes.html) — formal NVFP4 quantize/dequantize semantics.
- [TensorRT-LLM Quantization Guide](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/quantization.md) — model and hardware support, ModelOpt workflow, and KV cache.
- [Model Optimizer Hugging Face PTQ Guide](https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/hf_ptq/README.md) — calibration and export recipes.
- [vLLM Quantization Documentation](https://docs.vllm.ai/en/latest/features/quantization/) — supported checkpoint formats and execution backends.
- [TorchAO Quantized Inference](https://docs.pytorch.org/ao/stable/workflows/inference.html) — native PyTorch NVFP4 inference workflows.
- [Faster Diffusion on Blackwell with NVFP4, Diffusers, and TorchAO](https://pytorch.org/blog/faster-diffusion-on-blackwell-mxfp8-and-nvfp4-with-diffusers-and-torchao/) — end-to-end diffusion example and benchmarks.
- [Optimizing Long-Context Inference with NVFP4 KV Cache](https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/) — KV-cache accuracy and performance considerations.

If this list helps your work, consider starring the repository and sharing it with other researchers and practitioners.

## License

This repository is released under the [MIT License](LICENSE).
