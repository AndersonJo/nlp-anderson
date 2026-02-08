# MXFP4, Hopper, BF16, and Why the Error Happened

This is a technical memo based on a real MXFP4 kernel failure in a GPT-OSS + Unsloth setup.  
Sentences are short, but the key ideas are intact.

---

## Writing Plan

1. **Problem definition**: summarize what failed and under which conditions  
2. **Core concepts**: focus on MXFP4, Hopper, and BF16  
3. **Keyword list**: precision, architecture, and kernel terms  
4. **Root cause**: explain why the Hopper-only path failed  
5. **Fix strategy**: show the fastest practical workaround  

---

## Problem Summary

The error message says "only Hopper swizzling is supported."  
This means the runtime entered a **Hopper-only kernel path**, but the environment did not satisfy the Hopper assumptions.  
The safest workaround is to **avoid the MXFP4 path and use a BF16 checkpoint**.

---

## Core Concepts (short but essential)

### MXFP4
- A **mixed-precision 4-bit format**.  
- It compresses weights to reduce **memory and bandwidth**.  
- It requires **special kernels** and **scale handling**.  

### Hopper
- NVIDIA **H100/Hopper generation**.  
- Features **TMA (async memory movement)** and **FP8 Tensor Cores**.  
- Some **memory swizzle** paths only work on Hopper.  

### BF16
- A **16-bit float with wider exponent range** than FP16.  
- It is a stable default for **training and inference**.  
- It runs broadly without special hardware-only paths.  

---

## Keyword List (concise)

### Precision and formats
- **FP32**: standard float, slow but stable.  
- **TF32**: FP32 compromise, NVIDIA-specific.  
- **FP16**: faster and smaller, narrower range.  
- **BF16**: safer range, common default.  
- **FP8 (E4M3/E5M2)**: key low-precision format on Hopper+.  
- **MXFP4**: extreme 4-bit format, kernel-dependent.  
- **INT8/INT4**: integer quantization, larger accuracy loss.  
- **NF4**: 4-bit variant used in QLoRA.  

### Architectures
- **Ampere (A100)**: mature FP16/BF16 platform.  
- **Ada (L40, etc.)**: inference-oriented, partial FP8 support.  
- **Hopper (H100)**: TMA, FP8, swizzle-optimized.  
- **Blackwell (B100/GB200, etc.)**: FP8/FP4 focus, early cycle.  

### Kernels and acceleration
- **Triton**: custom GPU kernel DSL, performance-sensitive.  
- **GEMM**: matrix multiply, core of model compute.  
- **Swizzle**: memory layout optimization, hardware-dependent.  
- **TMA**: Hopper-only memory transfer unit.  

### Model/runtime
- **MoE**: activates a subset of experts.  
- **Router/Experts**: selection and routing inside MoE.  
- **KV Cache**: core to decoding speed.  
- **LoRA/QLoRA**: lightweight finetuning.  

---

## Precision/Format ↔ Kernel Path (text graph)

This is a directory-style view of common mappings.  
Actual paths depend on library versions.

```
Precision/Format
├─ FP32
│  └─ GEMM (CUDA/cuBLAS default, dtype="float32")
├─ TF32
│  └─ GEMM (Tensor Core path, dtype="float32" + TF32 enabled)
├─ FP16
│  └─ GEMM (Tensor Core path, dtype="float16")
├─ BF16
│  └─ GEMM (Tensor Core path, dtype="bfloat16", stable default)
├─ FP8 (E4M3/E5M2)
│  ├─ GEMM (FP8 kernels, dtype="float8_e4m3fn"/"float8_e5m2", Hopper-optimized)
│  └─ Swizzle/TMA (Hopper-only optimization)
├─ FP4 (NVFP4)
│  ├─ GEMM (FP4 kernels, NVFP4 format, dtype="float4"/"nvfp4", Blackwell-centered)
│  └─ Micro-tensor scaling (Blackwell-only flavor)
├─ MXFP4
│  ├─ GEMM (MXFP4 kernels, MXFP4 format)
│  └─ Swizzle/TMA (often assumes Hopper)
└─ INT8/INT4/NF4
   ├─ GEMM (int kernels, dtype="int8"/"int4", includes NF4)
   └─ Dequant (scale restore path)
```

The key idea: **precision choice is kernel path choice**.  
If that choice does not match the hardware, errors follow.

---

## Why the Hopper-only swizzle error appears

1. **MXFP4 often routes to Triton-only kernels.**  
2. Those kernels may **assume Hopper swizzle behavior**.  
3. If the environment is not Hopper, it **fails at compile time**.  
4. That is why it often crashes on the first forward call.  

The core issue is **kernel path selection**.

---

## What a "kernel path" means (more detail)

A kernel path is the **execution branch that chooses one concrete kernel implementation**.  
On the surface you call `matmul`, but under the hood multiple kernels compete.  

### Signals that change the kernel path
- **Precision**: BF16/FP16/FP8/MXFP4 select different kernels.  
- **GPU generation**: Ampere/Hopper/Blackwell enable different paths.  
- **Library versions**: Triton/Transformers/Unsloth combos shift routing.  
- **Options and flags**: `load_in_4bit`, `dtype`, `use_cache` can flip paths.  

### Why this matters
- Kernels make **strong compile-time assumptions**.  
- If assumptions are wrong, they **fail early**.  
- That is why the same code breaks by just changing the GPU.  

### One-line summary
Kernel paths are **performance shortcuts**,  
and they become **dead ends** when the hardware does not match.  

---

## Practical Fix Strategy

### 1) Fastest workaround
- **Use a BF16 checkpoint instead of MXFP4.**  
- The kernel path becomes simpler and far more compatible.  

### 2) Reproduction signals
- If it fails on the **first forward**, it is likely a kernel issue.  
- If logs mention **swizzle / Triton / matmul / mxfp**, the odds are high.  

### 3) Long-term approach
- Non-Hopper hardware needs a **fallback path**.  
- Version combos can toggle swizzle options.  
- The safest move is to **simplify the precision path**.  

---

## Example Code (most practical choice)

**Recommended stack: PyTorch + Transformers + Unsloth**  
Triton is low-level and overkill for a blog example.  
This is the most common production stack.

```python
from unsloth import FastLanguageModel
from transformers import TextStreamer

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gpt-oss-20b-BF16",
    dtype="bfloat16",
    max_seq_length=2000,
    load_in_4bit=False,
    full_finetuning=False,
    low_cpu_mem_usage=True,
    device_map="cuda",
)

messages = [{"role": "user", "content": "Solve x^5 + 3x^4 - 10 = 3."}]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
    reasoning_effort="medium",
).to("cuda")

_ = model.generate(
    **inputs,
    max_new_tokens=512,
    streamer=TextStreamer(tokenizer),
    use_cache=True,
)
```

The key is **using a BF16 checkpoint instead of MXFP4**.  
That single change switches the kernel path and avoids the failure.  

---

## One-line Summary

MXFP4 is fast but **hardware-dependent**,  
BF16 is slower but **predictable and stable**.  

In practice, **get a working path first**,  
then re-enable low-precision optimizations only when needed.  
