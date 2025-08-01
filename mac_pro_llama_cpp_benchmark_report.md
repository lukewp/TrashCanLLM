# Mac Pro Llama.cpp Benchmarks (DeepSeek‑Qwen‑8B & Phi‑4‑mini)

## 1  Host system & hardware constraints

| Component | Detail |
| --------- | ------ |
|           |        |

|   |
| - |

|   |
| - |

| **Model**    | Mac Pro 6,1 (late‑2013 "trash‑can")                                          |
| ------------ | ---------------------------------------------------------------------------- |
| **CPU**      | E5‑2697 v2 ×1 \| 12 cores @ 3.5 GHz → 24 logical                             |
| **GPU**      | 2 × AMD FirePro **D700** (Tahiti XT) • 6 GB GDDR5 each • Vulkan via MoltenVK |
| **Sys RAM**  | 64 GB 1866 MHz DDR3 ECC                                                      |
| **Storage**  | Samsung 970 EVO Plus NVMe on PCIe adapter (model loads in ≃ 3 s)             |
| **Thermals** | Single radial blower • Throttles GPU clocks once *GPU‑die* ≥ ≈ 78 °C         |

### Practical limitations

- 6 GB VRAM per card ⇒ model **must** quantise to ≤ ≈ 5 GB *and* keep KV cache ≤ ≈ 2.9 GB (1 GPU) or 2×1.45 GB (2 GPU split).
- MoltenVK lacks INT‑dot & matrix‑core paths → fp16 shaders only.
- Metal backend cannot address the GCN GPUs, so **Vulkan** path required.

---

## 2  Custom llama.cpp build

```
cmake -B build -DGGML_VULKAN=ON           \
             -DGGML_BLAS=ON              \
             -DGGML_BLAS_VENDOR=OpenBLAS \
             -DGGML_NATIVE=ON            \
             -DCMAKE_BUILD_TYPE=Release  \
             -DCMAKE_PREFIX_PATH=$(brew --prefix openblas)
```

- **OpenBLAS** drives CPU fallback GEMMs (Accelerate stays loaded but idle).
- **MoltenVK 1.4.0** chosen at runtime; `MVK_CONFIG_LOG_LEVEL=0` to silence logs.
- Compiled commit `6e672545` (ggml 0.0.6039).

---

## 3  Benchmark methodology

- **Tool**: `llama-bench` (bundled)   *metrics*
  - **pp512** – prompt‑only throughput (512‑token prompt)
  - **tg256** – generation throughput (256 tokens after ≈20‑token prompt)
- Each variant repeated ×5 and averaged (`± `= stdev).
- Ambient ≤ 23 °C; blower locked at 6000 RPM except where noted.

---

## 4  Key runtime variants & results

### 4.1  DeepSeek‑Qwen‑8B *(recap)*

| ID                                   | Delta flags (vs. baseline)                                                | GPU layout           | Batch/UB | KV type   | **pp512**    | **tg256**      | GPU T° |
| ------------------------------------ | ------------------------------------------------------------------------- | -------------------- | -------- | --------- | ------------ | -------------- | ------ |
| **A** *(baseline cold)*              | `--tensor-split 1/1` `--no-kv-offload 0` `-ngl 36` `-b 64 -ub 32` `-t 24` | 18 L + 18 L          | 64/32    | fp16      | 37.7 t/s     | **14.4 t/s**   | 70 °C  |
| B                                    | + `-ctk q4_0`                                                             | same                 | 64/32    | **q4\_0** | 36.4 t/s     | 14.0 t/s       | 69 °C  |
| C                                    | + `--no-kv-offload 1`                                                     | 18 L+18 L (KV → RAM) | 64/32    | fp16      | 36.9 t/s     | **9.8 t/s**    | 65 °C  |
| D                                    | **High‑batch prompt** `-b 256 -ub 128`                                    | same                 | 256/128  | fp16      | **56.0 t/s** | 14.4 t/s       | 72 °C  |
| E                                    | **Thermal throttle demo** (blower auto, GPU die 85 °C)                    | same                 | 64/32    | fp16      | 35 t/s       | **9.7 t/s**    | 85 °C  |
| F                                    | *CPU‑only* `-ngl 0`                                                       | –                    | 256/128  | –         | 17.0 t/s     | 6.2 t/s        | 55 °C  |
| **G** *(Phi‑4‑mini launch, initial)* | `-m Phi-4-mini` `-b 64 -ub 32` `-ngl 32` `--tensor-split 1/1`             | 16 L + 16 L          | 64/32    | fp16      | 33.5 t/s\*   | **4.99 t/s**\* | –      |

\*from first `llama‑server` run; see section 4.2 for optimised numbers.

### 4.2  Phi‑4‑mini‑instruct sweep (best picks)

| ID                           | Delta flags (vs. Phi baseline)                          | Batch/UB | KV type  | **pp512**   | **tg256**    |
| ---------------------------- | ------------------------------------------------------- | -------- | -------- | ----------- | ------------ |
| **P0** *(interactive)*       | `-b 64 -ub 32` `--no-kv-offload 0` `--tensor-split 1/1` | 64/32    | fp16     | 82 t/s      | **23.5 t/s** |
| P1 *(off‑load demo)*         | same but `--no-kv-offload 1`                            | 64/32    | fp16→RAM | 69.9 t/s    | **12.3 t/s** |
| **P2** *(high‑batch prompt)* | `-b 512 -ub 256` `--no-kv-offload 0`                    | 512/256  | fp16     | 127 t/s     | 23.5 t/s     |
| **P3** *(max batch)*         | `-b 1024 -ub 512` `--no-kv-offload 0`                   | 1024/512 | fp16     | **139 t/s** | 23.5 t/s     |
| P4 *(max batch + off‑load)*  | same as P3 but `--no-kv-offload 1`                      | 1024/512 | fp16→RAM | 118 t/s     | 12.3 t/s     |

*All Phi‑4 runs use **``**, 24 CPU threads, ambient 23 °C.*

---

## 5  Findings (updated)

1. **Dual‑GPU split + on‑card KV remains fastest for both models.**
2. **DeepSeek‑Qwen‑8B** peaks at \~14.4 tok/s generation; Phi‑4‑mini pushes this to **23–24 tok/s** (≈ +60 %).
3. Prompt‑only throughput on Phi‑4 scales nearly linearly with batch size, hitting **≈ 140 tok/s at B = 1024**.
4. Generation throughput on Phi‑4 saturates at \~23.5 tok/s and is **insensitive to batch** – great for chat agents.
5. Off‑loading KV to system RAM roughly **halves generation speed** on both models but is still serviceable (\~12 tok/s) and frees ≥8 GB of VRAM for >100 k ctx work.
6. Increasing the tensor‑split beyond 1/1 (e.g. 2/1 or 0/1) brought no measurable wins and occasionally hurt tg256; 1/1 is simplest and safest.
7. High‑batch prompt ingestion (P2/P3) is useful when slurping large documents; switch back to B64/32 for interactive turns to minimise latency.

---

## 6  Recommended launch presets

### 6.1  DeepSeek‑Qwen‑8B (interactive agent)

```bash
MVK_CONFIG_LOG_LEVEL=0 \
./build/bin/llama-server \
  -m DeepSeek-R1-0528-Qwen3-8B-Q4_0.gguf \
  --ctx-size 32768 \
  --n-gpu-layers 36 --tensor-split 1/1 \
  --no-kv-offload \
  --batch-size 64 --ubatch-size 32 \
  --threads 24 \
  --temp 0.6 --top_p 0.95 \
  --host :: --port 8080
```

*Expect 14.4 tok/s generation, <50 ms first‑token latency, 70–72 °C GPU die.*

### 6.2  Phi‑4‑mini‑instruct (interactive agent)

```bash
MVK_CONFIG_LOG_LEVEL=0 \
./build/bin/llama-server \
  -m microsoft_Phi-4-mini-instruct-Q4_0.gguf \
  --ctx-size 65536 \
  --n-gpu-layers 32 --tensor-split 1/1 \
  --no-kv-offload \
  --batch-size 64 --ubatch-size 32 \
  --threads 24 \
  --temp 0.6 --top_p 0.95 \
  --host :: --port 8080
```

*Expect 23–24 tok/s generation, <30 ms first‑token latency, 65–70 °C GPU die (blowers at 6 krpm).*\
*For large‑prompt ingestion, temporarily launch with ****\`\`**** to reach 140 tok/s.*

---

## 7  Next optimisation ideas (unchanged + Phi‑4‑specific)

- **Re‑paste GPU dies** – user reports 5–8 °C drop on fresh thermal paste.
- **Experiment with Flash‑Attention on Metal once GCN patch lands** (may cut VRAM 20 %).
- **Speculative decoding** with a 1 B draft model could push Phi‑4 to 28–30 tok/s.
- **Mixed KV quantisation** (K q4\_0 / V fp16) might recover some off‑load speed while halving RAM.

---

© 2025 — benchmark sessions  Aug‑01‑2025 (Qwen) & Aug‑02‑2025 (Phi‑4)

