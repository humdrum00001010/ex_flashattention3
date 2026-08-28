# 2xH100 results

Run: 2026-08-28, 2x NVIDIA H100 80GB HBM3 SXM, NV18, CUDA 13.0,
cuDNN 9.12, XLA 0.10.0, Elixir 1.20.2, OTP 29.0.2.

## Correctness

- GPU integration: passed single-GPU and TP2 BF16/FP16 forward, backward,
  sharding, all-reduce placement, and resident-buffer re-entry.
- CPU preflight: 6 passed, 2 GPU tests excluded.

## Forward and backward throughput

Shape: causal `{batch=4, sequence=2048, q_heads=24, kv_heads=4, dim=256}`.
Chain length: 64. `Forward+backward` uses `3 * forward FLOPs`.

| Precision | Operation | 1 GPU PFLOP/s | TP2 PFLOP/s | TP2 range | Scaling |
| --- | --- | ---: | ---: | ---: | ---: |
| BF16 | Forward | 0.70761 | 1.37640 | 1.36585-1.37726 | 1.945x |
| BF16 | Forward+backward | 0.33970 | 0.65972 | 0.65571-0.66322 | 1.942x |
| FP16 | Forward | 0.65509 | 1.28723 | 1.28360-1.36840 | 1.965x |
| FP16 | Forward+backward | 0.33922 | 0.65446 | 0.65304-0.65714 | 1.929x |

Ranges are slow-to-fast throughput from the timing p90 and p10 samples.

## 99% BF16 forward result

Target: 1.42 PFLOP/s. The 99% gate is 1.4058 PFLOP/s.

| Calls per executable | TP2 PFLOP/s | Range | Target fraction |
| ---: | ---: | ---: | ---: |
| 1 | 0.38752 | 0.36042-0.41564 | 27.290% |
| 64 | 1.37640 | 1.36585-1.37726 | 96.929% |
| 256 | 1.40670 | 1.40058-1.40794 | 99.063% |
| 512 | **1.41490** | 1.41035-1.41606 | **99.641%** |

The chain-512 slow-side sample is 1.41035 PFLOP/s, or 99.320%.

## Proof

- StableHLO contained 512 FA3 calls before and after optimization.
- Final TP2 thunks: one `kCommandBuffer` containing 512 `kCustomCall` thunks.
- TP2 local shapes were 12 Q heads and 2 KV heads per rank, versus 24/4 on
  one GPU.
- EXLA waits for PjRt output readiness; throughput includes device completion
  and one execution boundary per chain.
- Backward workspaces are compiler-visible; the current handler contains no
  `cudaMallocAsync` call.

## Reproducibility

- Nx: `37901d749105076e2882fdea89a6e18393eecc0f`
- FA3: `0251105a2fb19d2957484b7f023cd8c115286ced`
- CUTLASS: `7127592069c2fe01b041e174ba4345ef9b279671`
- `libfa3_xla.so`: `16c2b4f57b462eb66a4e207ad5a440b491d8bc768fdf6c9f2e087850dc28d599`
- HLO before optimization: `8d5dcc15e8406661656ba2914280811fc06aa1e0d4bf9043349c2eccb5b400e5`
- HLO after optimization: `3731d049d39383128fa4dff23a4b7bc9505e08872f4c745bdebbe4edb25daa60`
- Final thunk sequence: `f90c0d359a5f48b319e53e598b0662dc17eef1563632fb00d3e4deec7c73587f`

## Limits

- The 99.641% result is BF16 causal D256 with 512 calls in one command buffer.
- Chain-1 reaches 0.38752 PFLOP/s because it pays the boundary once per call.
- PFLOP/s counts useful attention work, not all hardware instructions.
- FP8 backward is unsupported.
