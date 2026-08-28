# 2xH100 results

Fresh-image rerun: 2026-08-28, 2x NVIDIA H100 80GB HBM3 SXM, NV18, CUDA 13.0,
cuDNN 9.12, XLA 0.10.0, Elixir 1.20.2, OTP 29.0.2.

## Correctness

- With the operation-attributes candidate alone, single-GPU and TP2 BF16/FP16
  forward/backward, sharding, and all-reduce placement passed. Resident-buffer
  re-entry then failed because a sharded executable has device ID `-1` while
  its resident buffers have physical device IDs `0` and `1`.
- With the separate EXLA sharded-buffer re-entry patch, the full GPU integration
  gate passed without host staging. That patch is not part of the
  operation-attributes change.
- CPU preflight: 6 passed, 2 GPU tests excluded.

## Forward and backward throughput

Shape: causal `{batch=4, sequence=2048, q_heads=24, kv_heads=4, dim=256}`.
Chain length: 64. `Forward+backward` uses `3 * forward FLOPs`.

| Precision | Operation | 1 GPU PFLOP/s | TP2 PFLOP/s (range) | Scaling |
| --- | --- | ---: | ---: | ---: |
| BF16 | Forward | 0.71013 | 1.37167 (1.33706-1.39267) | 1.932x |
| BF16 | Forward+backward | 0.34011 | 0.65769 (0.65539-0.65940) | 1.934x |
| FP16 | Forward | 0.67875 | 1.27962 (1.27062-1.32992) | 1.885x |
| FP16 | Forward+backward | 0.33967 | 0.65293 (0.64428-0.65551) | 1.922x |

Ranges are slow-to-fast throughput from the timing p90 and p10 samples.

## BF16 forward saturation

Target: 1.42 PFLOP/s. The 99% gate is 1.4058 PFLOP/s.

| Calls per executable | Samples | TP2 PFLOP/s (range) | Target fraction |
| ---: | ---: | ---: | ---: |
| 64 | 10 | 1.37167 (1.33706-1.39267) | 96.597% |
| 512 | 20 | 1.40401 (1.40212-1.40612) | 98.874% |
| 512 | 50 | 1.40032 (1.39637-1.40492) | 98.614% |

The fresh-instance median did not reach the 99% gate. An earlier same-shape
run reached 1.41490 PFLOP/s, but the two fresh-image repetitions above are the
current reproducibility result.

## Bottleneck attribution

- A node-level Nsight Systems capture measured four rank-local FA3 kernels at
  145.024-147.297 us. For the global TP2 workload this is a kernel-only
  equivalent of 1.400-1.422 PFLOP/s; the median is 1.412 PFLOP/s.
- Each causal call also records a 4-byte scheduler-semaphore memset. Its device
  duration was 0.768-0.800 us, less than 0.6% of a steady call.
- Comparing chain 64 and 512 gives an approximately 0.23-0.25 ms fixed outer
  execution boundary. At chain 512 that is only 0.4-0.5 us per FA3 call.
- The final graph contains one command buffer, so the FFI handler and PjRt
  launch are not repeated 512 times during replay.
- Under sustained load the GPUs ran mainly at 1815-1845 MHz despite a reported
  1980 MHz maximum. The container could not lock application clocks.

The observed steady-state bottleneck is therefore the native FA3 kernel path,
not the `Nx.block`, custom-call, FFI, or PjRt boundary. The scheduler memset and
lower sustained clock are measured adjacent contributors, not independently
proven root causes. Backward was validated and benchmarked, but this
counter-level attribution covers forward only.

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
- operation-attributes candidate: `9f26fac0006c4e61094990f25ed53d209469aab8`
- image: `expropriation/nx-exla@sha256:14d7e2930922adaa4d74152c3cfaf24d905c4bc750c0e929dfa5250fcda903e2`
- FA3: `0251105a2fb19d2957484b7f023cd8c115286ced`
- CUTLASS: `7127592069c2fe01b041e174ba4345ef9b279671`
- `libfa3_xla.so`: `36a28ab83fdf2438c6528905f00224ce2744b9783913274adfe887d71ba391de`
- Nsight Systems: 2025.3.0

## Limits

- The latest BF16 causal D256 chain-512 medians are 98.61-98.87% of the target.
- PFLOP/s counts useful attention work, not all hardware instructions.
- Forward has kernel-level attribution; backward does not yet have an Nsight
  Compute counter breakdown.
- FP8 backward is unsupported.
