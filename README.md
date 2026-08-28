# FA3 tensor-parallel XLA experiment

This is an executable boundary test for FlashAttention-3 under Nx/EXLA
`shard_jit`. It is intentionally not an `Nx` primitive.

See [RESULTS.md](RESULTS.md) for the exact two-H100 correctness, benchmark,
and Nsight evidence.

The ownership chain is:

1. `FA3TP.forward/4` presents the tensor operation and an FP32 Nx fallback.
2. `Nx.block/4` preserves that operation as a compiler-visible block.
3. `EXLA.CustomCall` selects a precision-specific StableHLO custom call and
   attaches row-major layout constraints plus a Shardy rule.
4. The external library registers the XLA FFI targets and the custom-call
   partitioner, validates the local ABI, and launches Hopper FA3 kernels.

The implemented native paths are BF16 and FP16 forward/backward for head
dimensions 128 and 256. FP8 backward is not supported or claimed.

## Layout

The default build expects sibling directories arranged like this:

```text
root/
  fa3_tp_experiment/
  nx-worktree/             # modified Nx/EXLA worktree
  build/flash-attention/   # upstream FA3 checkout, including CUTLASS
  build/fa3-xla/           # generated objects
  artifacts/libfa3_xla.so
```

The tested FA3 checkout is commit
`0251105a2fb19d2957484b7f023cd8c115286ced`, with CUTLASS submodule
`7127592069c2fe01b041e174ba4345ef9b279671`.

## Install and build

Use an Elixir/OTP pair supported by the modified worktree. The hardware run
used Elixir 1.20.2 and OTP 29.0.2.

```sh
export NX_OPERATION_ATTRIBUTES_WORKTREE=/absolute/path/to/nx-worktree
export XLA_TARGET=cuda13
mix deps.get
mix compile

make -C native \
  FA3_ROOT=/absolute/path/to/flash-attention \
  XLA_EXTENSION_DIR="$NX_OPERATION_ATTRIBUTES_WORKTREE/exla/cache/xla_extension" \
  OUTPUT=/absolute/path/to/libfa3_xla.so
```

EXLA reuses a matching XLA archive from its cache when one is already
installed. The native build is torch-free; it links the FA3/CUTLASS objects,
`libxla_extension`, and CUDA runtime directly.

## CPU preflight

CPU verifies the public fallback, analytic backward, StableHLO syntax,
precision target selection, layouts, and Shardy attributes. It does not prove
that a CUDA symbol loads or that TP executes on physical GPUs.

```sh
mix format --check-formatted
mix test
```

## Two-H100 correctness gate

Use a new empty HLO directory for every process. `XLA_FLAGS` is read when the
BEAM initializes, and the test rejects stale dumps.

```sh
export FA3_TP_DYLIB=/absolute/path/to/libfa3_xla.so
export FA3_TP_HLO_DIR=/absolute/path/to/new-empty-hlo-dir
export XLA_FLAGS="--xla_dump_to=$FA3_TP_HLO_DIR --xla_dump_hlo_as_text --xla_dump_hlo_pass_re=spmd|propagation|layout"

mix test test/fa3_tp_integration_test.exs --include integration
```

The gate requires, in order:

- two distinct, non-MIG compute-capability 9.0 GPUs;
- native single-GPU output against an independent FP32 oracle;
- TP2 output reconstruction with local Q/KV head shapes and no collective
  inside attention;
- an output-projection all-reduce after attention;
- BF16 and FP16 single-GPU plus TP2 forward/backward gradients;
- resident sharded-buffer reentry, with no hidden host staging.

## Performance

Default shape: causal `{4, 2048, 24, 4, 256}`. Throughput includes device
completion and one PjRt boundary per compiled chain.

| Precision | Operation | 1 GPU PFLOP/s | TP2 PFLOP/s | TP2 scaling |
| --- | --- | ---: | ---: | ---: |
| BF16 | Forward | 0.70761 | 1.37640 | 1.945x |
| BF16 | Forward+backward | 0.33970 | 0.65972 | 1.942x |
| FP16 | Forward | 0.65509 | 1.28723 | 1.965x |
| FP16 | Forward+backward | 0.33922 | 0.65446 | 1.929x |

BF16 TP2 forward reached **1.41490 PFLOP/s**, or **99.641%** of the
1.42 PFLOP/s target, at chain 512. The slow-side sample was 1.41035 PFLOP/s
(99.320%).

Run the full forward/backward benchmark:

```sh
FA3_TP_BENCHMARK=1 \
mix test test/fa3_tp_integration_test.exs --include integration
```

Run the 99% probe:

```sh
FA3_TP_BENCHMARK=1 FA3_TP_PRECISION=bf16 FA3_TP_FORWARD_ONLY=1 \
FA3_TP_CHAIN_LENGTH=512 FA3_TP_WARMUP=5 FA3_TP_ITERATIONS=20 \
mix test test/fa3_tp_integration_test.exs --include integration
```

See [RESULTS.md](RESULTS.md) for scaling, proof, and hashes.
