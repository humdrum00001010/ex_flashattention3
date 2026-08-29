# FA3 tensor-parallel XLA experiment

This is an executable boundary test for FlashAttention-3 under Nx/EXLA
`shard_jit`. It is intentionally not an `Nx` primitive.

The ownership chain is:

1. `FlashAttention3.attention/4` presents the tensor operation.
2. `Nx.block/4` preserves that operation as a compiler-visible block.
3. `EXLA.CustomCall` selects a precision-specific StableHLO custom call and
   attaches row-major layout constraints plus a Shardy rule.
4. The external library registers the XLA FFI targets and the custom-call
   partitioner, validates the local ABI, and launches Hopper FA3 kernels.

The implemented native paths are BF16 and FP16 forward/backward for head
dimensions 128 and 256. FP8 backward is not supported or claimed.

[RESULTS.md](RESULTS.md) records what the kernel measured on two H100s. It is
kept as evidence of that run and is stale with respect to this tree; the header
there says what changed under it.

## Use

```elixir
defn block(q, k, v) do
  FlashAttention3.attention(q, k, v, causal: true)
end
```

The application loads `libfa3_xla.so` once, before compiling anything that
contains an FA3 call. The FFI targets and the custom-call partitioner are
registered by the library's static initializers, so loading it is the whole
registration step:

```elixir
:ok = EXLA.NIF.load_dylib("/absolute/path/to/libfa3_xla.so")
```

Q, K, and V are BSHD `{batch, sequence, heads, dim}`. The result is the
attention output; `attention_with_lse/4` also returns the FP32 log-sum-exp for
callers that merge partial results across key/value chunks. Backward runs
through `Nx.Defn.grad/2` and needs no separate call.

This is a binding to the Hopper kernel, not an attention library. On a CUDA
client the operation lowers to `fa3_forward_bf16`, `fa3_forward_f16`,
`fa3_backward_bf16`, or `fa3_backward_f16`; on any other client it raises, and
an unsupported dtype or head dimension is a caller error. There is deliberately
no fallback: a score-matrix attention would silently change the memory
complexity of the model that called it, which is the cost FlashAttention exists
to avoid.

That gate lives in the block, which only EXLA consults. Called eagerly or under
`Nx.Defn.Evaluator`, `FlashAttention3.DenseAttention` runs instead, so those
paths are small-shape only.

`FlashAttention3.DenseAttention` is the formulation FlashAttention replaces —
the materialized score matrix, PyTorch's `math` backend. It exists only because
`Nx.block/4` applies a block's default implementation on every trace and cannot
be used without one. The block raises rather than skipping, so EXLA never
compiles it.

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

CPU verifies StableHLO syntax, precision target selection, layouts, Shardy
attributes, the head-parallel sharding policy, and the analytic backward of
`DenseAttention`. Kernel numerics are not checked here and cannot be: there is no
CPU path through FA3. It does not prove that a CUDA symbol loads or that TP
executes on physical GPUs.

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
export XLA_FLAGS="--xla_gpu_enable_command_buffer=+CUSTOM_CALL --xla_dump_to=$FA3_TP_HLO_DIR --xla_dump_hlo_as_text --xla_dump_hlo_pass_re=spmd|propagation|layout"

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
