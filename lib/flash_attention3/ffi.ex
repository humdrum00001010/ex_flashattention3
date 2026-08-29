defmodule FlashAttention3.FFI do
  @moduledoc """
  The boundary between an Nx block and the FA3 handler ABI.
  Nx sees the results; the handler sees those plus its scratch.
  This module maps between them: pad going in, drop coming out.
  Scratch is accumulators and counters, not part of attention.
  It is declared, not allocated: XLA gives handlers no allocator.
  Command buffers also need addresses fixed at compile time.
  Block tuple size equals result count equals `result_layouts` length.
  `Nx.Defn.Expr.block/4` sizes the block from the default's return.
  It ignores the output template, so the default must be padded.
  Breaking that equality fails MLIR verification.
  Buffers are named for their parameters in `native/fa3_xla.cc`.
  """

  alias FlashAttention3.{Block, DenseAttention}

  # Leading results the operation keeps. The rest is scratch.
  @forward_results 2
  @backward_results 3

  @doc """
  Runs the forward custom call.
  Returns `{output, lse}`; the call also declares one scratch buffer.
  """
  def forward(q, k, v, causal, softmax_scale, block \\ Block.Forward) do
    dims = FlashAttention3.Kernel.dims!(q, k, v, causal)

    results = [
      # output
      Nx.template({dims.batch, dims.seqlen_q, dims.q_heads, dims.value_dim}, q.type),
      # lse, always FP32 and BHQ-ordered whatever the input dtype
      Nx.template({dims.batch, dims.q_heads, dims.seqlen_q}, {:f, 32}),
      # workspace: the tile scheduler's counter. Only the first element is
      # used, and only when causal; the rest is ABI padding.
      Nx.template({dims.batch}, {:s, 32})
    ]

    block
    |> struct!(causal: causal, softmax_scale: softmax_scale)
    |> Nx.block([q, k, v], List.to_tuple(results), fn block, q, k, v ->
      q
      |> DenseAttention.attention(k, v, options(block))
      |> pad_scratch(results, @forward_results)
    end)
    |> drop_scratch(@forward_results)
  end

  @doc """
  Runs the backward custom call.
  Returns `{dq, dk, dv}`; the call declares six scratch buffers.
  """
  def backward(q, k, v, output, lse, doutput, causal, softmax_scale, block \\ Block.Backward) do
    dims = FlashAttention3.Kernel.dims!(q, k, v, causal)
    FlashAttention3.Kernel.backward_operands!(dims, output, lse, doutput, q.type)
    workspace = FlashAttention3.Kernel.workspace(dims, causal)

    # softmax_d and softmax_lse_log2 share a shape; the sequence length is
    # rounded up to the kernel's query-block size.
    softmax_stats =
      Nx.template({dims.batch, dims.q_heads, workspace.seqlen_q_rounded}, {:f, 32})

    results = [
      # dq, dk, dv
      Nx.template(q.shape, q.type),
      Nx.template(k.shape, k.type),
      Nx.template(v.shape, v.type),
      # softmax_d
      softmax_stats,
      # softmax_lse_log2
      softmax_stats,
      # dq_accum: FP32 accumulator, since query blocks are reduced across
      # key blocks and bf16 would lose the partial sums
      Nx.template(
        {dims.batch, dims.q_heads, workspace.seqlen_q_rounded, dims.head_dim},
        {:f, 32}
      ),
      # dq_semaphore: one counter per query block, ordering those accumulations
      Nx.template({workspace.q_blocks, dims.batch, dims.q_heads}, {:s, 32}),
      # dk_accum and dv_accum, keyed by KV head rather than query head
      Nx.template(
        {dims.batch, dims.kv_heads, workspace.seqlen_k_rounded, dims.head_dim},
        {:f, 32}
      ),
      Nx.template(
        {dims.batch, dims.kv_heads, workspace.seqlen_k_rounded, dims.value_dim},
        {:f, 32}
      )
    ]

    block
    |> struct!(causal: causal, softmax_scale: softmax_scale)
    |> Nx.block(
      [q, k, v, output, lse, doutput],
      List.to_tuple(results),
      fn block, q, k, v, _output, _lse, doutput ->
        q
        |> DenseAttention.backward(k, v, doutput, options(block))
        |> pad_scratch(results, @backward_results)
      end
    )
    |> drop_scratch(@backward_results)
  end

  defp options(block), do: block |> Map.from_struct() |> Map.to_list()

  # Widens the default's results to the native tuple size.
  # The zeros are never read: EXLA drops this branch when a spec
  # is returned, and the kernel writes its own scratch when not.
  defp pad_scratch(results, templates, result_count) do
    workspaces =
      for template <- Enum.drop(templates, result_count),
          do: Nx.broadcast(Nx.tensor(0, type: template.type), template.shape)

    (Tuple.to_list(results) ++ workspaces) |> List.to_tuple()
  end

  # Inverse of pad_scratch/3.
  # Takes the leading results, so the handler must declare them first.
  # XLA still allocates the scratch; this only hides it.
  defp drop_scratch(native, result_count),
    do: native |> Tuple.to_list() |> Enum.take(result_count) |> List.to_tuple()
end
