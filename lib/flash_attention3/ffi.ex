defmodule FlashAttention3.FFI do
  @moduledoc """
  Native results are the operation's results followed by workspace.
  This module widens the default to that shape and narrows it back.
  Workspace crosses the boundary because XLA has no handler scratch.
  It is declared, not allocated: command buffers fix addresses early.
  The block tuple size must equal the custom call result count.
  That count must equal the length of `result_layouts`.
  `Nx.Defn.Expr.block/4` sizes the block from the default's return.
  It ignores the output template, so the default must be padded.
  A mismatch emits a call whose layouts disagree with its results.
  MLIR then fails to verify it.
  Buffers are named for their parameters in `native/fa3_xla.cc`.
  """

  alias FlashAttention3.{Block, DenseAttention}

  # Leading results the operation keeps. The rest is workspace.
  @forward_results 2
  @backward_results 3

  @doc """
  Runs the forward custom call.
  Returns `{output, lse}`; the call also declares one workspace buffer.
  """
  def forward(q, k, v, causal, softmax_scale, block \\ Block.Forward) do
    dims = FlashAttention3.Kernel.dims!(q, k, v, causal)

    results = [
      # output
      Nx.template({dims.batch, dims.seqlen_q, dims.q_heads, dims.value_dim}, q.type),
      # lse, always FP32 and BHQ-ordered whatever the input dtype
      Nx.template({dims.batch, dims.q_heads, dims.seqlen_q}, {:f, 32}),
      # workspace: tile-scheduler state, one entry per batch
      Nx.template({dims.batch}, {:s, 32})
    ]

    block
    |> struct!(causal: causal, softmax_scale: softmax_scale)
    |> Nx.block([q, k, v], List.to_tuple(results), fn block, q, k, v ->
      q
      |> DenseAttention.attention(k, v, options(block))
      |> pad_workspaces(results, @forward_results)
    end)
    |> drop_workspaces(@forward_results)
  end

  @doc """
  Runs the backward custom call.
  Returns `{dq, dk, dv}`; the call declares six workspace buffers.
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
        |> pad_workspaces(results, @backward_results)
      end
    )
    |> drop_workspaces(@backward_results)
  end

  defp options(block), do: block |> Map.from_struct() |> Map.to_list()

  # Widens the default's results to the native tuple size.
  # The zeros are never read: EXLA drops this branch when a spec
  # is returned, and the kernel writes its own workspace when not.
  defp pad_workspaces(results, templates, result_count) do
    workspaces =
      for template <- Enum.drop(templates, result_count),
          do: Nx.broadcast(Nx.tensor(0, type: template.type), template.shape)

    (Tuple.to_list(results) ++ workspaces) |> List.to_tuple()
  end

  # Inverse of pad_workspaces/3.
  # Takes the leading results, so the handler must declare them first.
  # XLA still allocates the workspace; this only hides it.
  defp drop_workspaces(native, result_count),
    do: native |> Tuple.to_list() |> Enum.take(result_count) |> List.to_tuple()
end
