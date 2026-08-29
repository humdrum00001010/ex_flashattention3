defmodule FlashAttention3.FFI do
  @moduledoc """
  Runs the FA3 custom calls and hides the kernel's scratch buffers.

  XLA has no notion of handler-side scratch, so FA3's working memory has to be
  declared as extra results of the custom call and dropped again on the way
  out. Forward returns two results and one scratch buffer; backward returns
  three and six. Declaring them is mandatory, and dropping them here is what
  keeps them out of the operation.

  They are results rather than allocated inside the handler because a captured
  CUDA command buffer needs its buffer addresses fixed at compile time, which
  is also why their geometry has to be computed in Elixir. Every buffer below
  is named for its parameter in `native/fa3_xla.cc`.
  """

  alias FlashAttention3.{Block, DenseAttention}

  # How many leading results the operation keeps. Everything after them is
  # scratch.
  @forward_results 2
  @backward_results 3

  @doc """
  Runs the forward custom call and returns `{output, lse}`.
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
      DenseAttention.attention(q, k, v, options(block))
    end)
    |> drop_workspaces(@forward_results)
  end

  @doc """
  Runs the backward custom call and returns `{dq, dk, dv}`.
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
        DenseAttention.backward(q, k, v, doutput, options(block))
      end
    )
    |> drop_workspaces(@backward_results)
  end

  defp options(block), do: block |> Map.from_struct() |> Map.to_list()

  defp drop_workspaces(native, result_count),
    do: native |> Tuple.to_list() |> Enum.take(result_count) |> List.to_tuple()
end
