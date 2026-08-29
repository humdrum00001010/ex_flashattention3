defmodule FlashAttention3.FFI do
  @moduledoc """
  Owns the complete native result contract for the FA3 custom calls.

  The kernel returns compiler-owned workspaces alongside its semantic results.
  Those workspaces exist only because a captured CUDA command buffer needs
  stable buffers, so they are packed and dropped here and never reach the
  operation. Model code uses `FlashAttention3` instead of this module.
  """

  alias FlashAttention3.{Block, Reference, Shape}

  @doc """
  Runs the forward custom call and returns `{output, lse}`.
  """
  def forward(q, k, v, causal, softmax_scale, block \\ Block.Forward) do
    dims = Shape.attention!(q, k, v, causal)

    results = [
      Nx.template({dims.batch, dims.seqlen_q, dims.q_heads, dims.value_dim}, q.type),
      Nx.template({dims.batch, dims.q_heads, dims.seqlen_q}, {:f, 32}),
      Nx.template({dims.batch}, {:s, 32})
    ]

    block
    |> struct!(causal: causal, softmax_scale: softmax_scale)
    |> Nx.block([q, k, v], List.to_tuple(results), fn block, q, k, v ->
      pack(Reference.attention(q, k, v, options(block)), results, 2)
    end)
    |> unpack(2)
  end

  @doc """
  Runs the backward custom call and returns `{dq, dk, dv}`.
  """
  def backward(q, k, v, output, lse, doutput, causal, softmax_scale, block \\ Block.Backward) do
    dims = Shape.attention!(q, k, v, causal)
    Shape.backward!(dims, output, lse, doutput, q.type)
    workspace = Shape.workspace(dims, causal)

    softmax = Nx.template({dims.batch, dims.q_heads, workspace.seqlen_q_rounded}, {:f, 32})

    results = [
      Nx.template(q.shape, q.type),
      Nx.template(k.shape, k.type),
      Nx.template(v.shape, v.type),
      softmax,
      softmax,
      Nx.template(
        {dims.batch, dims.q_heads, workspace.seqlen_q_rounded, dims.head_dim},
        {:f, 32}
      ),
      Nx.template({workspace.q_blocks, dims.batch, dims.q_heads}, {:s, 32}),
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
        pack(Reference.backward(q, k, v, doutput, options(block)), results, 3)
      end
    )
    |> unpack(3)
  end

  defp options(block), do: block |> Map.from_struct() |> Map.to_list()

  defp pack(semantic, results, count) do
    workspaces =
      for template <- Enum.drop(results, count),
          do: Nx.broadcast(Nx.tensor(0, type: template.type), template.shape)

    (Tuple.to_list(semantic) ++ workspaces) |> List.to_tuple()
  end

  defp unpack(native, count),
    do: native |> Tuple.to_list() |> Enum.take(count) |> List.to_tuple()
end
