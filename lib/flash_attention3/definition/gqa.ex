defmodule FlashAttention3.Definition.Gqa do
  @moduledoc false

  @doc """
  Repeats each KV head across its query group.
  """
  def expand(tensor, 1), do: tensor

  def expand(tensor, groups) do
    {batch, seqlen, kv_heads, dim} = tensor.shape

    tensor
    |> Nx.reshape({batch, seqlen, kv_heads, 1, dim})
    |> Nx.broadcast({batch, seqlen, kv_heads, groups, dim}, axes: [0, 1, 2, 3, 4])
    |> Nx.reshape({batch, seqlen, kv_heads * groups, dim})
  end

  @doc """
  Sums a per-query-head gradient back onto its KV head.
  """
  def collapse(tensor, kv_heads, groups) do
    {batch, seqlen, _q_heads, dim} = tensor.shape

    tensor
    |> Nx.reshape({batch, seqlen, kv_heads, groups, dim})
    |> Nx.sum(axes: [3])
  end
end
