defmodule FlashAttention3.Definition.Scores do
  @moduledoc false

  @doc """
  Scaled Q·Kᵀ scores, masked when causal.

  This is the `{batch, heads, seqlen_q, seqlen_k}` allocation the kernel exists
  to avoid, and the reason this definition is not a viable implementation.
  """
  def masked(q_bhqd, k_bhkd, softmax_scale, causal, seqlen_q, seqlen_k) do
    Nx.dot(q_bhqd, [3], [0, 1], k_bhkd, [3], [0, 1])
    |> Nx.multiply(softmax_scale)
    |> mask(causal, seqlen_q, seqlen_k)
  end

  defp mask(scores, false, _seqlen_q, _seqlen_k), do: scores

  defp mask(scores, true, seqlen_q, seqlen_k) do
    allowed =
      Nx.iota({seqlen_q, seqlen_k}, axis: 0)
      |> Nx.greater_equal(Nx.iota({seqlen_q, seqlen_k}, axis: 1))
      |> Nx.reshape({1, 1, seqlen_q, seqlen_k})
      |> Nx.broadcast(scores.shape)

    Nx.select(allowed, scores, Nx.broadcast(-1.0e30, scores.shape))
  end
end
