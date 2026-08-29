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

  @doc """
  Row-stable softmax, returning the probabilities and the log-sum-exp.

  Deliberately not `Nx.logsumexp/2`. That returns only the normalizer, and
  recovering the probabilities from it costs a second `exp` over the score
  matrix, which is the largest tensor here. Subtracting the row max once yields
  both: the probabilities by division, and the normalizer by taking the log of
  the same denominator and adding the max back.
  """
  def normalize(scores) do
    row_max = Nx.reduce_max(scores, axes: [3], keep_axes: true)
    exponentials = Nx.exp(Nx.subtract(scores, row_max))
    denominator = Nx.sum(exponentials, axes: [3], keep_axes: true)

    probabilities = Nx.divide(exponentials, denominator)
    lse = denominator |> Nx.log() |> Nx.add(row_max) |> Nx.squeeze(axes: [3])

    {probabilities, lse}
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
