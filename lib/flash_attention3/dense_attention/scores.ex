defmodule FlashAttention3.DenseAttention.Scores do
  @moduledoc false

  @doc """
  Scaled Q·Kᵀ scores, masked when causal.

  This is the `{batch, heads, seqlen_q, seqlen_k}` allocation the kernel exists
  to avoid, and the reason this is not a viable implementation.
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
    row_max = scores |> Nx.reduce_max(axes: [3], keep_axes: true) |> stop_grad()
    exponentials = Nx.exp(Nx.subtract(scores, row_max))
    denominator = Nx.sum(exponentials, axes: [3], keep_axes: true)

    probabilities = Nx.divide(exponentials, denominator)
    lse = denominator |> Nx.log() |> Nx.add(row_max) |> Nx.squeeze(axes: [3])

    {probabilities, lse}
  end

  # The row max only shifts the exponent for numerical stability; its
  # contribution to the gradient cancels. Differentiating through it is wasted
  # work on a piecewise-constant function, so hold it fixed the way
  # `Nx.logsumexp/2` does. Only expressions can carry the annotation, so eager
  # tensors pass through.
  defp stop_grad(%Nx.Tensor{data: %Nx.Defn.Expr{}} = row_max),
    do: Nx.Defn.Kernel.stop_grad(row_max)

  defp stop_grad(row_max), do: row_max

  # The row max only shifts the exponent for numerical stability, and its
  # contribution to the gradient cancels. Differentiating through a
  # piecewise-constant function is wasted work, so hold it fixed the way
  # `Nx.logsumexp/2` does. Only expressions carry the annotation.
  defp stop_grad(%Nx.Tensor{data: %Nx.Defn.Expr{}} = row_max),
    do: Nx.Defn.Kernel.stop_grad(row_max)

  defp stop_grad(row_max), do: row_max

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
