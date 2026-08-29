defmodule FlashAttention3.Definition do
  @moduledoc """
  Attention written in plain Nx.

  `Nx.Defn.Expr.block/4` applies a block's default implementation on every
  trace, before any backend decides whether to replace it, so `Nx.block/4`
  cannot be used without one. Under `Nx.Defn.Evaluator` it is what actually
  runs. It is therefore required by the API rather than chosen as a policy, and
  it is not a fallback: `FlashAttention3.Block` raises rather than skipping on a
  non-CUDA client, so EXLA never compiles this.

  It materializes the full score matrix, which is the allocation
  FlashAttention exists to avoid. That is exactly why it is not reachable as a
  fallback.
  """

  alias FlashAttention3.Definition.{Gqa, Scores}

  @doc """
  Attention over BSHD tensors, returning `{output, lse}`.
  """
  def attention(q, k, v, opts \\ []) do
    opts = Keyword.validate!(opts, causal: false, softmax_scale: nil, upcast: true)
    causal = Keyword.fetch!(opts, :causal)
    dims = dims!(q, k, v)
    softmax_scale = Keyword.get(opts, :softmax_scale) || 1.0 / :math.sqrt(dims.head_dim)
    type = if Keyword.fetch!(opts, :upcast), do: {:f, 32}, else: q.type

    {q_bhqd, k_bhkd, v_bhkd} = to_bhsd(q, k, v, dims.groups, type)

    scores =
      Scores.masked(q_bhqd, k_bhkd, softmax_scale, causal, dims.seqlen_q, dims.seqlen_k)

    lse = Nx.logsumexp(scores, axes: [3], keep_axes: true)

    output =
      scores
      |> probabilities(lse)
      |> Nx.dot([3], [0, 1], v_bhkd, [2], [0, 1])
      |> Nx.transpose(axes: [0, 2, 1, 3])
      |> Nx.as_type(q.type)

    {output, Nx.squeeze(lse, axes: [3])}
  end

  @doc """
  The analytic gradient of `attention/4`, returning `{dq, dk, dv}`.
  """
  def backward(q, k, v, doutput, opts \\ []) do
    opts = Keyword.validate!(opts, causal: false, softmax_scale: nil)
    causal = Keyword.fetch!(opts, :causal)
    dims = dims!(q, k, v)
    softmax_scale = Keyword.get(opts, :softmax_scale) || 1.0 / :math.sqrt(dims.head_dim)

    {q_bhqd, k_bhkd, v_bhkd} = to_bhsd(q, k, v, dims.groups, {:f, 32})
    do_bhqd = doutput |> Nx.as_type({:f, 32}) |> Nx.transpose(axes: [0, 2, 1, 3])

    scores =
      Scores.masked(q_bhqd, k_bhkd, softmax_scale, causal, dims.seqlen_q, dims.seqlen_k)

    probabilities = probabilities(scores, Nx.logsumexp(scores, axes: [3], keep_axes: true))

    dprobabilities = Nx.dot(do_bhqd, [3], [0, 1], v_bhkd, [3], [0, 1])

    dscores =
      probabilities
      |> Nx.multiply(
        Nx.subtract(
          dprobabilities,
          Nx.sum(Nx.multiply(dprobabilities, probabilities), axes: [3], keep_axes: true)
        )
      )
      |> Nx.multiply(softmax_scale)

    dq =
      Nx.dot(dscores, [3], [0, 1], k_bhkd, [2], [0, 1])
      |> Nx.transpose(axes: [0, 2, 1, 3])
      |> Nx.as_type(q.type)

    dk =
      Nx.dot(dscores, [2], [0, 1], q_bhqd, [2], [0, 1])
      |> Nx.transpose(axes: [0, 2, 1, 3])
      |> Gqa.collapse(dims.kv_heads, dims.groups)
      |> Nx.as_type(k.type)

    dv =
      Nx.dot(probabilities, [2], [0, 1], do_bhqd, [2], [0, 1])
      |> Nx.transpose(axes: [0, 2, 1, 3])
      |> Gqa.collapse(dims.kv_heads, dims.groups)
      |> Nx.as_type(v.type)

    {dq, dk, dv}
  end

  # The identity FA3's backward relies on: recovering the probabilities from
  # the scores and the normalizer, rather than storing them.
  defp probabilities(scores, lse), do: Nx.exp(Nx.subtract(scores, lse))

  defp to_bhsd(q, k, v, groups, type) do
    {
      q |> Nx.as_type(type) |> Nx.transpose(axes: [0, 2, 1, 3]),
      k |> Nx.as_type(type) |> Gqa.expand(groups) |> Nx.transpose(axes: [0, 2, 1, 3]),
      v |> Nx.as_type(type) |> Gqa.expand(groups) |> Nx.transpose(axes: [0, 2, 1, 3])
    }
  end

  defp dims!(q, k, v) do
    {batch, seqlen_q, q_heads, _head_dim} = q.shape
    {^batch, seqlen_k, kv_heads, value_dim} = v.shape
    {^batch, ^seqlen_k, ^kv_heads, head_dim} = k.shape

    %{
      batch: batch,
      seqlen_q: seqlen_q,
      seqlen_k: seqlen_k,
      q_heads: q_heads,
      kv_heads: kv_heads,
      groups: div(q_heads, kv_heads),
      head_dim: head_dim,
      value_dim: value_dim
    }
  end
end
