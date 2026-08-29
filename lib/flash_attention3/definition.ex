defmodule FlashAttention3.Definition do
  @moduledoc """
  Attention written in plain Nx.

  `Nx.Defn.Expr.block/4` applies a block's default implementation on every
  trace, before any backend decides whether to replace it, so `Nx.block/4`
  cannot be used without one. Under `Nx.Defn.Evaluator` it is what actually
  runs. It is therefore required by the API rather than chosen as a policy, and
  it is not a fallback: `FlashAttention3.Block` raises instead of skipping on a
  non-CUDA client, so EXLA never compiles this.

  It materializes the full `{batch, heads, seqlen_q, seqlen_k}` score matrix,
  which is the allocation FlashAttention exists to avoid. That is exactly why
  it is not reachable as a fallback.
  """

  def attention(q, k, v, opts \\ []) do
    opts = Keyword.validate!(opts, causal: false, softmax_scale: nil, upcast: true)

    dims = dims!(q, k, v)
    causal = Keyword.fetch!(opts, :causal)
    softmax_scale = Keyword.get(opts, :softmax_scale) || 1.0 / :math.sqrt(dims.head_dim)

    intermediate_type = if Keyword.fetch!(opts, :upcast), do: {:f, 32}, else: q.type
    q_f32 = Nx.as_type(q, intermediate_type)
    k_f32 = repeat_kv_heads(Nx.as_type(k, intermediate_type), dims.groups)
    v_f32 = repeat_kv_heads(Nx.as_type(v, intermediate_type), dims.groups)

    q_bhqd = Nx.transpose(q_f32, axes: [0, 2, 1, 3])
    k_bhkd = Nx.transpose(k_f32, axes: [0, 2, 1, 3])
    v_bhkd = Nx.transpose(v_f32, axes: [0, 2, 1, 3])

    scores =
      Nx.dot(q_bhqd, [3], [0, 1], k_bhkd, [3], [0, 1])
      |> Nx.multiply(softmax_scale)
      |> maybe_apply_causal_mask(causal, dims.seqlen_q, dims.seqlen_k)

    row_max = Nx.reduce_max(scores, axes: [3], keep_axes: true)
    exponentials = Nx.exp(Nx.subtract(scores, row_max))
    denominator = Nx.sum(exponentials, axes: [3], keep_axes: true)
    probabilities = Nx.divide(exponentials, denominator)

    output =
      Nx.dot(probabilities, [3], [0, 1], v_bhkd, [2], [0, 1])
      |> Nx.transpose(axes: [0, 2, 1, 3])
      |> Nx.as_type(q.type)

    lse =
      denominator
      |> Nx.log()
      |> Nx.add(row_max)
      |> Nx.squeeze(axes: [3])

    {output, lse}
  end

  def backward(q, k, v, doutput, opts \\ []) do
    opts = Keyword.validate!(opts, causal: false, softmax_scale: nil)

    causal = Keyword.fetch!(opts, :causal)
    dims = dims!(q, k, v)

    softmax_scale = Keyword.get(opts, :softmax_scale) || 1.0 / :math.sqrt(dims.head_dim)

    q_f32 = Nx.as_type(q, {:f, 32})
    k_f32 = repeat_kv_heads(Nx.as_type(k, {:f, 32}), dims.groups)
    v_f32 = repeat_kv_heads(Nx.as_type(v, {:f, 32}), dims.groups)
    doutput_f32 = Nx.as_type(doutput, {:f, 32})

    q_bhqd = Nx.transpose(q_f32, axes: [0, 2, 1, 3])
    k_bhkd = Nx.transpose(k_f32, axes: [0, 2, 1, 3])
    v_bhkd = Nx.transpose(v_f32, axes: [0, 2, 1, 3])
    do_bhqd = Nx.transpose(doutput_f32, axes: [0, 2, 1, 3])

    scores =
      Nx.dot(q_bhqd, [3], [0, 1], k_bhkd, [3], [0, 1])
      |> Nx.multiply(softmax_scale)
      |> maybe_apply_causal_mask(causal, dims.seqlen_q, dims.seqlen_k)

    row_max = Nx.reduce_max(scores, axes: [3], keep_axes: true)
    exponentials = Nx.exp(Nx.subtract(scores, row_max))
    probabilities = Nx.divide(exponentials, Nx.sum(exponentials, axes: [3], keep_axes: true))

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
      |> collapse_kv_groups(dims.kv_heads, dims.groups)
      |> Nx.as_type(k.type)

    dv =
      Nx.dot(probabilities, [2], [0, 1], do_bhqd, [2], [0, 1])
      |> Nx.transpose(axes: [0, 2, 1, 3])
      |> collapse_kv_groups(dims.kv_heads, dims.groups)
      |> Nx.as_type(v.type)

    {dq, dk, dv}
  end

  defp repeat_kv_heads(tensor, 1), do: tensor

  defp repeat_kv_heads(tensor, groups) do
    {batch, seqlen, kv_heads, dim} = tensor.shape

    tensor
    |> Nx.reshape({batch, seqlen, kv_heads, 1, dim})
    |> Nx.broadcast({batch, seqlen, kv_heads, groups, dim}, axes: [0, 1, 2, 3, 4])
    |> Nx.reshape({batch, seqlen, kv_heads * groups, dim})
  end

  defp collapse_kv_groups(tensor, kv_heads, groups) do
    {batch, seqlen, _q_heads, dim} = tensor.shape

    tensor
    |> Nx.reshape({batch, seqlen, kv_heads, groups, dim})
    |> Nx.sum(axes: [3])
  end

  defp maybe_apply_causal_mask(scores, false, _seqlen_q, _seqlen_k), do: scores

  defp maybe_apply_causal_mask(scores, true, seqlen_q, seqlen_k) do
    q_index = Nx.iota({seqlen_q, seqlen_k}, axis: 0)
    k_index = Nx.iota({seqlen_q, seqlen_k}, axis: 1)

    allowed =
      q_index
      |> Nx.greater_equal(k_index)
      |> Nx.reshape({1, 1, seqlen_q, seqlen_k})
      |> Nx.broadcast(scores.shape)

    Nx.select(allowed, scores, Nx.broadcast(-1.0e30, scores.shape))
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
