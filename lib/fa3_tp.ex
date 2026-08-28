defmodule FA3TP.Block do
  @moduledoc false

  @enforce_keys [:platform, :spec, :causal, :softmax_scale]
  defstruct [:platform, :spec, :causal, :softmax_scale]
end

defmodule FA3TP do
  @moduledoc """
  Executable contract for the torch-free FA3 tensor-parallel experiment.

  Nx owns the public operation and FP32 fallback. EXLA selects the custom call.
  The external dylib owns the XLA FFI ABI, Hopper launch, and partitioner.
  The native experiment supports BF16 and FP16 forward/backward.
  """

  import Nx.Defn.Kernel, only: [custom_grad: 3]

  alias FlashAttention3.FFI

  def forward(q, k, v, opts \\ []) do
    opts =
      Keyword.validate!(opts,
        causal: false,
        softmax_scale: nil
      )

    {batch, seqlen_q, q_heads, head_dim} = rank4_shape!(q, "q")
    {^batch, seqlen_k, kv_heads, ^head_dim} = rank4_shape!(k, "k")
    {^batch, ^seqlen_k, ^kv_heads, _value_dim} = rank4_shape!(v, "v")

    unless q.type == k.type and q.type == v.type do
      raise ArgumentError,
            "FA3 requires q, k, and v to have one dtype, got #{inspect(q.type)}, " <>
              "#{inspect(k.type)}, and #{inspect(v.type)}"
    end

    unless rem(q_heads, kv_heads) == 0 do
      raise ArgumentError,
            "FA3 GQA requires q_heads to be divisible by kv_heads, got #{q_heads} and #{kv_heads}"
    end

    causal = Keyword.fetch!(opts, :causal)

    if causal and seqlen_q != seqlen_k do
      raise ArgumentError,
            "the v1 causal oracle requires equal q/k sequence lengths, got #{seqlen_q} and #{seqlen_k}"
    end

    softmax_scale = Keyword.get(opts, :softmax_scale) || 1.0 / :math.sqrt(head_dim)
    ffi = FFI.forward(q, k, v, causal, softmax_scale)

    block = %FA3TP.Block{
      platform: ffi.platform,
      spec: ffi.spec,
      causal: causal,
      softmax_scale: softmax_scale
    }

    native_results =
      Nx.block(block, ffi.operands, ffi.outputs, fn block, q, k, v ->
        {output, lse} =
          reference(q, k, v, causal: block.causal, softmax_scale: block.softmax_scale)

        FFI.pack_results(ffi, {output, lse})
      end)

    result = {forward_output, forward_lse} = FFI.semantic_results(ffi, native_results)

    case q.data do
      %Nx.Defn.Expr{} ->
        custom_grad(result, [q, k, v], fn {doutput, _dlse} ->
          # Nx differentiates through an f32 scalar loss, so the cotangent
          # arriving at this tuple boundary is f32 even when FA3's output is
          # bf16/f16. The upstream FA3 backward ABI requires dO to have the
          # same element type as Q/K/V/O.
          doutput = Nx.as_type(doutput, q.type)

          {dq, dk, dv} =
            backward(q, k, v, forward_output, forward_lse, doutput,
              causal: block.causal,
              softmax_scale: block.softmax_scale
            )

          [dq, dk, dv]
        end)

      _ ->
        result
    end
  end

  def backward(q, k, v, output, lse, doutput, opts \\ []) do
    opts =
      Keyword.validate!(opts,
        causal: false,
        softmax_scale: nil
      )

    {_batch, _seqlen_q, _q_heads, head_dim} = rank4_shape!(q, "q")
    rank4_shape!(k, "k")
    rank4_shape!(v, "v")
    rank4_shape!(output, "output")
    rank4_shape!(doutput, "doutput")

    causal = Keyword.fetch!(opts, :causal)
    softmax_scale = Keyword.get(opts, :softmax_scale) || 1.0 / :math.sqrt(head_dim)
    ffi = FFI.backward(q, k, v, output, lse, doutput, causal, softmax_scale)

    block = %FA3TP.Block{
      platform: ffi.platform,
      spec: ffi.spec,
      causal: causal,
      softmax_scale: softmax_scale
    }

    native_results =
      Nx.block(block, ffi.operands, ffi.outputs, fn block, q, k, v, _output, _lse, doutput ->
        {dq, dk, dv} =
          reference_backward(q, k, v, doutput,
            causal: block.causal,
            softmax_scale: block.softmax_scale
          )

        FFI.pack_results(ffi, {dq, dk, dv})
      end)

    FFI.semantic_results(ffi, native_results)
  end

  def reference(q, k, v, opts \\ []) do
    opts = Keyword.validate!(opts, causal: false, softmax_scale: nil, upcast: true)

    {batch, seqlen_q, q_heads, head_dim} = rank4_shape!(q, "q")
    {^batch, seqlen_k, kv_heads, ^head_dim} = rank4_shape!(k, "k")
    {^batch, ^seqlen_k, ^kv_heads, _value_dim} = rank4_shape!(v, "v")

    unless rem(q_heads, kv_heads) == 0 do
      raise ArgumentError,
            "FA3 GQA requires q_heads to be divisible by kv_heads, got #{q_heads} and #{kv_heads}"
    end

    causal = Keyword.fetch!(opts, :causal)

    if causal and seqlen_q != seqlen_k do
      raise ArgumentError,
            "the v1 causal oracle requires equal q/k sequence lengths, got #{seqlen_q} and #{seqlen_k}"
    end

    groups = div(q_heads, kv_heads)
    softmax_scale = Keyword.get(opts, :softmax_scale) || 1.0 / :math.sqrt(head_dim)

    intermediate_type = if Keyword.fetch!(opts, :upcast), do: {:f, 32}, else: q.type
    q_f32 = Nx.as_type(q, intermediate_type)
    k_f32 = repeat_kv_heads(Nx.as_type(k, intermediate_type), groups)
    v_f32 = repeat_kv_heads(Nx.as_type(v, intermediate_type), groups)

    q_bhqd = Nx.transpose(q_f32, axes: [0, 2, 1, 3])
    k_bhkd = Nx.transpose(k_f32, axes: [0, 2, 1, 3])
    v_bhkd = Nx.transpose(v_f32, axes: [0, 2, 1, 3])

    scores =
      Nx.dot(q_bhqd, [3], [0, 1], k_bhkd, [3], [0, 1])
      |> Nx.multiply(softmax_scale)
      |> maybe_apply_causal_mask(causal, seqlen_q, seqlen_k)

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

  def reference_backward(q, k, v, doutput, opts \\ []) do
    opts = Keyword.validate!(opts, causal: false, softmax_scale: nil)

    {batch, seqlen_q, q_heads, head_dim} = rank4_shape!(q, "q")
    {^batch, seqlen_k, kv_heads, ^head_dim} = rank4_shape!(k, "k")
    {^batch, ^seqlen_k, ^kv_heads, _value_dim} = rank4_shape!(v, "v")
    rank4_shape!(doutput, "doutput")

    groups = div(q_heads, kv_heads)
    causal = Keyword.fetch!(opts, :causal)
    softmax_scale = Keyword.get(opts, :softmax_scale) || 1.0 / :math.sqrt(head_dim)

    q_f32 = Nx.as_type(q, {:f, 32})
    k_f32 = repeat_kv_heads(Nx.as_type(k, {:f, 32}), groups)
    v_f32 = repeat_kv_heads(Nx.as_type(v, {:f, 32}), groups)
    doutput_f32 = Nx.as_type(doutput, {:f, 32})

    q_bhqd = Nx.transpose(q_f32, axes: [0, 2, 1, 3])
    k_bhkd = Nx.transpose(k_f32, axes: [0, 2, 1, 3])
    v_bhkd = Nx.transpose(v_f32, axes: [0, 2, 1, 3])
    do_bhqd = Nx.transpose(doutput_f32, axes: [0, 2, 1, 3])

    scores =
      Nx.dot(q_bhqd, [3], [0, 1], k_bhkd, [3], [0, 1])
      |> Nx.multiply(softmax_scale)
      |> maybe_apply_causal_mask(causal, seqlen_q, seqlen_k)

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
      |> collapse_kv_groups(kv_heads, groups)
      |> Nx.as_type(k.type)

    dv =
      Nx.dot(probabilities, [2], [0, 1], do_bhqd, [2], [0, 1])
      |> Nx.transpose(axes: [0, 2, 1, 3])
      |> collapse_kv_groups(kv_heads, groups)
      |> Nx.as_type(v.type)

    {dq, dk, dv}
  end

  def shard_inputs(q, k, v, partitions) when is_integer(partitions) and partitions > 0 do
    {_batch, _seqlen_q, q_heads, _head_dim} = rank4_shape!(q, "q")
    {_batch, _seqlen_k, kv_heads, _head_dim} = rank4_shape!(k, "k")

    unless rem(kv_heads, partitions) == 0 do
      raise ArgumentError,
            "TP must keep complete KV groups: #{kv_heads} KV heads cannot be split over #{partitions} partitions"
    end

    unless rem(q_heads, partitions) == 0 do
      raise ArgumentError,
            "#{q_heads} Q heads cannot be split over #{partitions} partitions"
    end

    q_per_partition = div(q_heads, partitions)
    kv_per_partition = div(kv_heads, partitions)

    for partition <- 0..(partitions - 1) do
      [
        Nx.slice_along_axis(q, partition * q_per_partition, q_per_partition, axis: 2),
        Nx.slice_along_axis(k, partition * kv_per_partition, kv_per_partition, axis: 2),
        Nx.slice_along_axis(v, partition * kv_per_partition, kv_per_partition, axis: 2)
      ]
    end
  end

  def assemble_outputs(outputs) when is_list(outputs) do
    {output_shards, lse_shards} = Enum.unzip(outputs)
    {Nx.concatenate(output_shards, axis: 2), Nx.concatenate(lse_shards, axis: 1)}
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

  defp rank4_shape!(%Nx.Tensor{shape: {a, b, c, d}}, _name), do: {a, b, c, d}

  defp rank4_shape!(%Nx.Tensor{shape: shape}, name) do
    raise ArgumentError,
          "#{name} must have shape {batch, sequence, heads, dim}, got #{inspect(shape)}"
  end
end

defimpl EXLA.CustomCall, for: FA3TP.Block do
  def call(block, _out, _inputs, %{platform: platform}) do
    if platform == block.platform and block.spec do
      {:ok, block.spec}
    else
      :skip
    end
  end

  def call(_, _, _, _), do: :skip
end
