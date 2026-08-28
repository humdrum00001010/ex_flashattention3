defmodule FA3TP.Block do
  @moduledoc false

  @enforce_keys [
    :call_target_name,
    :backward_call_target_name,
    :causal,
    :softmax_scale
  ]
  defstruct [
    :call_target_name,
    :backward_call_target_name,
    :causal,
    :softmax_scale,
    platforms: [:cuda]
  ]
end

defmodule FA3TP.BackwardBlock do
  @moduledoc false

  @enforce_keys [:call_target_name, :causal, :softmax_scale]
  defstruct [:call_target_name, :causal, :softmax_scale, platforms: [:cuda]]
end

defmodule FA3TP do
  @moduledoc """
  Executable contract for the torch-free FA3 tensor-parallel experiment.

  Nx owns the public operation and FP32 fallback. EXLA selects the custom call.
  The external dylib owns the XLA FFI ABI, Hopper launch, and partitioner.
  The native experiment supports BF16 and FP16 forward/backward.
  """

  import Nx.Defn.Kernel, only: [custom_grad: 3]

  def forward(q, k, v, opts \\ []) do
    opts =
      Keyword.validate!(opts,
        causal: false,
        softmax_scale: nil,
        call_target_name: "exla_fa3_forward",
        backward_call_target_name: "exla_fa3_backward",
        platforms: [:cuda]
      )

    {batch, seqlen_q, q_heads, head_dim} = rank4_shape!(q, "q")
    {^batch, seqlen_k, kv_heads, ^head_dim} = rank4_shape!(k, "k")
    {^batch, ^seqlen_k, ^kv_heads, value_dim} = rank4_shape!(v, "v")

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

    block = %FA3TP.Block{
      call_target_name: Keyword.fetch!(opts, :call_target_name),
      backward_call_target_name: Keyword.fetch!(opts, :backward_call_target_name),
      causal: causal,
      softmax_scale: softmax_scale,
      platforms: Keyword.fetch!(opts, :platforms)
    }

    output = Nx.template({batch, seqlen_q, q_heads, value_dim}, q.type)
    lse = Nx.template({batch, q_heads, seqlen_q}, {:f, 32})
    scheduler_workspace = Nx.template({batch}, {:s, 32})

    {forward_output, forward_lse, _scheduler_workspace} =
      Nx.block(block, [q, k, v], {output, lse, scheduler_workspace}, fn block, q, k, v ->
        {output, lse} =
          reference(q, k, v, causal: block.causal, softmax_scale: block.softmax_scale)

        {output, lse, Nx.broadcast(Nx.tensor(0, type: {:s, 32}), {batch})}
      end)

    result = {forward_output, forward_lse}

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
              softmax_scale: block.softmax_scale,
              call_target_name: block.backward_call_target_name,
              platforms: block.platforms
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
        softmax_scale: nil,
        call_target_name: "exla_fa3_backward",
        platforms: [:cuda]
      )

    {batch, seqlen_q, q_heads, head_dim} = rank4_shape!(q, "q")
    {^batch, seqlen_k, kv_heads, ^head_dim} = rank4_shape!(k, "k")
    rank4_shape!(v, "v")
    rank4_shape!(output, "output")
    rank4_shape!(doutput, "doutput")

    block = %FA3TP.BackwardBlock{
      call_target_name: Keyword.fetch!(opts, :call_target_name),
      causal: Keyword.fetch!(opts, :causal),
      softmax_scale: Keyword.get(opts, :softmax_scale) || 1.0 / :math.sqrt(head_dim),
      platforms: Keyword.fetch!(opts, :platforms)
    }

    k_block_m = if head_dim <= 128, do: if(block.causal, do: 64, else: 80), else: 64
    k_block_n = if head_dim <= 128, do: 128, else: 80
    seqlen_q_rounded = round_up(seqlen_q, k_block_m)
    seqlen_k_rounded = round_up(seqlen_k, k_block_n)
    q_blocks = div(seqlen_q_rounded, k_block_m)

    softmax_shape = {batch, q_heads, seqlen_q_rounded}
    dq_accum_shape = {batch, q_heads, seqlen_q_rounded, head_dim}
    dq_semaphore_shape = {q_blocks, batch, q_heads}
    dk_accum_shape = {batch, kv_heads, seqlen_k_rounded, head_dim}

    outputs = {
      Nx.template(q.shape, q.type),
      Nx.template(k.shape, k.type),
      Nx.template(v.shape, v.type),
      Nx.template(softmax_shape, {:f, 32}),
      Nx.template(softmax_shape, {:f, 32}),
      Nx.template(dq_accum_shape, {:f, 32}),
      Nx.template(dq_semaphore_shape, {:s, 32}),
      Nx.template(dk_accum_shape, {:f, 32}),
      Nx.template(dk_accum_shape, {:f, 32})
    }

    {dq, dk, dv, _softmax_d, _lse_log2, _dq_accum, _dq_semaphore, _dk_accum, _dv_accum} =
      Nx.block(block, [q, k, v, output, lse, doutput], outputs, fn block,
                                                                   q,
                                                                   k,
                                                                   v,
                                                                   _output,
                                                                   _lse,
                                                                   doutput ->
        {dq, dk, dv} =
          reference_backward(q, k, v, doutput,
            causal: block.causal,
            softmax_scale: block.softmax_scale
          )

        {dq, dk, dv, zero(softmax_shape, {:f, 32}), zero(softmax_shape, {:f, 32}),
         zero(dq_accum_shape, {:f, 32}), zero(dq_semaphore_shape, {:s, 32}),
         zero(dk_accum_shape, {:f, 32}), zero(dk_accum_shape, {:f, 32})}
      end)

    {dq, dk, dv}
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

  def operation_attributes(q, k, v) do
    {batch, seqlen_q, q_heads, head_dim} = rank4_shape!(q, "q")
    {^batch, seqlen_k, kv_heads, ^head_dim} = rank4_shape!(k, "k")
    {^batch, ^seqlen_k, ^kv_heads, value_dim} = rank4_shape!(v, "v")
    groups = div(q_heads, kv_heads)

    sharding_rule =
      "#sdy.op_sharding_rule<" <>
        "([i, j, lm, n], [i, k, m, n], [i, k, m, o])->" <>
        "([i, j, lm, o], [i, lm, j], [i]) " <>
        "{i=#{batch}, j=#{seqlen_q}, k=#{seqlen_k}, l=#{groups}, " <>
        "m=#{kv_heads}, n=#{head_dim}, o=#{value_dim}} " <>
        "need_replication={i, j, k, l, n, o}, custom>"

    [
      {"operand_layouts",
       "[dense<[3, 2, 1, 0]> : tensor<4xindex>, " <>
         "dense<[3, 2, 1, 0]> : tensor<4xindex>, " <>
         "dense<[3, 2, 1, 0]> : tensor<4xindex>]"},
      {"result_layouts",
       "[dense<[3, 2, 1, 0]> : tensor<4xindex>, " <>
         "dense<[2, 1, 0]> : tensor<3xindex>, " <>
         "dense<[0]> : tensor<1xindex>]"},
      {"sdy.sharding_rule", sharding_rule}
    ]
  end

  def backward_operation_attributes(q, k, v, causal) do
    {batch, seqlen_q, q_heads, head_dim} = rank4_shape!(q, "q")
    {^batch, seqlen_k, kv_heads, ^head_dim} = rank4_shape!(k, "k")
    {^batch, ^seqlen_k, ^kv_heads, value_dim} = rank4_shape!(v, "v")
    groups = div(q_heads, kv_heads)
    k_block_m = if head_dim <= 128, do: if(causal, do: 64, else: 80), else: 64
    k_block_n = if head_dim <= 128, do: 128, else: 80
    seqlen_q_rounded = round_up(seqlen_q, k_block_m)
    seqlen_k_rounded = round_up(seqlen_k, k_block_n)
    q_blocks = div(seqlen_q_rounded, k_block_m)

    sharding_rule =
      "#sdy.op_sharding_rule<" <>
        "([i, j, lm, n], [i, k, m, n], [i, k, m, o], " <>
        "[i, j, lm, o], [i, lm, j], [i, j, lm, o])->" <>
        "([i, j, lm, n], [i, k, m, n], [i, k, m, o], " <>
        "[i, lm, p], [i, lm, p], [i, lm, p, n], [r, i, lm], " <>
        "[i, m, q, n], [i, m, q, o]) " <>
        "{i=#{batch}, j=#{seqlen_q}, k=#{seqlen_k}, l=#{groups}, " <>
        "m=#{kv_heads}, n=#{head_dim}, o=#{value_dim}, " <>
        "p=#{seqlen_q_rounded}, q=#{seqlen_k_rounded}, r=#{q_blocks}} " <>
        "need_replication={i, j, k, l, n, o, p, q, r}, custom>"

    row_major_4d = "dense<[3, 2, 1, 0]> : tensor<4xindex>"
    row_major_3d = "dense<[2, 1, 0]> : tensor<3xindex>"

    [
      {"operand_layouts",
       "[#{row_major_4d}, #{row_major_4d}, #{row_major_4d}, " <>
         "#{row_major_4d}, #{row_major_3d}, #{row_major_4d}]"},
      {"result_layouts",
       "[#{row_major_4d}, #{row_major_4d}, #{row_major_4d}, " <>
         "#{row_major_3d}, #{row_major_3d}, #{row_major_4d}, " <>
         "#{row_major_3d}, #{row_major_4d}, #{row_major_4d}]"},
      {"sdy.sharding_rule", sharding_rule}
    ]
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

  defp round_up(value, block), do: div(value + block - 1, block) * block

  defp zero(shape, type), do: Nx.broadcast(Nx.tensor(0, type: type), shape)
end

defimpl EXLA.CustomCall, for: FA3TP.Block do
  alias EXLA.CustomCall.Spec

  def call(block, _out, [q, k, v], %{platform: platform}) do
    target = precision_target(block.call_target_name, q.type)

    if platform in block.platforms and target do
      attributes = [
        {"causal", to_string(block.causal)},
        {"softmax_scale", "#{block.softmax_scale} : f32"}
      ]

      {:ok,
       %Spec{
         call_target_name: target,
         attributes: attributes,
         operation_attributes: FA3TP.operation_attributes(q, k, v)
       }}
    else
      :skip
    end
  end

  def call(_, _, _, _), do: :skip

  defp precision_target(base, {:bf, 16}), do: base
  defp precision_target(base, {:f, 16}), do: base <> "_f16"
  defp precision_target(_base, _type), do: nil
end

defimpl EXLA.CustomCall, for: FA3TP.BackwardBlock do
  alias EXLA.CustomCall.Spec

  def call(block, _out, [q, k, v, _output, _lse, _doutput], %{platform: platform}) do
    target = precision_target(block.call_target_name, q.type)

    if platform in block.platforms and target do
      attributes = [
        {"causal", to_string(block.causal)},
        {"softmax_scale", "#{block.softmax_scale} : f32"}
      ]

      {:ok,
       %Spec{
         call_target_name: target,
         attributes: attributes,
         operation_attributes: FA3TP.backward_operation_attributes(q, k, v, block.causal)
       }}
    else
      :skip
    end
  end

  def call(_, _, _, _), do: :skip

  defp precision_target(base, {:bf, 16}), do: base
  defp precision_target(base, {:f, 16}), do: base <> "_f16"
  defp precision_target(_base, _type), do: nil
end
