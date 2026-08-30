defmodule FlashAttention3.Kernel do
  @moduledoc """
  What the FA3 Hopper kernel accepts, and how it is described to XLA.

  This is the whole ABI in one place: which handler runs for a dtype, what
  shapes the kernel admits, how its buffers are laid out, and which dimension
  it can be partitioned over. It mirrors the native handler's signature, and
  nothing here changes without a matching change there.
  """

  alias EXLA.CustomCall.Spec

  # Not a property of FlashAttention-3. CUTLASS kernels are templates
  # parameterized by head dimension and dtype, and upstream ships each
  # instantiation as its own translation unit because each is a slow, large
  # compile. The native build links eight of them: forward and backward, head
  # dimensions 128 and 256, BF16 and FP16, all sm90. These lists and the target
  # map below are that choice mirrored, so an unsupported combination fails at
  # trace time rather than as a missing symbol.
  #
  # Widening either means adding instantiations to the native build first.
  # Editing only these emits a call nothing can service.
  @head_dims [128, 256]

  # Shardy dimension variables, shared by both directions:
  #
  #   i batch      j seqlen_q   k seqlen_k   ml q_heads (m kv_heads x l groups)
  #   n head_dim   o value_dim
  #
  # Backward adds the rounded scratch extents:
  #
  #   p seqlen_q_rounded   q seqlen_k_rounded   r q_blocks
  #
  # A mapping's length is its buffer's rank, so the layout constraints are
  # derived from these rather than repeated alongside them.
  #
  # Shardy orders a dimension's factors major to minor, and the head axis is
  # KV-major: `q_head = kv_head * groups + group`. That is the kernel's layout,
  # not a choice made here. `DenseAttention.Gqa.expand/2` mirrors it and
  # `TensorParallel` assumes it; what ties either to the kernel is the two-GPU
  # gate, which compares kernel output against that dense implementation at
  # four groups, where the two orders diverge.
  #
  # Hence `m` before `l`. Only `m` is shardable: query heads within a group
  # read one shared KV pair, so splitting `l` would put readers of the same KV
  # on different devices and need a collective inside attention.

  @forward %{
    targets: %{{:bf, 16} => "fa3_forward_bf16", {:f, 16} => "fa3_forward_f16"},
    operands: [[:i, :j, :ml, :n], [:i, :k, :m, :n], [:i, :k, :m, :o]],
    results: [[:i, :j, :ml, :o], [:i, :ml, :j], [:i]],
    replicate: [:i, :j, :k, :l, :n, :o]
  }

  @backward %{
    targets: %{{:bf, 16} => "fa3_backward_bf16", {:f, 16} => "fa3_backward_f16"},
    operands: [
      [:i, :j, :ml, :n],
      [:i, :k, :m, :n],
      [:i, :k, :m, :o],
      [:i, :j, :ml, :o],
      [:i, :ml, :j],
      [:i, :j, :ml, :o]
    ],
    results: [
      [:i, :j, :ml, :n],
      [:i, :k, :m, :n],
      [:i, :k, :m, :o],
      [:i, :ml, :p],
      [:i, :ml, :p],
      [:i, :ml, :p, :n],
      [:r, :i, :ml],
      [:i, :m, :q, :n],
      [:i, :m, :q, :o]
    ],
    replicate: [:i, :j, :k, :l, :n, :o, :p, :q, :r]
  }

  @doc """
  Validates Q, K, and V against the kernel and returns their named dimensions.
  """
  def dims!(q, k, v, causal) do
    # A vectorized axis is not part of `shape`, but it is a leading dimension
    # of the buffer XLA passes. Validation here would see rank 4 while the
    # handler received rank 5, against layouts declaring rank 4.
    unless q.vectorized_axes == [] and k.vectorized_axes == [] and
             v.vectorized_axes == [] do
      raise ArgumentError,
            "FA3 does not accept vectorized tensors: the kernel's ABI is " <>
              "rank-4 BSHD and a vectorized axis becomes a fifth buffer " <>
              "dimension. Devectorize and fold the axis into the batch first."
    end

    {batch, seqlen_q, q_heads, head_dim} = rank4!(q, "q")
    {k_batch, seqlen_k, kv_heads, k_head_dim} = rank4!(k, "k")
    {v_batch, v_seqlen_k, v_kv_heads, value_dim} = rank4!(v, "v")

    unless k_batch == batch and v_batch == batch do
      raise ArgumentError,
            "FA3 requires one batch size across q, k, and v, got #{batch}, " <>
              "#{k_batch}, and #{v_batch}"
    end

    unless k_head_dim == head_dim do
      raise ArgumentError,
            "FA3 requires k to have q's head dimension, got #{k_head_dim} and #{head_dim}"
    end

    # The kernel indexes V with Q's head dimension, so it admits only square
    # attention. Rejected here rather than by the handler at execution.
    unless value_dim == head_dim do
      raise ArgumentError,
            "FA3 requires v's head dimension to equal q's, got #{value_dim} and #{head_dim}"
    end

    unless v_seqlen_k == seqlen_k and v_kv_heads == kv_heads do
      raise ArgumentError,
            "FA3 requires v to match k's sequence length and head count, got " <>
              "{#{v_seqlen_k}, #{v_kv_heads}} and {#{seqlen_k}, #{kv_heads}}"
    end

    unless q.type == k.type and q.type == v.type do
      raise ArgumentError,
            "FA3 requires q, k, and v to have one dtype, got #{inspect(q.type)}, " <>
              "#{inspect(k.type)}, and #{inspect(v.type)}"
    end

    unless Map.has_key?(@forward.targets, q.type) do
      raise ArgumentError, "FA3 supports BF16 and FP16, got: #{inspect(q.type)}"
    end

    unless head_dim in @head_dims do
      raise ArgumentError,
            "FA3 supports head dimensions #{inspect(@head_dims)}, got: #{head_dim}"
    end

    unless rem(q_heads, kv_heads) == 0 do
      raise ArgumentError,
            "FA3 GQA requires q_heads to be divisible by kv_heads, got #{q_heads} " <>
              "and #{kv_heads}"
    end

    if causal and seqlen_q != seqlen_k do
      raise ArgumentError,
            "causal FA3 requires equal q/k sequence lengths, got #{seqlen_q} and #{seqlen_k}"
    end

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

  @doc """
  Returns true when the kernel can run this dtype and head dimension.
  """
  def supported?(type, head_dim),
    do: Map.has_key?(@forward.targets, type) and head_dim in @head_dims

  @doc """
  Asserts that the backward operands match the forward result contract.
  """
  def backward_operands!(dims, output, lse, doutput, type) do
    output_shape = {dims.batch, dims.seqlen_q, dims.q_heads, dims.value_dim}

    unless output.shape == output_shape and output.type == type and
             lse.shape == {dims.batch, dims.q_heads, dims.seqlen_q} and lse.type == {:f, 32} and
             doutput.shape == output_shape and doutput.type == type do
      raise ArgumentError,
            "FA3 backward requires O/dO to match the output and FP32 LSE in BHQ order"
    end

    :ok
  end

  @doc """
  Returns the rounded extents that size the backward scratch buffers.

  This mirrors the tile selection in the FA3 backward kernel. The scratch
  buffers are declared as results rather than allocated by the handler so that
  they stay capturable in a CUDA command buffer, which is why their geometry
  has to be known here.
  """
  def scratch_extents(%{head_dim: head_dim, seqlen_q: seqlen_q, seqlen_k: seqlen_k}, causal) do
    block_m = if head_dim <= 128, do: if(causal, do: 64, else: 80), else: 64
    block_n = if head_dim <= 128, do: 128, else: 80
    seqlen_q_rounded = round_up(seqlen_q, block_m)

    %{
      seqlen_q_rounded: seqlen_q_rounded,
      seqlen_k_rounded: round_up(seqlen_k, block_n),
      q_blocks: div(seqlen_q_rounded, block_m)
    }
  end

  @doc """
  Describes the forward call to XLA.
  """
  def forward([q, k, v], causal, softmax_scale) do
    dims = dims!(q, k, v, causal)
    spec(@forward, q.type, sizes(dims), causal, softmax_scale)
  end

  @doc """
  Describes the backward call to XLA.

  O, LSE, and dO introduce no new dimensions; their mappings are written in
  terms of the forward variables. They are named so that the native call's
  arity is stated once.
  """
  def backward([q, k, v, _output, _lse, _doutput], causal, softmax_scale) do
    dims = dims!(q, k, v, causal)
    scratch = scratch_extents(dims, causal)

    sizes =
      sizes(dims) ++
        [
          p: scratch.seqlen_q_rounded,
          q: scratch.seqlen_k_rounded,
          r: scratch.q_blocks
        ]

    spec(@backward, q.type, sizes, causal, softmax_scale)
  end

  defp spec(call, type, sizes, causal, softmax_scale) do
    %Spec{
      call_target_name: Map.fetch!(call.targets, type),
      attributes: handler_attributes(causal, softmax_scale),
      mlir_attributes: [
        {"operand_layouts", layouts(call.operands)},
        {"result_layouts", layouts(call.results)},
        {"sdy.sharding_rule", sharding_rule(call, sizes)}
      ]
    }
  end

  # Interpolated straight into MLIR, so the scale has to render as a float
  # literal. An integer parses as `1 : f32` and nil renders as nothing at all,
  # both of which fail the MLIR parser rather than this function.
  defp handler_attributes(causal, softmax_scale)
       when is_boolean(causal) and is_number(softmax_scale) do
    [
      {"causal", to_string(causal)},
      {"softmax_scale", "#{softmax_scale * 1.0} : f32"}
    ]
  end

  defp handler_attributes(causal, softmax_scale) do
    raise ArgumentError,
          "FA3 requires a boolean causal and a numeric softmax_scale, got " <>
            "#{inspect(causal)} and #{inspect(softmax_scale)}"
  end

  defp layouts(mappings),
    do: "[" <> Enum.map_join(mappings, ", ", &row_major(length(&1))) <> "]"

  defp row_major(rank),
    do: "dense<[#{Enum.join((rank - 1)..0//-1, ", ")}]> : tensor<#{rank}xindex>"

  defp sharding_rule(call, sizes) do
    "#sdy.op_sharding_rule<" <>
      "(#{mappings(call.operands)})->(#{mappings(call.results)}) " <>
      "{#{Enum.map_join(sizes, ", ", fn {dim, size} -> "#{dim}=#{size}" end)}} " <>
      "need_replication={#{Enum.join(call.replicate, ", ")}}, custom>"
  end

  defp mappings(list),
    do: Enum.map_join(list, ", ", &("[" <> Enum.join(&1, ", ") <> "]"))

  defp sizes(dims) do
    [
      i: dims.batch,
      j: dims.seqlen_q,
      k: dims.seqlen_k,
      l: dims.groups,
      m: dims.kv_heads,
      n: dims.head_dim,
      o: dims.value_dim
    ]
  end

  defp round_up(value, block), do: div(value + block - 1, block) * block

  defp rank4!(%Nx.Tensor{shape: {a, b, c, d}}, _name), do: {a, b, c, d}

  defp rank4!(%Nx.Tensor{shape: shape}, name) do
    raise ArgumentError,
          "#{name} must have shape {batch, sequence, heads, dim}, got #{inspect(shape)}"
  end
end
