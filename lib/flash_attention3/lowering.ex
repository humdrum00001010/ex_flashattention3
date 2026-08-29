defmodule FlashAttention3.Lowering do
  @moduledoc false

  alias EXLA.CustomCall.Spec
  alias FlashAttention3.{Shape, StableHLO}

  @head_dims [128, 256]

  # Shardy dimension variables, shared by both directions:
  #
  #   i batch      j seqlen_q   k seqlen_k   lm q_heads (l groups x m kv_heads)
  #   n head_dim   o value_dim
  #
  # Backward adds the rounded workspace extents:
  #
  #   p seqlen_q_rounded   q seqlen_k_rounded   r q_blocks
  #
  # Each mapping's length is the buffer's rank, so the layout constraints are
  # derived from these rather than repeated alongside them.

  @forward %{
    targets: %{{:bf, 16} => "fa3_forward_bf16", {:f, 16} => "fa3_forward_f16"},
    operands: [[:i, :j, :lm, :n], [:i, :k, :m, :n], [:i, :k, :m, :o]],
    results: [[:i, :j, :lm, :o], [:i, :lm, :j], [:i]],
    replicate: [:i, :j, :k, :l, :n, :o]
  }

  @backward %{
    targets: %{{:bf, 16} => "fa3_backward_bf16", {:f, 16} => "fa3_backward_f16"},
    operands: [
      [:i, :j, :lm, :n],
      [:i, :k, :m, :n],
      [:i, :k, :m, :o],
      [:i, :j, :lm, :o],
      [:i, :lm, :j],
      [:i, :j, :lm, :o]
    ],
    results: [
      [:i, :j, :lm, :n],
      [:i, :k, :m, :n],
      [:i, :k, :m, :o],
      [:i, :lm, :p],
      [:i, :lm, :p],
      [:i, :lm, :p, :n],
      [:r, :i, :lm],
      [:i, :m, :q, :n],
      [:i, :m, :q, :o]
    ],
    replicate: [:i, :j, :k, :l, :n, :o, :p, :q, :r]
  }

  @doc """
  Describes the forward custom call, or raises when the kernel cannot run it.

  Takes the operand list the protocol received, so its arity is the call's.
  This is only reached once a client has selected the native path, which is why
  an unsupported dtype or head dimension is a caller error here rather than a
  reason to fall back.
  """
  def forward([q, k, v], causal, softmax_scale) do
    dims = kernel_dims!(q, k, v, causal)
    spec(@forward, q.type, sizes(dims), causal, softmax_scale)
  end

  @doc """
  Describes the backward custom call, or raises when the kernel cannot run it.

  O, LSE, and dO do not contribute new dimensions; their mappings are written
  in terms of the forward variables. They are named here so that the arity of
  the native call is stated in one place.
  """
  def backward([q, k, v, _output, _lse, _doutput], causal, softmax_scale) do
    dims = kernel_dims!(q, k, v, causal)
    workspace = Shape.workspace(dims, causal)

    sizes =
      sizes(dims) ++
        [
          p: workspace.seqlen_q_rounded,
          q: workspace.seqlen_k_rounded,
          r: workspace.q_blocks
        ]

    spec(@backward, q.type, sizes, causal, softmax_scale)
  end

  @doc """
  Returns true when the native kernel can run this dtype and head dimension.
  """
  def supported?(type, head_dim),
    do: Map.has_key?(@forward.targets, type) and head_dim in @head_dims

  defp spec(call, type, sizes, causal, softmax_scale) do
    %Spec{
      call_target_name: Map.fetch!(call.targets, type),
      attributes: [
        {"causal", to_string(causal)},
        {"softmax_scale", "#{softmax_scale} : f32"}
      ],
      operation_attributes: [
        {"operand_layouts", StableHLO.layouts(ranks(call.operands))},
        {"result_layouts", StableHLO.layouts(ranks(call.results))},
        {"sdy.sharding_rule",
         StableHLO.sharding_rule(call.operands, call.results, sizes, call.replicate)}
      ]
    }
  end

  defp ranks(mappings), do: Enum.map(mappings, &length/1)

  defp kernel_dims!(q, k, v, causal) do
    dims = Shape.attention!(q, k, v, causal)

    unless Map.has_key?(@forward.targets, q.type) do
      raise ArgumentError, "FA3 supports BF16 and FP16, got: #{inspect(q.type)}"
    end

    unless dims.head_dim in @head_dims do
      raise ArgumentError,
            "FA3 supports head dimensions #{inspect(@head_dims)}, got: #{dims.head_dim}"
    end

    dims
  end

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
end
