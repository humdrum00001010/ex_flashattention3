defmodule FlashAttention3.Lowering do
  @moduledoc false

  alias EXLA.CustomCall.Spec
  alias FlashAttention3.{Shape, StableHLO}

  @forward_targets %{{:bf, 16} => "fa3_forward_bf16", {:f, 16} => "fa3_forward_f16"}
  @backward_targets %{{:bf, 16} => "fa3_backward_bf16", {:f, 16} => "fa3_backward_f16"}
  @head_dims [128, 256]

  @doc """
  Builds the forward custom-call spec, or raises when the kernel cannot run it.

  This is only reached once a client has selected the native path, so an
  unsupported dtype or head dimension is a caller error rather than a reason to
  fall back.
  """
  def forward(q, k, v, causal, softmax_scale) do
    dims = Shape.attention!(q, k, v, causal)
    supported!(q.type, dims.head_dim)

    %Spec{
      call_target_name: Map.fetch!(@forward_targets, q.type),
      attributes: handler_attributes(causal, softmax_scale),
      operation_attributes: [
        {"operand_layouts", StableHLO.layouts([4, 4, 4])},
        {"result_layouts", StableHLO.layouts([4, 3, 1])},
        {"sdy.sharding_rule",
         StableHLO.sharding_rule(
           [[:i, :j, :lm, :n], [:i, :k, :m, :n], [:i, :k, :m, :o]],
           [[:i, :j, :lm, :o], [:i, :lm, :j], [:i]],
           dimension_sizes(dims),
           [:i, :j, :k, :l, :n, :o]
         )}
      ]
    }
  end

  @doc """
  Builds the backward custom-call spec, or raises when the kernel cannot run it.
  """
  def backward(q, k, v, causal, softmax_scale) do
    dims = Shape.attention!(q, k, v, causal)
    supported!(q.type, dims.head_dim)
    workspace = Shape.workspace(dims, causal)

    sizes =
      dimension_sizes(dims) ++
        [
          p: workspace.seqlen_q_rounded,
          q: workspace.seqlen_k_rounded,
          r: workspace.q_blocks
        ]

    %Spec{
      call_target_name: Map.fetch!(@backward_targets, q.type),
      attributes: handler_attributes(causal, softmax_scale),
      operation_attributes: [
        {"operand_layouts", StableHLO.layouts([4, 4, 4, 4, 3, 4])},
        {"result_layouts", StableHLO.layouts([4, 4, 4, 3, 3, 4, 3, 4, 4])},
        {"sdy.sharding_rule",
         StableHLO.sharding_rule(
           [
             [:i, :j, :lm, :n],
             [:i, :k, :m, :n],
             [:i, :k, :m, :o],
             [:i, :j, :lm, :o],
             [:i, :lm, :j],
             [:i, :j, :lm, :o]
           ],
           [
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
           sizes,
           [:i, :j, :k, :l, :n, :o, :p, :q, :r]
         )}
      ]
    }
  end

  @doc """
  Returns true when the native kernel can run this dtype and head dimension.
  """
  def supported?(type, head_dim),
    do: Map.has_key?(@forward_targets, type) and head_dim in @head_dims

  defp supported!(type, head_dim) do
    unless Map.has_key?(@forward_targets, type) do
      raise ArgumentError, "FA3 supports BF16 and FP16, got: #{inspect(type)}"
    end

    unless head_dim in @head_dims do
      raise ArgumentError,
            "FA3 supports head dimensions #{inspect(@head_dims)}, got: #{head_dim}"
    end

    :ok
  end

  defp dimension_sizes(dims) do
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

  defp handler_attributes(causal, softmax_scale) do
    [
      {"causal", to_string(causal)},
      {"softmax_scale", "#{softmax_scale} : f32"}
    ]
  end
end
