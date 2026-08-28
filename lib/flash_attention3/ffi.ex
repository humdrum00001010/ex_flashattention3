defmodule FlashAttention3.FFI do
  @moduledoc """
  XLA FFI contract implemented by the native FlashAttention-3 library.

  Each call describes both the StableHLO custom-call metadata and the complete
  native result tuple. Model code does not call this module directly; the Nx
  block adapter uses it to keep handler symbols, layouts, Shardy rules, and
  compiler-owned workspaces behind one boundary.
  """

  alias EXLA.CustomCall.Spec

  @enforce_keys [:platform, :operands, :spec, :outputs, :semantic_result_count]
  defstruct [:platform, :operands, :spec, :outputs, :semantic_result_count]

  @type t :: %__MODULE__{
          platform: :cuda,
          operands: [Nx.Tensor.t()],
          spec: Spec.t(),
          outputs: tuple(),
          semantic_result_count: pos_integer()
        }

  @forward_target "exla_fa3_forward"
  @row_major_4d "dense<[3, 2, 1, 0]> : tensor<4xindex>"

  @doc """
  Builds the native forward contract for BSHD Q, K, and V tensors.

  The result tuple is output, FP32 LSE, and an S32 scheduler workspace.
  """
  @spec forward(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), boolean(), number()) :: t()
  def forward(%Nx.Tensor{} = q, %Nx.Tensor{} = k, %Nx.Tensor{} = v, causal, softmax_scale)
      when is_boolean(causal) and is_number(softmax_scale) do
    {batch, seqlen_q, q_heads, head_dim, seqlen_k, kv_heads, value_dim} =
      qkv_shape!(q, k, v, causal)

    groups = div(q_heads, kv_heads)

    sharding_rule =
      "#sdy.op_sharding_rule<" <>
        "([i, j, lm, n], [i, k, m, n], [i, k, m, o])->" <>
        "([i, j, lm, o], [i, lm, j], [i]) " <>
        "{i=#{batch}, j=#{seqlen_q}, k=#{seqlen_k}, l=#{groups}, " <>
        "m=#{kv_heads}, n=#{head_dim}, o=#{value_dim}} " <>
        "need_replication={i, j, k, l, n, o}, custom>"

    spec = %Spec{
      call_target_name: precision_target!(@forward_target, q.type),
      attributes: handler_attributes(causal, softmax_scale),
      operation_attributes: [
        {"operand_layouts", "[#{@row_major_4d}, #{@row_major_4d}, #{@row_major_4d}]"},
        {"result_layouts",
         "[#{@row_major_4d}, dense<[2, 1, 0]> : tensor<3xindex>, " <>
           "dense<[0]> : tensor<1xindex>]"},
        {"sdy.sharding_rule", sharding_rule}
      ]
    }

    outputs = {
      Nx.template({batch, seqlen_q, q_heads, value_dim}, q.type),
      Nx.template({batch, q_heads, seqlen_q}, {:f, 32}),
      Nx.template({batch}, {:s, 32})
    }

    %__MODULE__{
      platform: :cuda,
      operands: [q, k, v],
      spec: spec,
      outputs: outputs,
      semantic_result_count: 2
    }
  end

  defp qkv_shape!(q, k, v, causal) do
    {batch, seqlen_q, q_heads, head_dim} = rank4_shape!(q, "q")
    {^batch, seqlen_k, kv_heads, ^head_dim} = rank4_shape!(k, "k")
    {^batch, ^seqlen_k, ^kv_heads, value_dim} = rank4_shape!(v, "v")

    unless q.type == k.type and q.type == v.type do
      raise ArgumentError, "FA3 FFI requires Q, K, and V to use one dtype"
    end

    unless head_dim in [128, 256] do
      raise ArgumentError, "FA3 FFI supports head dimensions 128 and 256"
    end

    unless value_dim == head_dim do
      raise ArgumentError, "FA3 FFI requires equal QK and V head dimensions"
    end

    unless rem(q_heads, kv_heads) == 0 do
      raise ArgumentError, "FA3 FFI requires complete GQA groups"
    end

    if causal and seqlen_q != seqlen_k do
      raise ArgumentError, "causal FA3 FFI requires equal Q and K sequence lengths"
    end

    {batch, seqlen_q, q_heads, head_dim, seqlen_k, kv_heads, value_dim}
  end

  defp handler_attributes(causal, softmax_scale) do
    [
      {"causal", to_string(causal)},
      {"softmax_scale", "#{softmax_scale} : f32"}
    ]
  end

  defp precision_target!(base, {:bf, 16}), do: base
  defp precision_target!(base, {:f, 16}), do: base <> "_f16"

  defp precision_target!(_base, type) do
    raise ArgumentError, "FA3 FFI supports BF16 and FP16, got: #{inspect(type)}"
  end

  defp rank4_shape!(%Nx.Tensor{shape: {a, b, c, d}}, _name), do: {a, b, c, d}

  defp rank4_shape!(%Nx.Tensor{shape: shape}, name) do
    raise ArgumentError,
          "#{name} must have shape {batch, sequence, heads, dim}, got #{inspect(shape)}"
  end
end
