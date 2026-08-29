defmodule FlashAttention3.Shape do
  @moduledoc false

  @doc """
  Validates the attention operands and returns their named dimensions.

  These are operation-level invariants, independent of any implementation. The
  FA3 kernel's own limits (dtype, head dimension) live in
  `FlashAttention3.Lowering`, because they only apply when the custom call is
  actually selected.
  """
  def attention!(q, k, v, causal) do
    {batch, seqlen_q, q_heads, head_dim} = rank4!(q, "q")
    {k_batch, seqlen_k, kv_heads, k_head_dim} = rank4!(k, "k")
    {v_batch, v_seqlen_k, v_kv_heads, value_dim} = rank4!(v, "v")

    unless k_batch == batch and v_batch == batch do
      raise ArgumentError,
            "FA3 requires one batch size across q, k, and v, got #{batch}, #{k_batch}, " <>
              "and #{v_batch}"
    end

    unless k_head_dim == head_dim do
      raise ArgumentError,
            "FA3 requires k to have q's head dimension, got #{k_head_dim} and #{head_dim}"
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

    unless rem(q_heads, kv_heads) == 0 do
      raise ArgumentError,
            "FA3 GQA requires q_heads to be divisible by kv_heads, got #{q_heads} and #{kv_heads}"
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
  Asserts that the backward operands match the forward result contract.
  """
  def backward!(dims, output, lse, doutput, type) do
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
  Returns the rounded sequence lengths that size the backward workspaces.

  This mirrors the tile selection in the FA3 backward kernel; see
  `native/fa3_xla.cc`. The workspaces are compiler-owned results rather than
  handler-side scratch so that they stay capturable in a CUDA command buffer,
  which means their geometry has to be known here.
  """
  def workspace(%{head_dim: head_dim, seqlen_q: seqlen_q, seqlen_k: seqlen_k}, causal) do
    block_m = if head_dim <= 128, do: if(causal, do: 64, else: 80), else: 64
    block_n = if head_dim <= 128, do: 128, else: 80
    seqlen_q_rounded = round_up(seqlen_q, block_m)

    %{
      seqlen_q_rounded: seqlen_q_rounded,
      seqlen_k_rounded: round_up(seqlen_k, block_n),
      q_blocks: div(seqlen_q_rounded, block_m)
    }
  end

  def rank4!(%Nx.Tensor{shape: {a, b, c, d}}, _name), do: {a, b, c, d}

  def rank4!(%Nx.Tensor{shape: shape}, name) do
    raise ArgumentError,
          "#{name} must have shape {batch, sequence, heads, dim}, got #{inspect(shape)}"
  end

  defp round_up(value, block), do: div(value + block - 1, block) * block
end
