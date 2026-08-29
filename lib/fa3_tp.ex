defmodule FA3TP do
  @moduledoc """
  Tensor-parallel harness for the torch-free FA3 experiment.

  The attention operation itself lives in `FlashAttention3`. What remains here
  is the sharding policy the two-GPU gate exercises: splitting Q/KV heads
  across partitions and reassembling the result.
  """

  alias FlashAttention3.{FFI, Reference, Shape}

  @doc false
  defdelegate forward(q, k, v, opts \\ []), to: FlashAttention3, as: :attention_with_lse

  @doc false
  def backward(q, k, v, output, lse, doutput, opts \\ []) do
    opts = Keyword.validate!(opts, causal: false, softmax_scale: nil)
    causal = Keyword.fetch!(opts, :causal)
    {_batch, _sequence, _heads, head_dim} = Shape.rank4!(q, "q")
    softmax_scale = Keyword.get(opts, :softmax_scale) || 1.0 / :math.sqrt(head_dim)

    FFI.backward(q, k, v, output, lse, doutput, causal, softmax_scale)
  end

  @doc false
  defdelegate reference(q, k, v, opts \\ []), to: Reference, as: :attention

  @doc false
  defdelegate reference_backward(q, k, v, doutput, opts \\ []), to: Reference, as: :backward

  @doc """
  Splits Q, K, and V across `partitions` along the head axis.

  Tensor parallelism here is head-parallel only, which is what the FA3 custom
  call's Shardy rule allows, so a partition must own complete GQA groups.
  """
  def shard_inputs(q, k, v, partitions) when is_integer(partitions) and partitions > 0 do
    {_batch, _seqlen_q, q_heads, _head_dim} = Shape.rank4!(q, "q")
    {_batch, _seqlen_k, kv_heads, _head_dim} = Shape.rank4!(k, "k")

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

  @doc """
  Concatenates per-partition `{output, lse}` shards back into one result.
  """
  def assemble_outputs(outputs) when is_list(outputs) do
    {output_shards, lse_shards} = Enum.unzip(outputs)
    {Nx.concatenate(output_shards, axis: 2), Nx.concatenate(lse_shards, axis: 1)}
  end
end
