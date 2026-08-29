defmodule FlashAttention3.TensorParallel do
  @moduledoc """
  Head-parallel sharding policy for FA3.

  The kernel's Shardy rule replicates every dimension except KV heads, so a
  partition must own complete GQA groups. Splitting a group would need a
  collective inside attention, which this custom call does not express.
  """

  @doc """
  Splits Q, K, and V across `partitions` along the head axis.

  This is a sharding policy over head counts, so it does not impose the
  kernel's dtype or head-dimension limits; those are checked when the call is
  actually described.
  """
  def shard_inputs(q, k, v, partitions) when is_integer(partitions) and partitions > 0 do
    q_heads = Nx.axis_size(q, 2)
    kv_heads = Nx.axis_size(k, 2)

    unless rem(q_heads, kv_heads) == 0 do
      raise ArgumentError,
            "FA3 GQA requires q_heads to be divisible by kv_heads, got #{q_heads} " <>
              "and #{kv_heads}"
    end

    unless rem(kv_heads, partitions) == 0 do
      raise ArgumentError,
            "TP must keep complete KV groups: #{kv_heads} KV heads cannot be " <>
              "split over #{partitions} partitions"
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
