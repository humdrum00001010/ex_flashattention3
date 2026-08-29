defmodule FlashAttention3.StableHLO do
  @moduledoc false

  @doc """
  Renders row-major layout constraints for buffers of the given ranks.
  """
  def layouts(ranks), do: "[" <> Enum.map_join(ranks, ", ", &row_major/1) <> "]"

  @doc """
  Renders a single row-major layout for a buffer of the given rank.
  """
  def row_major(rank),
    do: "dense<[#{Enum.join((rank - 1)..0//-1, ", ")}]> : tensor<#{rank}xindex>"

  @doc """
  Renders a Shardy sharding rule from operand and result dimension mappings.
  """
  def sharding_rule(operands, results, sizes, replicate) do
    "#sdy.op_sharding_rule<" <>
      "(#{mappings(operands)})->(#{mappings(results)}) " <>
      "{#{Enum.map_join(sizes, ", ", fn {dim, size} -> "#{dim}=#{size}" end)}} " <>
      "need_replication={#{Enum.join(replicate, ", ")}}, custom>"
  end

  defp mappings(buffers),
    do: Enum.map_join(buffers, ", ", &("[" <> Enum.join(&1, ", ") <> "]"))
end
