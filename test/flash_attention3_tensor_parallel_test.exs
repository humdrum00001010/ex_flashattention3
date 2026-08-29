defmodule FlashAttention3.TensorParallelTest do
  use ExUnit.Case, async: true

  alias FlashAttention3.{DenseAttention, TensorParallel}

  defp fixtures do
    q = Nx.iota({1, 8, 8, 4}, type: {:f, 32}) |> Nx.divide(100)
    k = Nx.iota({1, 8, 2, 4}, type: {:f, 32}) |> Nx.subtract(13) |> Nx.divide(80)
    v = Nx.iota({1, 8, 2, 4}, type: {:f, 32}) |> Nx.subtract(7) |> Nx.divide(40)
    {q, k, v}
  end

  test "GQA head sharding reconstructs the unsharded attention result" do
    {q, k, v} = fixtures()
    {expected_output, expected_lse} = DenseAttention.attention(q, k, v, causal: true)

    {output, lse} =
      q
      |> TensorParallel.shard_inputs(k, v, 2)
      |> Enum.map(fn [q_shard, k_shard, v_shard] ->
        DenseAttention.attention(q_shard, k_shard, v_shard, causal: true)
      end)
      |> TensorParallel.assemble_outputs()

    assert_all_close(output, expected_output)
    assert_all_close(lse, expected_lse)
  end

  test "TP rejects a split KV group" do
    q = Nx.iota({1, 4, 8, 4}, type: {:f, 32})
    k = Nx.iota({1, 4, 2, 4}, type: {:f, 32})
    v = Nx.iota({1, 4, 2, 4}, type: {:f, 32})

    assert_raise ArgumentError, ~r/must keep complete KV groups/, fn ->
      TensorParallel.shard_inputs(q, k, v, 4)
    end
  end

  test "TP rejects heads that do not form GQA groups" do
    r = fn shape -> Nx.iota(shape, type: {:f, 32}) end

    assert_raise ArgumentError, ~r/divisible by kv_heads/, fn ->
      TensorParallel.shard_inputs(r.({1, 4, 6, 4}), r.({1, 4, 4, 4}), r.({1, 4, 4, 4}), 2)
    end
  end

  test "the definition's analytic backward matches automatic differentiation" do
    {q, k, v} = fixtures()
    doutput = Nx.iota(q.shape, type: {:f, 32}) |> Nx.remainder(17) |> Nx.divide(19)

    expected =
      Nx.Defn.grad({q, k, v}, fn {q, k, v} ->
        {output, _lse} = DenseAttention.attention(q, k, v, causal: true)
        output |> Nx.multiply(doutput) |> Nx.sum()
      end)

    actual = DenseAttention.backward(q, k, v, doutput, causal: true)

    Enum.zip(Tuple.to_list(actual), Tuple.to_list(expected))
    |> Enum.each(fn {actual, expected} -> assert_all_close(actual, expected) end)
  end

  @tag :integration
  test "preflight: shard_jit outputs reenter shard_jit without a host transfer" do
    mesh = %Nx.Mesh{name: "fa3_reentry", shape: {2}}

    fun =
      EXLA.shard_jit(&Nx.add(&1, 1), mesh, client: :host, input_shardings: [%{0 => [0]}])

    first = fun.([[Nx.tensor([1, 2])], [Nx.tensor([3, 4])]])
    assert Enum.map(first, & &1.data.buffer.device_id) == [0, 1]

    second = fun.(Enum.map(first, &[&1]))
    assert Enum.map(second, &Nx.to_flat_list/1) == [[3, 4], [5, 6]]
  end

  defp assert_all_close(left, right) do
    assert Nx.to_number(Nx.all_close(left, right, atol: 1.0e-5, rtol: 1.0e-5)) == 1
  end
end
