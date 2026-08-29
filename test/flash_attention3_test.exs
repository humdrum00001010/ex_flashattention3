defmodule FlashAttention3Test do
  use ExUnit.Case, async: true

  import Nx.Defn

  alias FlashAttention3.Reference

  defp fixtures do
    q = Nx.iota({1, 8, 8, 4}, type: {:f, 32}) |> Nx.divide(100)
    k = Nx.iota({1, 8, 2, 4}, type: {:f, 32}) |> Nx.subtract(13) |> Nx.divide(80)
    v = Nx.iota({1, 8, 2, 4}, type: {:f, 32}) |> Nx.subtract(7) |> Nx.divide(40)
    {q, k, v}
  end

  defn(wrapped(q, k, v), do: FlashAttention3.attention(q, k, v, causal: true))

  test "attention returns the output alone and matches the definition" do
    {q, k, v} = fixtures()

    output = FlashAttention3.attention(q, k, v, causal: true)
    {expected, _lse} = Reference.attention(q, k, v, causal: true)

    assert output.shape == q.shape
    assert_all_close(output, expected)
  end

  test "attention_with_lse also returns the FP32 normalizer in BHQ order" do
    {q, k, v} = fixtures()

    {output, lse} = FlashAttention3.attention_with_lse(q, k, v, causal: true)
    {expected_output, expected_lse} = Reference.attention(q, k, v, causal: true)

    assert lse.shape == {1, 8, 8}
    assert lse.type == {:f, 32}
    assert_all_close(output, expected_output)
    assert_all_close(lse, expected_lse)
  end

  test "attention is callable from defn and differentiable" do
    {q, k, v} = fixtures()
    doutput = Nx.iota(q.shape, type: {:f, 32}) |> Nx.remainder(17) |> Nx.divide(19)

    actual =
      Nx.Defn.grad({q, k, v}, fn {q, k, v} ->
        wrapped(q, k, v) |> Nx.multiply(doutput) |> Nx.sum()
      end)

    expected = Reference.backward(q, k, v, doutput, causal: true)

    Enum.zip(Tuple.to_list(actual), Tuple.to_list(expected))
    |> Enum.each(fn {actual, expected} -> assert_all_close(actual, expected) end)
  end

  test "the production block skips on a host client and compiles the definition" do
    q = Nx.broadcast(Nx.tensor(0.1, type: {:bf, 16}), {1, 8, 8, 128})
    k = Nx.broadcast(Nx.tensor(0.2, type: {:bf, 16}), {1, 8, 2, 128})
    v = Nx.broadcast(Nx.tensor(0.3, type: {:bf, 16}), {1, 8, 2, 128})

    %{mlir_module: mlir} = EXLA.to_mlir_module(&wrapped/3, [q, k, v], client: :host)

    refute mlir =~ "stablehlo.custom_call @fa3_forward"
    assert mlir =~ "stablehlo.dot_general"
  end

  test "unsupported options are rejected" do
    {q, k, v} = fixtures()

    assert_raise ArgumentError, ~r/unknown keys \[:window_size\]/, fn ->
      FlashAttention3.attention(q, k, v, window_size: 256)
    end

    short = &Nx.slice_along_axis(&1, 0, 4, axis: 1)

    assert_raise ArgumentError, ~r/causal FA3 requires equal q\/k sequence lengths/, fn ->
      FlashAttention3.attention(q, short.(k), short.(v), causal: true)
    end

    assert_raise ArgumentError, ~r/v to match k's sequence length and head count/, fn ->
      FlashAttention3.attention(q, short.(k), v, causal: false)
    end
  end

  defp assert_all_close(left, right) do
    assert Nx.to_number(Nx.all_close(left, right, atol: 1.0e-6, rtol: 1.0e-6)) == 1
  end
end
