defmodule FA3TP.ContractTest do
  use ExUnit.Case, async: true

  import Nx.Defn.Kernel, only: [custom_grad: 3]

  alias FlashAttention3.FFI
  alias FlashAttention3.HostTestBlock

  defp fixtures do
    q = Nx.iota({1, 8, 8, 4}, type: {:f, 32}) |> Nx.divide(100)
    k = Nx.iota({1, 8, 2, 4}, type: {:f, 32}) |> Nx.subtract(13) |> Nx.divide(80)
    v = Nx.iota({1, 8, 2, 4}, type: {:f, 32}) |> Nx.subtract(7) |> Nx.divide(40)
    {q, k, v}
  end

  test "FFI result packing preserves the Nx fallback surface" do
    {q, k, v} = fixtures()
    doutput = Nx.iota(q.shape, type: {:f, 32}) |> Nx.remainder(17) |> Nx.divide(19)

    assert {output, lse} = FA3TP.forward(q, k, v, causal: true)
    assert {expected_output, expected_lse} = FA3TP.reference(q, k, v, causal: true)
    assert_all_close(output, expected_output, atol: 1.0e-6, rtol: 1.0e-6)
    assert_all_close(lse, expected_lse, atol: 1.0e-6, rtol: 1.0e-6)

    actual_gradients = FA3TP.backward(q, k, v, output, lse, doutput, causal: true)
    expected_gradients = FA3TP.reference_backward(q, k, v, doutput, causal: true)

    Enum.zip(Tuple.to_list(actual_gradients), Tuple.to_list(expected_gradients))
    |> Enum.each(fn {actual, expected} ->
      assert_all_close(actual, expected, atol: 1.0e-5, rtol: 1.0e-5)
    end)
  end

  test "GQA head sharding reconstructs the unsharded attention result" do
    {q, k, v} = fixtures()
    expected = FA3TP.reference(q, k, v, causal: true)

    actual =
      q
      |> FA3TP.shard_inputs(k, v, 2)
      |> Enum.map(fn [q_shard, k_shard, v_shard] ->
        FA3TP.reference(q_shard, k_shard, v_shard, causal: true)
      end)
      |> FA3TP.assemble_outputs()

    assert_all_close(elem(actual, 0), elem(expected, 0), atol: 1.0e-6, rtol: 1.0e-6)
    assert_all_close(elem(actual, 1), elem(expected, 1), atol: 1.0e-6, rtol: 1.0e-6)
  end

  test "TP rejects a split KV group" do
    q = Nx.iota({1, 4, 8, 4}, type: {:f, 32})
    k = Nx.iota({1, 4, 2, 4}, type: {:f, 32})
    v = Nx.iota({1, 4, 2, 4}, type: {:f, 32})

    assert_raise ArgumentError, ~r/must keep complete KV groups/, fn ->
      FA3TP.shard_inputs(q, k, v, 4)
    end
  end

  test "reference backward matches automatic differentiation" do
    {q, k, v} = fixtures()
    doutput = Nx.iota(q.shape, type: {:f, 32}) |> Nx.remainder(17) |> Nx.divide(19)

    expected =
      Nx.Defn.grad({q, k, v}, fn {q, k, v} ->
        {output, _lse} = FA3TP.reference(q, k, v, causal: true)
        output |> Nx.multiply(doutput) |> Nx.sum()
      end)

    actual = FA3TP.reference_backward(q, k, v, doutput, causal: true)

    Enum.zip(Tuple.to_list(actual), Tuple.to_list(expected))
    |> Enum.each(fn {actual, expected} ->
      assert_all_close(actual, expected, atol: 1.0e-5, rtol: 1.0e-5)
    end)
  end

  test "the real Nx.block path emits the FA3 layouts and GQA sharding rule" do
    q = Nx.broadcast(Nx.tensor(0.1, type: {:bf, 16}), {1, 8, 8, 128})
    k = Nx.broadcast(Nx.tensor(0.2, type: {:bf, 16}), {1, 8, 2, 128})
    v = Nx.broadcast(Nx.tensor(0.3, type: {:bf, 16}), {1, 8, 2, 128})

    fun = &host_forward(&1, &2, &3, true)

    %{mlir_module: mlir} = EXLA.to_mlir_module(fun, [q, k, v], client: :host)

    assert mlir =~ "stablehlo.custom_call @fa3_forward_bf16"
    assert mlir =~ "operand_layouts = [dense<[3, 2, 1, 0]>"
    assert mlir =~ "result_layouts = [dense<[3, 2, 1, 0]>"
    assert mlir =~ "sdy.sharding_rule = #sdy.op_sharding_rule<"
    assert mlir =~ "[i, j, lm, n]"
    assert mlir =~ "need_replication={i, j, k, l, n, o}, custom>"
  end

  test "gradient tracing emits the FA3 backward custom call" do
    q = Nx.broadcast(Nx.tensor(0.1, type: {:bf, 16}), {1, 8, 8, 128})
    k = Nx.broadcast(Nx.tensor(0.2, type: {:bf, 16}), {1, 8, 2, 128})
    v = Nx.broadcast(Nx.tensor(0.3, type: {:bf, 16}), {1, 8, 2, 128})
    doutput = Nx.broadcast(Nx.tensor(0.4, type: {:bf, 16}), q.shape)

    fun = fn q, k, v, doutput ->
      Nx.Defn.grad({q, k, v}, fn {q, k, v} ->
        {output, _lse} = host_forward(q, k, v, true)

        output |> Nx.multiply(doutput) |> Nx.sum()
      end)
    end

    %{mlir_module: mlir} =
      EXLA.to_mlir_module(fun, [q, k, v, doutput], client: :host)

    assert mlir =~ "stablehlo.custom_call @fa3_forward_bf16"
    assert mlir =~ "stablehlo.custom_call @fa3_backward_bf16"
    assert mlir =~ "result_layouts = [dense<[3, 2, 1, 0]>"
  end

  test "compute benchmark keeps repeated FA3 calls inside one StableHLO executable" do
    q = Nx.broadcast(Nx.tensor(0.1, type: {:bf, 16}), {1, 8, 8, 128})
    k = Nx.broadcast(Nx.tensor(0.2, type: {:bf, 16}), {1, 8, 2, 128})
    v = Nx.broadcast(Nx.tensor(0.3, type: {:bf, 16}), {1, 8, 2, 128})
    doutput = Nx.broadcast(Nx.tensor(0.4, type: {:bf, 16}), q.shape)

    forward = fn q, k, v ->
      {output, _lse} = host_forward(q, k, v, true)
      {output, _lse} = host_forward(output, k, v, true)
      output
    end

    forward_backward = fn q, k, v, doutput ->
      Nx.Defn.grad({q, k, v}, fn {q, k, v} ->
        output = forward.(q, k, v)
        output |> Nx.multiply(doutput) |> Nx.sum()
      end)
    end

    %{mlir_module: forward_mlir} =
      EXLA.to_mlir_module(forward, [q, k, v], client: :host)

    %{mlir_module: backward_mlir} =
      EXLA.to_mlir_module(forward_backward, [q, k, v, doutput], client: :host)

    assert length(Regex.scan(~r/stablehlo\.custom_call @fa3_forward_bf16/, forward_mlir)) ==
             2

    assert length(Regex.scan(~r/stablehlo\.custom_call @fa3_forward_bf16/, backward_mlir)) ==
             2

    assert length(Regex.scan(~r/stablehlo\.custom_call @fa3_backward_bf16/, backward_mlir)) ==
             2
  end

  @tag :integration
  test "preflight: shard_jit outputs reenter shard_jit without a host transfer" do
    mesh = %Nx.Mesh{name: "fa3_reentry", shape: {2}}

    fun =
      EXLA.shard_jit(&Nx.add(&1, 1), mesh,
        client: :host,
        input_shardings: [%{0 => [0]}]
      )

    first = fun.([[Nx.tensor([1, 2])], [Nx.tensor([3, 4])]])
    assert Enum.map(first, & &1.data.buffer.device_id) == [0, 1]

    second = fun.(Enum.map(first, &[&1]))
    assert Enum.map(second, &Nx.to_flat_list/1) == [[3, 4], [5, 6]]
  end

  defp host_forward(q, k, v, causal) do
    {_batch, _sequence, _heads, head_dim} = q.shape
    softmax_scale = 1.0 / :math.sqrt(head_dim)

    {output, lse} = FFI.forward(q, k, v, causal, softmax_scale, HostTestBlock.Forward)

    case q.data do
      %Nx.Defn.Expr{} ->
        graded =
          custom_grad(output, [q, k, v], fn doutput ->
            doutput = Nx.as_type(doutput, q.type)

            q
            |> host_backward(k, v, output, lse, doutput, causal, softmax_scale)
            |> Tuple.to_list()
          end)

        {graded, lse}

      _ ->
        {output, lse}
    end
  end

  defp host_backward(q, k, v, output, lse, doutput, causal, softmax_scale) do
    FFI.backward(
      q,
      k,
      v,
      output,
      lse,
      doutput,
      causal,
      softmax_scale,
      HostTestBlock.Backward
    )
  end

  defp assert_all_close(left, right, opts) do
    assert Nx.to_number(Nx.all_close(left, right, opts)) == 1
  end
end
