defmodule FlashAttention3.LoweringTest do
  use ExUnit.Case, async: true

  alias EXLA.CustomCall.Spec
  alias FlashAttention3.Lowering

  defp bf16(shape), do: Nx.template(shape, {:bf, 16})

  test "forward selects the precision target and describes the operation" do
    q = bf16({2, 16, 8, 128})
    k = bf16({2, 16, 2, 128})
    v = bf16({2, 16, 2, 128})

    assert %Spec{
             call_target_name: "fa3_forward_bf16",
             attributes: [{"causal", "true"}, {"softmax_scale", "0.125 : f32"}],
             operation_attributes: operation_attributes
           } = Lowering.forward(q, k, v, true, 0.125)

    assert {"operand_layouts",
            "[dense<[3, 2, 1, 0]> : tensor<4xindex>, " <>
              "dense<[3, 2, 1, 0]> : tensor<4xindex>, " <>
              "dense<[3, 2, 1, 0]> : tensor<4xindex>]"} in operation_attributes

    assert {"result_layouts",
            "[dense<[3, 2, 1, 0]> : tensor<4xindex>, " <>
              "dense<[2, 1, 0]> : tensor<3xindex>, " <>
              "dense<[0]> : tensor<1xindex>]"} in operation_attributes

    assert {"sdy.sharding_rule", sharding_rule} =
             List.keyfind(operation_attributes, "sdy.sharding_rule", 0)

    assert sharding_rule =~ "([i, j, lm, n], [i, k, m, n], [i, k, m, o])->"
    assert sharding_rule =~ "([i, j, lm, o], [i, lm, j], [i])"
    assert sharding_rule =~ "{i=2, j=16, k=16, l=4, m=2, n=128, o=128}"
    assert sharding_rule =~ "need_replication={i, j, k, l, n, o}, custom>"

    assert %Spec{call_target_name: "fa3_forward_f16"} =
             Lowering.forward(
               Nx.template({2, 16, 8, 128}, {:f, 16}),
               Nx.template({2, 16, 2, 128}, {:f, 16}),
               Nx.template({2, 16, 2, 128}, {:f, 16}),
               false,
               0.125
             )
  end

  test "backward describes six operands and nine results" do
    q = bf16({2, 16, 8, 128})
    k = bf16({2, 16, 2, 128})
    v = bf16({2, 16, 2, 128})

    assert %Spec{
             call_target_name: "fa3_backward_bf16",
             attributes: [{"causal", "true"}, {"softmax_scale", "0.125 : f32"}],
             operation_attributes: operation_attributes
           } = Lowering.backward(q, k, v, true, 0.125)

    assert {"operand_layouts", operand_layouts} =
             List.keyfind(operation_attributes, "operand_layouts", 0)

    assert {"result_layouts", result_layouts} =
             List.keyfind(operation_attributes, "result_layouts", 0)

    assert length(String.split(operand_layouts, "dense<")) - 1 == 6
    assert length(String.split(result_layouts, "dense<")) - 1 == 9

    assert {"sdy.sharding_rule", sharding_rule} =
             List.keyfind(operation_attributes, "sdy.sharding_rule", 0)

    # seqlen_q rounds to 64 with the causal head_dim-128 block, and 16 keys
    # round to 128, giving one query block.
    assert sharding_rule =~ "p=64, q=128, r=1}"
    assert sharding_rule =~ "need_replication={i, j, k, l, n, o, p, q, r}, custom>"

    assert %Spec{call_target_name: "fa3_backward_f16"} =
             Lowering.backward(
               Nx.template({2, 16, 8, 128}, {:f, 16}),
               Nx.template({2, 16, 2, 128}, {:f, 16}),
               Nx.template({2, 16, 2, 128}, {:f, 16}),
               false,
               0.125
             )
  end

  test "the kernel's dtype and head dimension limits are caller errors" do
    f32 = fn shape -> Nx.template(shape, {:f, 32}) end

    assert_raise ArgumentError, ~r/supports BF16 and FP16/, fn ->
      Lowering.forward(
        f32.({1, 8, 8, 128}),
        f32.({1, 8, 2, 128}),
        f32.({1, 8, 2, 128}),
        true,
        0.5
      )
    end

    assert_raise ArgumentError, ~r/supports head dimensions \[128, 256\]/, fn ->
      Lowering.forward(bf16({1, 8, 8, 64}), bf16({1, 8, 2, 64}), bf16({1, 8, 2, 64}), true, 0.125)
    end

    refute Lowering.supported?({:f, 32}, 128)
    refute Lowering.supported?({:bf, 16}, 64)
    assert Lowering.supported?({:bf, 16}, 128)
    assert Lowering.supported?({:f, 16}, 256)
  end
end
