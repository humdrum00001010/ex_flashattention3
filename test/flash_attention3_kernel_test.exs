defmodule FlashAttention3.KernelTest do
  use ExUnit.Case, async: true

  alias EXLA.CustomCall.Spec
  alias FlashAttention3.Kernel

  defp bf16(shape), do: Nx.template(shape, {:bf, 16})

  defp forward_operands(type) do
    [
      Nx.template({2, 16, 8, 128}, type),
      Nx.template({2, 16, 2, 128}, type),
      Nx.template({2, 16, 2, 128}, type)
    ]
  end

  defp backward_operands(type) do
    [q, k, v] = forward_operands(type)

    [
      q,
      k,
      v,
      Nx.template({2, 16, 8, 128}, type),
      Nx.template({2, 8, 16}, {:f, 32}),
      Nx.template({2, 16, 8, 128}, type)
    ]
  end

  test "forward selects the precision target and describes the operation" do
    assert %Spec{
             call_target_name: "fa3_forward_bf16",
             attributes: [{"causal", "true"}, {"softmax_scale", "0.125 : f32"}],
             operation_attributes: operation_attributes
           } = Kernel.forward(forward_operands({:bf, 16}), true, 0.125)

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

    assert sharding_rule =~ "([i, j, ml, n], [i, k, m, n], [i, k, m, o])->"
    assert sharding_rule =~ "([i, j, ml, o], [i, ml, j], [i])"
    assert sharding_rule =~ "{i=2, j=16, k=16, l=4, m=2, n=128, o=128}"
    assert sharding_rule =~ "need_replication={i, j, k, l, n, o}, custom>"

    assert %Spec{call_target_name: "fa3_forward_f16"} =
             Kernel.forward(forward_operands({:f, 16}), false, 0.125)
  end

  test "backward describes six operands and nine results" do
    assert %Spec{
             call_target_name: "fa3_backward_bf16",
             attributes: [{"causal", "true"}, {"softmax_scale", "0.125 : f32"}],
             operation_attributes: operation_attributes
           } = Kernel.backward(backward_operands({:bf, 16}), true, 0.125)

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
             Kernel.backward(backward_operands({:f, 16}), false, 0.125)
  end

  test "the kernel's dtype and head dimension limits are caller errors" do
    assert_raise ArgumentError, ~r/supports BF16 and FP16/, fn ->
      Kernel.forward(forward_operands({:f, 32}), true, 0.5)
    end

    assert_raise ArgumentError, ~r/supports head dimensions \[128, 256\]/, fn ->
      Kernel.forward(
        [bf16({1, 8, 8, 64}), bf16({1, 8, 2, 64}), bf16({1, 8, 2, 64})],
        true,
        0.125
      )
    end

    assert_raise ArgumentError, ~r/supports BF16 and FP16/, fn ->
      Kernel.backward(backward_operands({:f, 32}), true, 0.5)
    end

    refute Kernel.supported?({:f, 32}, 128)
    refute Kernel.supported?({:bf, 16}, 64)
    assert Kernel.supported?({:bf, 16}, 128)
    assert Kernel.supported?({:f, 16}, 256)
  end
end
