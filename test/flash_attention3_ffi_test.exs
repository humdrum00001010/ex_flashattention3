defmodule FlashAttention3.FFITest do
  use ExUnit.Case, async: true

  alias EXLA.CustomCall.Spec
  alias FlashAttention3.FFI

  test "forward owns metadata and the complete native result tuple" do
    q = Nx.template({2, 16, 8, 128}, {:bf, 16})
    k = Nx.template({2, 16, 2, 128}, {:bf, 16})
    v = Nx.template({2, 16, 2, 128}, {:bf, 16})

    assert %FFI{
             platform: :cuda,
             operands: [^q, ^k, ^v],
             spec: %Spec{
               call_target_name: "exla_fa3_forward",
               attributes: [{"causal", "true"}, {"softmax_scale", "0.125 : f32"}],
               operation_attributes: operation_attributes
             },
             outputs: {output, lse, scheduler_workspace},
             semantic_result_count: 2
           } = FFI.forward(q, k, v, true, 0.125)

    assert output.shape == {2, 16, 8, 128} and output.type == {:bf, 16}
    assert lse.shape == {2, 8, 16} and lse.type == {:f, 32}
    assert scheduler_workspace.shape == {2} and scheduler_workspace.type == {:s, 32}

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

    assert sharding_rule =~ "{i=2, j=16, k=16, l=4, m=2, n=128, o=128}"

    q = Nx.template(q.shape, {:f, 16})
    k = Nx.template(k.shape, {:f, 16})
    v = Nx.template(v.shape, {:f, 16})

    assert %FFI{spec: %Spec{call_target_name: "exla_fa3_forward_f16"}} =
             FFI.forward(q, k, v, false, 0.125)
  end

  test "backward owns metadata and the complete native result tuple" do
    q = Nx.template({2, 16, 8, 128}, {:bf, 16})
    k = Nx.template({2, 16, 2, 128}, {:bf, 16})
    v = Nx.template({2, 16, 2, 128}, {:bf, 16})
    output = Nx.template(q.shape, q.type)
    lse = Nx.template({2, 8, 16}, {:f, 32})
    doutput = Nx.template(output.shape, output.type)

    assert %FFI{
             platform: :cuda,
             operands: [^q, ^k, ^v, ^output, ^lse, ^doutput],
             spec: %Spec{
               call_target_name: "exla_fa3_backward",
               attributes: [{"causal", "true"}, {"softmax_scale", "0.125 : f32"}],
               operation_attributes: operation_attributes
             },
             outputs: outputs,
             semantic_result_count: 3
           } = FFI.backward(q, k, v, output, lse, doutput, true, 0.125)

    assert tuple_size(outputs) == 9

    assert Enum.map(Tuple.to_list(outputs), &{&1.shape, &1.type}) == [
             {{2, 16, 8, 128}, {:bf, 16}},
             {{2, 16, 2, 128}, {:bf, 16}},
             {{2, 16, 2, 128}, {:bf, 16}},
             {{2, 8, 64}, {:f, 32}},
             {{2, 8, 64}, {:f, 32}},
             {{2, 8, 64, 128}, {:f, 32}},
             {{1, 2, 8}, {:s, 32}},
             {{2, 2, 128, 128}, {:f, 32}},
             {{2, 2, 128, 128}, {:f, 32}}
           ]

    assert {"operand_layouts", operand_layouts} =
             List.keyfind(operation_attributes, "operand_layouts", 0)

    assert length(Regex.scan(~r/tensor<4xindex>/, operand_layouts)) == 5
    assert length(Regex.scan(~r/tensor<3xindex>/, operand_layouts)) == 1

    assert {"result_layouts", result_layouts} =
             List.keyfind(operation_attributes, "result_layouts", 0)

    assert length(Regex.scan(~r/tensor<4xindex>/, result_layouts)) == 6
    assert length(Regex.scan(~r/tensor<3xindex>/, result_layouts)) == 3

    assert {"sdy.sharding_rule", sharding_rule} =
             List.keyfind(operation_attributes, "sdy.sharding_rule", 0)

    assert sharding_rule =~
             "{i=2, j=16, k=16, l=4, m=2, n=128, o=128, p=64, q=128, r=1}"

    assert_raise ArgumentError, ~r/requires O\/dO to match/, fn ->
      FFI.backward(
        q,
        k,
        v,
        output,
        Nx.template(lse.shape, {:bf, 16}),
        doutput,
        true,
        0.125
      )
    end

    q = Nx.template(q.shape, {:f, 16})
    k = Nx.template(k.shape, {:f, 16})
    v = Nx.template(v.shape, {:f, 16})

    assert %FFI{spec: %Spec{call_target_name: "exla_fa3_backward_f16"}} =
             FFI.backward(
               q,
               k,
               v,
               Nx.template(q.shape, q.type),
               Nx.template(lse.shape, lse.type),
               Nx.template(q.shape, q.type),
               false,
               0.125
             )
  end

  test "non-native tensors retain the fallback result contract" do
    q = Nx.template({1, 8, 8, 4}, {:f, 32})
    k = Nx.template({1, 8, 2, 4}, {:f, 32})
    v = Nx.template({1, 8, 2, 4}, {:f, 32})

    assert %FFI{spec: nil, outputs: forward_outputs, semantic_result_count: 2} =
             FFI.forward(q, k, v, true, 0.5)

    assert tuple_size(forward_outputs) == 3

    output = Nx.template(q.shape, q.type)
    lse = Nx.template({1, 8, 8}, {:f, 32})

    assert %FFI{spec: nil, outputs: backward_outputs, semantic_result_count: 3} =
             FFI.backward(q, k, v, output, lse, output, true, 0.5)

    assert tuple_size(backward_outputs) == 9
  end
end
