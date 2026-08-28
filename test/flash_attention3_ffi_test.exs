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
end
