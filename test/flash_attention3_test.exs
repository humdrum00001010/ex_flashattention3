defmodule FlashAttention3Test do
  use ExUnit.Case, async: true

  import Nx.Defn.Kernel, only: [custom_grad: 3]

  alias FlashAttention3.{FFI, HostTestBlock}

  defp bf16(shape), do: Nx.broadcast(Nx.tensor(0.1, type: {:bf, 16}), shape)

  defp operands do
    {bf16({1, 8, 8, 128}), bf16({1, 8, 2, 128}), bf16({1, 8, 2, 128})}
  end

  describe "the operation refuses to run without the kernel" do
    test "a host client has nothing to fall back to" do
      {q, k, v} = operands()

      assert_raise ArgumentError, ~r/requires a CUDA client, got: :host/, fn ->
        EXLA.to_mlir_module(&FlashAttention3.attention(&1, &2, &3, causal: true), [q, k, v],
          client: :host
        )
      end
    end

    test "unknown options are rejected" do
      {q, k, v} = operands()

      assert_raise ArgumentError, ~r/unknown keys \[:window_size\]/, fn ->
        FlashAttention3.attention(q, k, v, window_size: 256)
      end
    end
  end

  describe "the kernel's limits are stated at the operation" do
    test "dtype, head dimension, GQA, and causal lengths" do
      f32 = &Nx.template(&1, {:f, 32})

      assert_raise ArgumentError, ~r/supports BF16 and FP16/, fn ->
        FFI.forward(f32.({1, 8, 8, 128}), f32.({1, 8, 2, 128}), f32.({1, 8, 2, 128}), true, 0.5)
      end

      assert_raise ArgumentError, ~r/supports head dimensions \[128, 256\]/, fn ->
        FFI.forward(bf16({1, 8, 8, 64}), bf16({1, 8, 2, 64}), bf16({1, 8, 2, 64}), true, 0.125)
      end

      assert_raise ArgumentError, ~r/divisible by kv_heads/, fn ->
        FFI.forward(bf16({1, 8, 7, 128}), bf16({1, 8, 2, 128}), bf16({1, 8, 2, 128}), true, 0.1)
      end

      assert_raise ArgumentError, ~r/causal FA3 requires equal q\/k sequence lengths/, fn ->
        FFI.forward(bf16({1, 8, 8, 128}), bf16({1, 4, 2, 128}), bf16({1, 4, 2, 128}), true, 0.1)
      end

      assert_raise ArgumentError, ~r/v to match k's sequence length and head count/, fn ->
        FFI.forward(bf16({1, 8, 8, 128}), bf16({1, 4, 2, 128}), bf16({1, 8, 2, 128}), false, 0.1)
      end

      # A vectorized axis is invisible to `shape` but becomes a leading buffer
      # dimension, so rank-4 validation would pass while XLA emits rank 5.
      vec = &Nx.vectorize(bf16(Tuple.insert_at(&1, 0, 2)), :b)

      assert_raise ArgumentError, ~r/does not accept vectorized tensors/, fn ->
        FFI.forward(vec.({1, 8, 8, 128}), vec.({1, 8, 2, 128}), vec.({1, 8, 2, 128}), true, 0.125)
      end
    end
  end

  describe "the emitted StableHLO" do
    test "carries the FA3 layouts and GQA sharding rule" do
      {q, k, v} = operands()

      %{mlir_module: mlir} =
        EXLA.to_mlir_module(&host_forward(&1, &2, &3, true), [q, k, v], client: :host)

      assert calls(mlir, "fa3_forward_bf16") == 1
      assert mlir =~ "operand_layouts = [dense<[3, 2, 1, 0]>"
      assert mlir =~ "result_layouts = [dense<[3, 2, 1, 0]>"
      assert mlir =~ "sdy.sharding_rule = #sdy.op_sharding_rule<"
      assert mlir =~ "[i, j, lm, n]"
      assert mlir =~ "need_replication={i, j, k, l, n, o}, custom>"
    end

    test "gradient tracing emits the backward call" do
      {q, k, v} = operands()
      doutput = bf16({1, 8, 8, 128})

      fun = fn q, k, v, doutput ->
        Nx.Defn.grad({q, k, v}, fn {q, k, v} ->
          {output, _lse} = host_forward(q, k, v, true)
          output |> Nx.multiply(doutput) |> Nx.sum()
        end)
      end

      %{mlir_module: mlir} =
        EXLA.to_mlir_module(fun, [q, k, v, doutput], client: :host)

      assert calls(mlir, "fa3_forward_bf16") == 1
      assert calls(mlir, "fa3_backward_bf16") == 1
      assert mlir =~ "result_layouts = [dense<[3, 2, 1, 0]>"

      # Both calls carry their own Shardy rule, which is what lets shard_jit
      # partition a training step and not just a forward pass. The backward
      # rule is the one naming the rounded scratch extents.
      assert mlir =~ "need_replication={i, j, k, l, n, o}, custom>"
      assert mlir =~ "need_replication={i, j, k, l, n, o, p, q, r}, custom>"
    end

    test "repeated calls stay inside one executable" do
      {q, k, v} = operands()
      doutput = bf16({1, 8, 8, 128})

      forward = fn q, k, v ->
        {output, _lse} = host_forward(q, k, v, true)
        {output, _lse} = host_forward(output, k, v, true)
        output
      end

      forward_backward = fn q, k, v, doutput ->
        Nx.Defn.grad({q, k, v}, fn {q, k, v} ->
          forward.(q, k, v) |> Nx.multiply(doutput) |> Nx.sum()
        end)
      end

      %{mlir_module: forward_mlir} = EXLA.to_mlir_module(forward, [q, k, v], client: :host)

      %{mlir_module: backward_mlir} =
        EXLA.to_mlir_module(forward_backward, [q, k, v, doutput], client: :host)

      assert calls(forward_mlir, "fa3_forward_bf16") == 2
      assert calls(backward_mlir, "fa3_forward_bf16") == 2
      assert calls(backward_mlir, "fa3_backward_bf16") == 2
    end
  end

  # MLIR may print the module in either the pretty or the generic form, so
  # match the target name in both.
  defp calls(mlir, target) do
    pattern =
      ~r/(?:custom_call @#{Regex.escape(target)}\b|call_target_name = "#{Regex.escape(target)}")/

    length(Regex.scan(pattern, mlir))
  end

  # The production block skips on a host client, so the preflight drives the
  # same FFI path through a block that lowers there instead.
  defp host_forward(q, k, v, causal) do
    softmax_scale = 1.0 / :math.sqrt(Nx.axis_size(q, 3))
    {output, lse} = FFI.forward(q, k, v, causal, softmax_scale, HostTestBlock.Forward)

    case q.data do
      %Nx.Defn.Expr{} ->
        graded =
          custom_grad(output, [q, k, v], fn doutput ->
            doutput = Nx.as_type(doutput, q.type)

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
            |> Tuple.to_list()
          end)

        {graded, lse}

      _ ->
        {output, lse}
    end
  end
end
