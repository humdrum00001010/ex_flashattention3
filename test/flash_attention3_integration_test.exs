defmodule FlashAttention3.IntegrationTest do
  use ExUnit.Case, async: false

  @moduletag :integration
  @moduletag timeout: 600_000

  # Run on a fresh 2xH100 process (XLA_FLAGS is read at VM initialization):
  #
  #   XLA_TARGET=cuda13 \
  #   FA3_TP_DYLIB=/absolute/path/libfa3_xla.so \
  #   FA3_TP_HLO_DIR=/fresh/hlo/directory \
  #   XLA_FLAGS="--xla_dump_to=/fresh/hlo/directory --xla_dump_hlo_as_text --xla_dump_hlo_pass_re=spmd|propagation|layout" \
  #   mix test test/fa3_tp_integration_test.exs --include integration
  #

  # External-library contract exercised by this test:
  #
  #   * CUDA FFI target: fa3_forward_bf16
  #   * operands: Q/K/V bf16, logical [batch, sequence, heads, dim]
  #   * results: O bf16 [batch, q_sequence, q_heads, value_dim]
  #              LSE f32 [batch, q_heads, q_sequence]
  #   * backend_config attrs: causal: bool, softmax_scale: f32
  #   * the dylib also registers the target's CustomCallPartitioner

  test "single-device correctness -> TP correctness -> resident reentry" do
    client = EXLA.Client.fetch!(:cuda)
    assert client.device_count >= 2, "FA3 TP2 requires two physical CUDA devices"
    assert_hopper_topology!()

    load_external_fa3!()
    hlo_dir = fresh_hlo_dir!()

    target = "fa3_forward_bf16"
    causal = env_boolean("FA3_TP_CAUSAL", true)
    mesh = %Nx.Mesh{name: "fa3_tp", shape: {2}}
    input_shardings = [%{2 => [0]}, %{2 => [0]}, %{2 => [0]}]

    {q_f32, k_f32, v_f32} = deterministic_inputs({1, 64, 8, 2, 128})
    q = Nx.as_type(q_f32, {:bf, 16})
    k = Nx.as_type(k_f32, {:bf, 16})
    v = Nx.as_type(v_f32, {:bf, 16})

    forward = fn q, k, v ->
      FlashAttention3.attention_with_lse(q, k, v, causal: causal)
    end

    # Gate 1: the native target itself must match an independent FP32 Nx oracle.
    single = EXLA.jit(forward, client: :cuda).(q, k, v) |> to_host()
    assert_matches_oracle!(single, {q_f32, k_f32, v_f32}, {q, k, v}, causal)

    # Gate 2: TP receives complete KV groups, executes one local FA3 per rank,
    # and reconstructs the same global O/LSE without a collective inside FA3.
    host_shards = FlashAttention3.TensorParallel.shard_inputs(q, k, v, 2)

    tp_forward =
      EXLA.shard_jit(forward, mesh,
        client: :cuda,
        input_shardings: input_shardings
      )

    sharded = tp_forward.(host_shards)
    assert Enum.map(sharded, fn {output, _lse} -> output.data.buffer.device_id end) == [0, 1]

    assembled =
      sharded |> Enum.map(&to_host/1) |> FlashAttention3.TensorParallel.assemble_outputs()

    assert_matches_oracle!(assembled, {q_f32, k_f32, v_f32}, {q, k, v}, causal)
    audit_attention_hlo!(hlo_dir, target)

    # Gate 3: a complete TP attention layer reduces after the local FA3 output,
    # not inside FA3. Shard the output-projection contracting dimension and
    # require XLA/NCCL to produce the replicated dense result on both ranks.
    model_dim = 16
    projection_f32 = deterministic_tensor({8 * 128, model_dim}, 19, 9, 10)
    projection = Nx.as_type(projection_f32, {:bf, 16})
    projection_per_rank = div(elem(projection.shape, 0), 2)

    layer_args =
      host_shards
      |> Enum.with_index()
      |> Enum.map(fn {qkv, rank} ->
        qkv ++
          [
            Nx.slice_along_axis(
              projection,
              rank * projection_per_rank,
              projection_per_rank,
              axis: 0
            )
          ]
      end)

    layer = fn q, k, v, weight ->
      {output, _lse} = forward.(q, k, v)
      {batch, seqlen, heads, dim} = output.shape
      output |> Nx.reshape({batch, seqlen, heads * dim}) |> Nx.dot(weight)
    end

    tp_layer =
      EXLA.shard_jit(layer, mesh,
        client: :cuda,
        input_shardings: input_shardings ++ [%{0 => [0]}]
      )

    layer_results = tp_layer.(layer_args)
    assert Enum.map(layer_results, & &1.data.buffer.device_id) == [0, 1]

    {expected_output, _expected_lse} =
      FlashAttention3.DenseAttention.attention(q_f32, k_f32, v_f32, causal: causal)

    {baseline_output, _baseline_lse} =
      FlashAttention3.DenseAttention.attention(q, k, v, causal: causal)

    expected_projection =
      expected_output
      |> Nx.reshape({1, 64, 8 * 128})
      |> Nx.dot(projection_f32)

    baseline_projection =
      baseline_output
      |> Nx.reshape({1, 64, 8 * 128})
      |> Nx.dot(projection)
      |> Nx.as_type({:f, 32})

    Enum.each(layer_results, fn result ->
      assert_tensor_matches_oracle!(
        result |> to_host() |> Nx.as_type({:f, 32}),
        expected_projection,
        baseline_projection
      )
    end)

    audit_projection_hlo!(hlo_dir, target)

    # Gate 4: BF16 and FP16 native forward/backward, including TP2 gradients,
    # must remain within the corresponding low-precision reference error.
    assert_forward_backward_precisions!(causal, mesh, input_shardings)

    # Gate 5: timing is forbidden until a shard_jit-produced buffer can be fed
    # back to shard_jit. This catches executable device_id=-1 vs physical buffer
    # device_id=0/1 instead of hiding H2D in timings.
    stage_on_devices =
      EXLA.shard_jit(fn q, k, v -> {q, k, v} end, mesh,
        client: :cuda,
        input_shardings: input_shardings
      )

    resident_shards = stage_on_devices.(host_shards)
    resident_args = Enum.map(resident_shards, &Tuple.to_list/1)

    resident_result =
      try do
        tp_forward.(resident_args)
      rescue
        error ->
          flunk("""
          shard_jit output reentry failed before benchmarking.
          This is an EXLA buffer/executable placement failure, not an FA3 result:
          #{Exception.format(:error, error, __STACKTRACE__)}
          """)
      end

    resident_assembled =
      resident_result |> Enum.map(&to_host/1) |> FlashAttention3.TensorParallel.assemble_outputs()

    assert_matches_oracle!(resident_assembled, {q_f32, k_f32, v_f32}, {q, k, v}, causal)
  end

  defp load_external_fa3! do
    path =
      System.get_env("FA3_TP_DYLIB") ||
        flunk("set FA3_TP_DYLIB to the torch-free FA3 XLA FFI shared library")

    unless File.regular?(path) do
      flunk("FA3_TP_DYLIB does not name a regular file: #{path}")
    end

    assert :ok = EXLA.load_dylib(path)
  end

  defp assert_hopper_topology! do
    executable =
      System.find_executable("nvidia-smi") ||
        flunk("nvidia-smi is required to prove the CUDA devices are physical Hopper GPUs")

    {inventory, 0} =
      System.cmd(
        executable,
        [
          "--query-gpu=index,name,uuid,compute_cap,mig.mode.current,memory.total",
          "--format=csv,noheader,nounits"
        ],
        stderr_to_stdout: true
      )

    devices =
      inventory
      |> String.split("\n", trim: true)
      |> Enum.map(fn line -> line |> String.split(",") |> Enum.map(&String.trim/1) end)

    assert length(devices) >= 2,
           "nvidia-smi reported fewer than two physical GPUs:\n#{inventory}"

    selected = Enum.take(devices, 2)
    assert selected |> Enum.map(&Enum.at(&1, 2)) |> Enum.uniq() |> length() == 2

    Enum.each(selected, fn [index, name, _uuid, compute_capability, mig_mode, memory_mb] ->
      assert compute_capability == "9.0",
             "GPU #{index} (#{name}) is compute capability #{compute_capability}; FA3 requires Hopper 9.0"

      assert mig_mode == "Disabled",
             "GPU #{index} (#{name}) has MIG #{mig_mode}; TP requires distinct physical GPUs"

      IO.puts(
        "FA3_TP_GPU index=#{index} name=#{name} compute_capability=#{compute_capability} " <>
          "memory_mb=#{memory_mb}"
      )
    end)

    case System.cmd(executable, ["topo", "-m"], stderr_to_stdout: true) do
      {topology, 0} -> IO.puts("FA3_TP_TOPOLOGY\n#{topology}")
      {output, status} -> flunk("nvidia-smi topo -m failed (#{status}):\n#{output}")
    end
  end

  defp fresh_hlo_dir! do
    path =
      System.get_env("FA3_TP_HLO_DIR") ||
        flunk("""
        set FA3_TP_HLO_DIR to a fresh directory and start the VM with:
        XLA_FLAGS="--xla_dump_to=DIR --xla_dump_hlo_as_text --xla_dump_hlo_pass_re=spmd|propagation|layout"
        """)

    File.mkdir_p!(path)

    unless Path.wildcard(Path.join(path, "**/*")) == [] do
      flunk("FA3_TP_HLO_DIR must be empty so stale compiler evidence cannot pass: #{path}")
    end

    path
  end

  defp audit_attention_hlo!(hlo_dir, target) do
    candidates =
      hlo_dir
      |> Path.join("**/*after_optimizations*.txt")
      |> Path.wildcard()
      |> Enum.flat_map(fn path ->
        contents = File.read!(path)
        if contents =~ target, do: [{path, contents}], else: []
      end)

    assert candidates != [],
           "no optimized HLO containing #{inspect(target)} was dumped under #{hlo_dir}"

    {_path, tp_hlo} =
      Enum.find(candidates, fn {_path, hlo} ->
        hlo =~ "bf16[1,64,4,128]{3,2,1,0" and
          hlo =~ "bf16[1,64,1,128]{3,2,1,0"
      end) ||
        flunk("optimized HLO never exposed the TP2-local Q/K layouts")

    assert tp_hlo =~ "custom_call_target=\"#{target}\""

    refute Regex.match?(~r/\b(all-reduce|all-gather|collective-permute)\b/, tp_hlo),
           "attention-only FA3 inserted a collective; head-local FA3 must not communicate"
  end

  defp audit_projection_hlo!(hlo_dir, target) do
    projection_hlos =
      hlo_dir
      |> Path.join("**/*after_optimizations*.txt")
      |> Path.wildcard()
      |> Enum.map(&File.read!/1)
      |> Enum.filter(&(&1 =~ target and &1 =~ "all-reduce"))

    assert projection_hlos != [],
           "the sharded output projection produced no all-reduce after FA3"
  end

  defp assert_forward_backward_precisions!(causal, mesh, input_shardings) do
    {q_f32, k_f32, v_f32} = deterministic_inputs({1, 32, 8, 2, 128})
    doutput_f32 = deterministic_tensor(q_f32.shape, 17, 8, 9)

    reference_gradient = fn q, k, v, doutput ->
      Nx.Defn.grad({q, k, v}, fn {q, k, v} ->
        {output, _lse} = FlashAttention3.DenseAttention.attention(q, k, v, causal: causal)
        output |> Nx.multiply(doutput) |> Nx.sum()
      end)
    end

    low_precision_gradient = fn q, k, v, doutput ->
      Nx.Defn.grad({q, k, v}, fn {q, k, v} ->
        {output, _lse} =
          FlashAttention3.DenseAttention.attention(q, k, v, causal: causal, upcast: false)

        output |> Nx.multiply(doutput) |> Nx.sum()
      end)
    end

    expected_gradients =
      EXLA.jit(reference_gradient, client: :cuda).(q_f32, k_f32, v_f32, doutput_f32)
      |> to_host()

    for {label, type} <- [bf16: {:bf, 16}, fp16: {:f, 16}] do
      q = Nx.as_type(q_f32, type)
      k = Nx.as_type(k_f32, type)
      v = Nx.as_type(v_f32, type)
      doutput = Nx.as_type(doutput_f32, type)

      forward = fn q, k, v ->
        FlashAttention3.attention_with_lse(q, k, v, causal: causal)
      end

      actual_forward = EXLA.jit(forward, client: :cuda).(q, k, v) |> to_host()
      assert_matches_oracle!(actual_forward, {q_f32, k_f32, v_f32}, {q, k, v}, causal)

      native_gradient = fn q, k, v, doutput ->
        Nx.Defn.grad({q, k, v}, fn {q, k, v} ->
          {output, _lse} = forward.(q, k, v)
          output |> Nx.multiply(doutput) |> Nx.sum()
        end)
      end

      actual_gradients =
        EXLA.jit(native_gradient, client: :cuda).(q, k, v, doutput)
        |> to_host()

      baseline_gradients =
        EXLA.jit(low_precision_gradient, client: :cuda).(q, k, v, doutput)
        |> to_host()
        |> map_tuple_tensors(&Nx.as_type(&1, {:f, 32}))

      assert_gradient_tuple!(actual_gradients, expected_gradients, baseline_gradients)

      qkv_shards = FlashAttention3.TensorParallel.shard_inputs(q, k, v, 2)
      q_heads_per_rank = div(elem(q.shape, 2), 2)

      gradient_args =
        qkv_shards
        |> Enum.with_index()
        |> Enum.map(fn {qkv, rank} ->
          qkv ++
            [
              Nx.slice_along_axis(
                doutput,
                rank * q_heads_per_rank,
                q_heads_per_rank,
                axis: 2
              )
            ]
        end)

      tp_gradient =
        EXLA.shard_jit(native_gradient, mesh,
          client: :cuda,
          input_shardings: input_shardings ++ [%{2 => [0]}]
        )

      sharded_gradients = tp_gradient.(gradient_args) |> Enum.map(&to_host/1)

      assembled_gradients =
        for tuple_index <- 0..2 do
          sharded_gradients
          |> Enum.map(&elem(&1, tuple_index))
          |> Nx.concatenate(axis: 2)
        end
        |> List.to_tuple()

      assert_gradient_tuple!(assembled_gradients, expected_gradients, baseline_gradients)
      IO.puts("FA3_PRECISION_PASS=#{label}")
    end
  end

  defp assert_gradient_tuple!(actual, expected, baseline) do
    actual = map_tuple_tensors(actual, &Nx.as_type(&1, {:f, 32}))

    Enum.zip([Tuple.to_list(actual), Tuple.to_list(expected), Tuple.to_list(baseline)])
    |> Enum.each(fn {actual_tensor, expected_tensor, baseline_tensor} ->
      assert_tensor_matches_oracle!(actual_tensor, expected_tensor, baseline_tensor)
    end)
  end

  defp deterministic_inputs({batch, seqlen, q_heads, kv_heads, dim}) do
    q = deterministic_tensor({batch, seqlen, q_heads, dim}, 31, 15, 16)
    k = deterministic_tensor({batch, seqlen, kv_heads, dim}, 29, 14, 15)
    v = deterministic_tensor({batch, seqlen, kv_heads, dim}, 23, 11, 12)
    {q, k, v}
  end

  defp deterministic_tensor(shape, modulus, center, divisor) do
    shape
    |> Nx.iota(type: {:f, 32})
    |> Nx.remainder(modulus)
    |> Nx.subtract(center)
    |> Nx.divide(divisor)
  end

  defp assert_matches_oracle!(actual, f32_inputs, bf16_inputs, causal) do
    expected = apply_reference(f32_inputs, causal)
    bf16_baseline = apply_reference(bf16_inputs, causal) |> map_tensors(&Nx.as_type(&1, {:f, 32}))
    actual = map_tensors(actual, &Nx.as_type(&1, {:f, 32}))

    Enum.zip([Tuple.to_list(actual), Tuple.to_list(expected), Tuple.to_list(bf16_baseline)])
    |> Enum.each(fn {actual_tensor, expected_tensor, baseline_tensor} ->
      assert_tensor_matches_oracle!(actual_tensor, expected_tensor, baseline_tensor)
    end)
  end

  defp assert_tensor_matches_oracle!(actual, expected, baseline) do
    actual_error = max_abs(Nx.subtract(actual, expected))
    baseline_error = max_abs(Nx.subtract(baseline, expected))

    rounding_allowance =
      expected
      |> Nx.add(0.3)
      |> Nx.subtract(0.3)
      |> Nx.subtract(expected)
      |> max_abs()
      |> Kernel.*(2.0)

    assert actual_error <= 2.0 * baseline_error + rounding_allowance + 1.0e-5,
           "max error #{actual_error} exceeds the bf16 oracle bound " <>
             "#{2.0 * baseline_error + rounding_allowance + 1.0e-5}"
  end

  defp apply_reference({q, k, v}, causal),
    do: FlashAttention3.DenseAttention.attention(q, k, v, causal: causal)

  defp map_tensors({left, right}, fun), do: {fun.(left), fun.(right)}

  defp map_tuple_tensors(tuple, fun) do
    tuple |> Tuple.to_list() |> Enum.map(fun) |> List.to_tuple()
  end

  defp max_abs(tensor) do
    tensor |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
  end

  defp to_host(value), do: Nx.backend_copy(value, Nx.BinaryBackend)

  defp env_boolean(name, default) do
    case System.get_env(name) do
      nil -> default
      value when value in ["1", "true", "TRUE"] -> true
      value when value in ["0", "false", "FALSE"] -> false
      value -> flunk("#{name} must be true/false or 1/0, got #{inspect(value)}")
    end
  end
end
