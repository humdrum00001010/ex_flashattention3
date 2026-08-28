defmodule FA3TP.IntegrationTest do
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
  # Add FA3_TP_BENCHMARK=1 only after the correctness and reentry gates pass.

  # External-library contract exercised by this test:
  #
  #   * CUDA FFI target: exla_fa3_forward
  #   * operands: Q/K/V bf16, logical [batch, sequence, heads, dim]
  #   * results: O bf16 [batch, q_sequence, q_heads, value_dim]
  #              LSE f32 [batch, q_heads, q_sequence]
  #   * backend_config attrs: causal: bool, softmax_scale: f32
  #   * the dylib also registers the target's CustomCallPartitioner

  test "single-device correctness -> TP correctness -> resident reentry -> benchmark" do
    client = EXLA.Client.fetch!(:cuda)
    assert client.device_count >= 2, "FA3 TP2 requires two physical CUDA devices"
    assert_hopper_topology!()

    load_external_fa3!()
    hlo_dir = fresh_hlo_dir!()

    target = "exla_fa3_forward"
    causal = env_boolean("FA3_TP_CAUSAL", true)
    mesh = %Nx.Mesh{name: "fa3_tp", shape: {2}}
    input_shardings = [%{2 => [0]}, %{2 => [0]}, %{2 => [0]}]

    {q_f32, k_f32, v_f32} = deterministic_inputs({1, 64, 8, 2, 128})
    q = Nx.as_type(q_f32, {:bf, 16})
    k = Nx.as_type(k_f32, {:bf, 16})
    v = Nx.as_type(v_f32, {:bf, 16})

    forward = fn q, k, v ->
      FA3TP.forward(q, k, v, causal: causal)
    end

    # Gate 1: the native target itself must match an independent FP32 Nx oracle.
    single = EXLA.jit(forward, client: :cuda).(q, k, v) |> to_host()
    assert_matches_oracle!(single, {q_f32, k_f32, v_f32}, {q, k, v}, causal)

    # Gate 2: TP receives complete KV groups, executes one local FA3 per rank,
    # and reconstructs the same global O/LSE without a collective inside FA3.
    host_shards = FA3TP.shard_inputs(q, k, v, 2)

    tp_forward =
      EXLA.shard_jit(forward, mesh,
        client: :cuda,
        input_shardings: input_shardings
      )

    sharded = tp_forward.(host_shards)
    assert Enum.map(sharded, fn {output, _lse} -> output.data.buffer.device_id end) == [0, 1]

    assembled = sharded |> Enum.map(&to_host/1) |> FA3TP.assemble_outputs()
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

    {expected_output, _expected_lse} = FA3TP.reference(q_f32, k_f32, v_f32, causal: causal)
    {baseline_output, _baseline_lse} = FA3TP.reference(q, k, v, causal: causal)

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

    resident_assembled = resident_result |> Enum.map(&to_host/1) |> FA3TP.assemble_outputs()
    assert_matches_oracle!(resident_assembled, {q_f32, k_f32, v_f32}, {q, k, v}, causal)

    if env_boolean("FA3_TP_BENCHMARK", false) do
      benchmark!(causal, mesh, input_shardings)
    end
  end

  defp load_external_fa3! do
    path =
      System.get_env("FA3_TP_DYLIB") ||
        flunk("set FA3_TP_DYLIB to the torch-free FA3 XLA FFI shared library")

    unless File.regular?(path) do
      flunk("FA3_TP_DYLIB does not name a regular file: #{path}")
    end

    assert :ok = EXLA.NIF.load_dylib(path)
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
        {output, _lse} = FA3TP.reference(q, k, v, causal: causal)
        output |> Nx.multiply(doutput) |> Nx.sum()
      end)
    end

    low_precision_gradient = fn q, k, v, doutput ->
      Nx.Defn.grad({q, k, v}, fn {q, k, v} ->
        {output, _lse} = FA3TP.reference(q, k, v, causal: causal, upcast: false)
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
        FA3TP.forward(q, k, v, causal: causal)
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

      qkv_shards = FA3TP.shard_inputs(q, k, v, 2)
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

  defp apply_reference({q, k, v}, causal), do: FA3TP.reference(q, k, v, causal: causal)

  defp map_tensors({left, right}, fun), do: {fun.(left), fun.(right)}

  defp map_tuple_tensors(tuple, fun) do
    tuple |> Tuple.to_list() |> Enum.map(fun) |> List.to_tuple()
  end

  defp max_abs(tensor) do
    tensor |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
  end

  defp to_host(value), do: Nx.backend_copy(value, Nx.BinaryBackend)

  defp benchmark!(causal, mesh, input_shardings) do
    batch = env_integer("FA3_TP_BATCH", 4)
    seqlen = env_integer("FA3_TP_SEQLEN", 2048)
    q_heads = env_integer("FA3_TP_Q_HEADS", 24)
    kv_heads = env_integer("FA3_TP_KV_HEADS", 4)
    dim = env_integer("FA3_TP_HEAD_DIM", 256)
    chain_length = env_integer("FA3_TP_CHAIN_LENGTH", 64)
    warmup = env_integer("FA3_TP_WARMUP", 3)
    iterations = env_integer("FA3_TP_ITERATIONS", 10)
    calls_per_sample = env_integer("FA3_TP_CALLS_PER_SAMPLE", 1)

    for {precision, type} <- benchmark_precisions() do
      args = [
        precision,
        type,
        {batch, seqlen, q_heads, kv_heads, dim},
        causal,
        mesh,
        input_shardings,
        chain_length,
        warmup,
        iterations,
        calls_per_sample
      ]

      benchmark =
        if env_boolean("FA3_TP_FORWARD_ONLY", false),
          do: &benchmark_forward_only!/10,
          else: &benchmark_precision!/10

      apply(benchmark, args)
    end
  end

  defp benchmark_forward_only!(
         precision,
         type,
         {batch, seqlen, q_heads, kv_heads, dim},
         causal,
         mesh,
         input_shardings,
         chain_length,
         warmup,
         iterations,
         calls_per_sample
       ) do
    q = Nx.broadcast(Nx.tensor(0.01, type: type), {batch, seqlen, q_heads, dim})
    k = Nx.broadcast(Nx.tensor(0.02, type: type), {batch, seqlen, kv_heads, dim})
    v = Nx.broadcast(Nx.tensor(0.03, type: type), {batch, seqlen, kv_heads, dim})

    chain_opts = [
      chain_length: chain_length,
      causal: causal
    ]

    forward_chain = fn q, k, v ->
      FA3TP.Benchmark.forward_chain(q, k, v, chain_opts)
    end

    single_args =
      EXLA.jit(fn q, k, v -> {q, k, v} end, client: :cuda).(q, k, v)
      |> Tuple.to_list()

    host_shards = FA3TP.shard_inputs(q, k, v, 2)

    tp_args =
      EXLA.shard_jit(fn q, k, v -> {q, k, v} end, mesh,
        client: :cuda,
        input_shardings: input_shardings
      ).(host_shards)
      |> Enum.map(&Tuple.to_list/1)

    single_forward = EXLA.jit(forward_chain, client: :cuda)

    tp_forward =
      EXLA.shard_jit(forward_chain, mesh,
        client: :cuda,
        input_shardings: input_shardings
      )

    single_stats =
      measure(single_forward, single_args, warmup, iterations, calls_per_sample, chain_length)

    tp_stats =
      measure(tp_forward, [tp_args], warmup, iterations, calls_per_sample, chain_length)

    forward_flops =
      4.0 * batch * q_heads * seqlen * seqlen * dim * if(causal, do: 0.5, else: 1.0)

    IO.puts("""
    FA3_TP_RESULT precision=#{precision} global_shape={#{batch},#{seqlen},#{q_heads},#{kv_heads},#{dim}} chain_length=#{chain_length} warmup=#{warmup} samples=#{iterations} calls_per_sample=#{calls_per_sample} forward_only=true
    single_forward_logical_pflops=#{forward_flops / single_stats.median / 1.0e9}
    tp2_forward_logical_pflops=#{forward_flops / tp_stats.median / 1.0e9}
    tp2_forward_throughput_scaling=#{single_stats.median / tp_stats.median}
    single_forward_median_us=#{single_stats.median} p10_us=#{single_stats.p10} p90_us=#{single_stats.p90}
    tp2_forward_median_us=#{tp_stats.median} p10_us=#{tp_stats.p10} p90_us=#{tp_stats.p90}
    """)

    deallocate_result(single_args)
    deallocate_result(tp_args)
  end

  defp benchmark_precision!(
         precision,
         type,
         {batch, seqlen, q_heads, kv_heads, dim},
         causal,
         mesh,
         input_shardings,
         chain_length,
         warmup,
         iterations,
         calls_per_sample
       ) do
    q = Nx.broadcast(Nx.tensor(0.01, type: type), {batch, seqlen, q_heads, dim})
    k = Nx.broadcast(Nx.tensor(0.02, type: type), {batch, seqlen, kv_heads, dim})
    v = Nx.broadcast(Nx.tensor(0.03, type: type), {batch, seqlen, kv_heads, dim})
    doutput = Nx.broadcast(Nx.tensor(0.04, type: type), q.shape)

    chain_opts = [
      chain_length: chain_length,
      causal: causal
    ]

    forward_chain = fn q, k, v ->
      FA3TP.Benchmark.forward_chain(q, k, v, chain_opts)
    end

    forward_backward_chain = fn q, k, v, doutput ->
      FA3TP.Benchmark.forward_backward_chain(q, k, v, doutput, chain_opts)
    end

    stage_single = EXLA.jit(fn q, k, v, doutput -> {q, k, v, doutput} end, client: :cuda)
    single_gradient_args = stage_single.(q, k, v, doutput) |> Tuple.to_list()
    single_forward_args = Enum.take(single_gradient_args, 3)
    single_forward = EXLA.jit(forward_chain, client: :cuda)
    single_forward_backward = EXLA.jit(forward_backward_chain, client: :cuda)

    qkv_shards = FA3TP.shard_inputs(q, k, v, 2)
    q_heads_per_rank = div(q_heads, 2)

    host_gradient_shards =
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

    stage_tp =
      EXLA.shard_jit(fn q, k, v, doutput -> {q, k, v, doutput} end, mesh,
        client: :cuda,
        input_shardings: input_shardings ++ [%{2 => [0]}]
      )

    tp_gradient_args = stage_tp.(host_gradient_shards) |> Enum.map(&Tuple.to_list/1)
    tp_forward_args = Enum.map(tp_gradient_args, &Enum.take(&1, 3))

    tp_forward =
      EXLA.shard_jit(forward_chain, mesh,
        client: :cuda,
        input_shardings: input_shardings
      )

    tp_forward_backward =
      EXLA.shard_jit(forward_backward_chain, mesh,
        client: :cuda,
        input_shardings: input_shardings ++ [%{2 => [0]}]
      )

    single_forward_stats =
      measure(
        single_forward,
        single_forward_args,
        warmup,
        iterations,
        calls_per_sample,
        chain_length
      )

    tp_forward_stats =
      measure(
        tp_forward,
        [tp_forward_args],
        warmup,
        iterations,
        calls_per_sample,
        chain_length
      )

    single_forward_backward_stats =
      measure(
        single_forward_backward,
        single_gradient_args,
        warmup,
        iterations,
        calls_per_sample,
        chain_length
      )

    tp_forward_backward_stats =
      measure(
        tp_forward_backward,
        [tp_gradient_args],
        warmup,
        iterations,
        calls_per_sample,
        chain_length
      )

    single_forward_us = single_forward_stats.median
    tp_forward_us = tp_forward_stats.median
    single_forward_backward_us = single_forward_backward_stats.median
    tp_forward_backward_us = tp_forward_backward_stats.median

    forward_flops =
      4.0 * batch * q_heads * seqlen * seqlen * dim * if(causal, do: 0.5, else: 1.0)

    IO.puts("""
    FA3_TP_RESULT precision=#{precision} global_shape={#{batch},#{seqlen},#{q_heads},#{kv_heads},#{dim}} chain_length=#{chain_length} warmup=#{warmup} samples=#{iterations} calls_per_sample=#{calls_per_sample}
    single_forward_logical_pflops=#{forward_flops / single_forward_us / 1.0e9}
    tp2_forward_logical_pflops=#{forward_flops / tp_forward_us / 1.0e9}
    tp2_forward_throughput_scaling=#{single_forward_us / tp_forward_us}
    single_forward_backward_logical_pflops=#{3.0 * forward_flops / single_forward_backward_us / 1.0e9}
    tp2_forward_backward_logical_pflops=#{3.0 * forward_flops / tp_forward_backward_us / 1.0e9}
    tp2_forward_backward_throughput_scaling=#{single_forward_backward_us / tp_forward_backward_us}
    single_forward_median_us=#{single_forward_us} p10_us=#{single_forward_stats.p10} p90_us=#{single_forward_stats.p90}
    tp2_forward_median_us=#{tp_forward_us} p10_us=#{tp_forward_stats.p10} p90_us=#{tp_forward_stats.p90}
    single_forward_backward_median_us=#{single_forward_backward_us} p10_us=#{single_forward_backward_stats.p10} p90_us=#{single_forward_backward_stats.p90}
    tp2_forward_backward_median_us=#{tp_forward_backward_us} p10_us=#{tp_forward_backward_stats.p10} p90_us=#{tp_forward_backward_stats.p90}
    """)

    deallocate_result(single_gradient_args)
    deallocate_result(tp_gradient_args)
  end

  defp measure(fun, args, warmup, iterations, calls_per_sample, work_per_call) do
    for _ <- 1..warmup do
      result = apply(fun, args)
      deallocate_result(result)
    end

    samples =
      for _ <- 1..iterations do
        total =
          for _ <- 1..calls_per_sample, reduce: 0 do
            elapsed_sum ->
              {elapsed, result} = :timer.tc(fn -> apply(fun, args) end)
              deallocate_result(result)
              elapsed_sum + elapsed
          end

        total / calls_per_sample / work_per_call
      end

    samples = Enum.sort(samples)

    %{
      median: percentile(samples, 0.50),
      p10: percentile(samples, 0.10),
      p90: percentile(samples, 0.90)
    }
  end

  defp percentile(samples, quantile) do
    Enum.at(samples, floor((length(samples) - 1) * quantile))
  end

  defp deallocate_result(results) when is_list(results) do
    Enum.each(results, &deallocate_result/1)
  end

  defp deallocate_result(result), do: Nx.backend_deallocate(result)

  defp env_boolean(name, default) do
    case System.get_env(name) do
      nil -> default
      value when value in ["1", "true", "TRUE"] -> true
      value when value in ["0", "false", "FALSE"] -> false
      value -> flunk("#{name} must be true/false or 1/0, got #{inspect(value)}")
    end
  end

  defp env_integer(name, default) do
    case System.get_env(name) do
      nil ->
        default

      value ->
        case Integer.parse(value) do
          {integer, ""} when integer > 0 -> integer
          _ -> flunk("#{name} must be a positive integer, got #{inspect(value)}")
        end
    end
  end

  defp benchmark_precisions do
    case System.get_env("FA3_TP_PRECISION", "both") do
      "both" -> [bf16: {:bf, 16}, fp16: {:f, 16}]
      "bf16" -> [bf16: {:bf, 16}]
      "fp16" -> [fp16: {:f, 16}]
      value -> flunk("FA3_TP_PRECISION must be bf16, fp16, or both, got #{inspect(value)}")
    end
  end
end
