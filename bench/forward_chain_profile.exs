path =
  System.get_env("FA3_TP_DYLIB") ||
    raise "set FA3_TP_DYLIB to libfa3_xla.so"

:ok = EXLA.NIF.load_dylib(path)

chain_length = System.get_env("FA3_TP_CHAIN_LENGTH", "64") |> String.to_integer()
runs = System.get_env("FA3_TP_PROFILE_RUNS", "5") |> String.to_integer()

q = Nx.broadcast(Nx.tensor(0.01, type: {:bf, 16}), {4, 2048, 24, 256})
k = Nx.broadcast(Nx.tensor(0.02, type: {:bf, 16}), {4, 2048, 4, 256})
v = Nx.broadcast(Nx.tensor(0.03, type: {:bf, 16}), {4, 2048, 4, 256})

opts = [
  chain_length: chain_length,
  causal: true
]

forward = fn q, k, v -> FA3TP.Benchmark.forward_chain(q, k, v, opts) end
mesh = %Nx.Mesh{name: "fa3_profile", shape: {2}}
shardings = [%{2 => [0]}, %{2 => [0]}, %{2 => [0]}]
host_shards = FA3TP.shard_inputs(q, k, v, 2)

stage =
  EXLA.shard_jit(fn q, k, v -> {q, k, v} end, mesh,
    client: :cuda,
    input_shardings: shardings
  )

args = stage.(host_shards) |> Enum.map(&Tuple.to_list/1)
compiled = EXLA.shard_jit(forward, mesh, client: :cuda, input_shardings: shardings)

deallocate = fn results -> Enum.each(results, &Nx.backend_deallocate/1) end

compiled.(args) |> deallocate.()

for _ <- 1..runs do
  compiled.(args) |> deallocate.()
end

Enum.each(args, fn tensors -> Enum.each(tensors, &Nx.backend_deallocate/1) end)

IO.puts("FA3_PROFILE_DONE chain_length=#{chain_length} runs=#{runs}")
