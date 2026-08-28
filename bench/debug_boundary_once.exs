path =
  System.get_env("FA3_TP_DYLIB") ||
    raise "set FA3_TP_DYLIB to libfa3_xla.so"

:ok = EXLA.NIF.load_dylib(path)

chain_length =
  System.get_env("FA3_TP_CHAIN_LENGTH", "8")
  |> String.to_integer()

runs =
  System.get_env("FA3_TP_DEBUG_RUNS", "2")
  |> String.to_integer()

q = Nx.broadcast(Nx.tensor(0.01, type: {:bf, 16}), {1, 256, 24, 256})
k = Nx.broadcast(Nx.tensor(0.02, type: {:bf, 16}), {1, 256, 4, 256})
v = Nx.broadcast(Nx.tensor(0.03, type: {:bf, 16}), {1, 256, 4, 256})

opts = [
  chain_length: chain_length,
  causal: true,
  call_target_name: "exla_fa3_forward",
  platforms: [:cuda]
]

forward = fn q, k, v -> FA3TP.Benchmark.forward_chain(q, k, v, opts) end
compiled = EXLA.jit(forward, client: :cuda)

for _ <- 1..runs do
  compiled.(q, k, v) |> Nx.backend_deallocate()
end

IO.puts("FA3_DEBUG_ONCE_DONE chain_length=#{chain_length} runs=#{runs}")
