path =
  System.get_env("FA3_TP_DYLIB") ||
    raise "set FA3_TP_DYLIB to libfa3_xla.so"

:ok = EXLA.NIF.load_dylib(path)

chain_length =
  System.get_env("FA3_TP_CHAIN_LENGTH", "8")
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
stage = EXLA.jit(fn q, k, v -> {q, k, v} end, client: :cuda)
args = stage.(q, k, v) |> Tuple.to_list()
compiled = EXLA.jit(forward, client: :cuda)

apply(compiled, args) |> Nx.backend_deallocate()

IO.puts("FA3_DEBUG_READY os_pid=#{System.pid()} chain_length=#{chain_length}")
Process.sleep(60_000)

apply(compiled, args) |> Nx.backend_deallocate()
Enum.each(args, &Nx.backend_deallocate/1)

IO.puts("FA3_DEBUG_DONE")
