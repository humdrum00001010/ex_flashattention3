defmodule FlashAttention3.Native do
  @moduledoc """
  Loads the external library that registers the FA3 XLA FFI handlers.

  Emitting the custom call is only half of the contract: the target names must
  be registered with XLA in this OS process before a computation containing
  them executes. Registration happens in the library's static initializers, so
  loading it is the whole job.

  Call this once at application boot. It raises immediately on a missing or
  unloadable library, rather than letting the failure surface later as an
  unregistered custom-call target from inside XLA.
  """

  @doc """
  Loads the FA3 shared library.

  The path comes from `config :fa3_tp_experiment, :dylib` or the `FA3_TP_DYLIB`
  environment variable.
  """
  def load!(path \\ nil) do
    path =
      path || Application.get_env(:fa3_tp_experiment, :dylib) ||
        System.get_env("FA3_TP_DYLIB") ||
        raise ArgumentError,
              "set config :fa3_tp_experiment, :dylib or FA3_TP_DYLIB to libfa3_xla.so"

    unless File.regular?(path) do
      raise ArgumentError, "FA3 library is not a regular file: #{path}"
    end

    :ok = EXLA.NIF.load_dylib(path)
  end
end
