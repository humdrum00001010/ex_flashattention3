defmodule FlashAttention3.Block do
  @moduledoc false

  @doc """
  Refuses to lower on a client that cannot run the kernel.

  Skipping here would compile `FlashAttention3.DenseAttention` instead, silently
  replacing the kernel with a score-matrix attention and changing the memory
  complexity of the caller.
  """
  def no_kernel!(client) do
    raise ArgumentError,
          "FlashAttention-3 requires a CUDA client, got: " <>
            "#{inspect(Map.get(client, :platform))}. This library binds the Hopper " <>
            "kernel and has no portable path; use an Nx attention implementation " <>
            "on other platforms."
  end
end

defmodule FlashAttention3.Block.Forward do
  @moduledoc false
  defstruct causal: false, softmax_scale: nil
end

defmodule FlashAttention3.Block.Backward do
  @moduledoc false
  defstruct causal: false, softmax_scale: nil
end

defimpl EXLA.CustomCall, for: FlashAttention3.Block.Forward do
  def call(%{causal: causal, softmax_scale: scale}, _out, operands, %{platform: :cuda}),
    do: {:ok, FlashAttention3.Kernel.forward(operands, causal, scale)}

  def call(_block, _out, _operands, client), do: FlashAttention3.Block.no_kernel!(client)
end

defimpl EXLA.CustomCall, for: FlashAttention3.Block.Backward do
  def call(%{causal: causal, softmax_scale: scale}, _out, operands, %{platform: :cuda}),
    do: {:ok, FlashAttention3.Kernel.backward(operands, causal, scale)}

  def call(_block, _out, _operands, client), do: FlashAttention3.Block.no_kernel!(client)
end
