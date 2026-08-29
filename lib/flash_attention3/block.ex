defmodule FlashAttention3.Block.Forward do
  @moduledoc false
  defstruct causal: false, softmax_scale: nil
end

defmodule FlashAttention3.Block.Backward do
  @moduledoc false
  defstruct causal: false, softmax_scale: nil
end

defimpl EXLA.CustomCall, for: FlashAttention3.Block.Forward do
  alias FlashAttention3.Lowering

  def call(%{causal: causal, softmax_scale: scale}, _out, operands, %{platform: :cuda}),
    do: {:ok, Lowering.forward(operands, causal, scale)}

  def call(_block, _out, _operands, _client), do: :skip
end

defimpl EXLA.CustomCall, for: FlashAttention3.Block.Backward do
  alias FlashAttention3.Lowering

  def call(%{causal: causal, softmax_scale: scale}, _out, operands, %{platform: :cuda}),
    do: {:ok, Lowering.backward(operands, causal, scale)}

  def call(_block, _out, _operands, _client), do: :skip
end
