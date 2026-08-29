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

  def call(%{causal: causal, softmax_scale: softmax_scale}, _out, [q, k, v], %{platform: :cuda}),
    do: {:ok, Lowering.forward(q, k, v, causal, softmax_scale)}

  def call(_block, _out, _args, _client), do: :skip
end

defimpl EXLA.CustomCall, for: FlashAttention3.Block.Backward do
  alias FlashAttention3.Lowering

  def call(
        %{causal: causal, softmax_scale: softmax_scale},
        _out,
        [q, k, v, _output, _lse, _doutput],
        %{platform: :cuda}
      ),
      do: {:ok, Lowering.backward(q, k, v, causal, softmax_scale)}

  def call(_block, _out, _args, _client), do: :skip
end
