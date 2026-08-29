defmodule FlashAttention3.HostTestBlock.Forward do
  @moduledoc """
  Mirrors `FlashAttention3.Block.Forward`, but lowers on the host client.

  The CPU preflight has no CUDA device, so the production block always skips
  and the fallback is compiled instead. This block runs the same
  `FlashAttention3.Lowering` code on `:host`, which is what lets the preflight
  assert the emitted StableHLO without a GPU.
  """
  defstruct causal: false, softmax_scale: nil
end

defmodule FlashAttention3.HostTestBlock.Backward do
  @moduledoc false
  defstruct causal: false, softmax_scale: nil
end

defimpl EXLA.CustomCall, for: FlashAttention3.HostTestBlock.Forward do
  alias FlashAttention3.Lowering

  def call(%{causal: causal, softmax_scale: scale}, _out, [q, k, v], %{platform: :host}),
    do: {:ok, Lowering.forward(q, k, v, causal, scale)}

  def call(_block, _out, _args, _client), do: :skip
end

defimpl EXLA.CustomCall, for: FlashAttention3.HostTestBlock.Backward do
  alias FlashAttention3.Lowering

  def call(
        %{causal: causal, softmax_scale: scale},
        _out,
        [q, k, v, _output, _lse, _doutput],
        %{platform: :host}
      ),
      do: {:ok, Lowering.backward(q, k, v, causal, scale)}

  def call(_block, _out, _args, _client), do: :skip
end
