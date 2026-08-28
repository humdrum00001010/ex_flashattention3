defmodule FlashAttention3.HostTestBlock do
  @moduledoc false
  @enforce_keys [:spec]
  defstruct [:spec]
end

defimpl EXLA.CustomCall, for: FlashAttention3.HostTestBlock do
  def call(%{spec: spec}, _outputs, _inputs, %{platform: :host}), do: {:ok, spec}
  def call(_block, _outputs, _inputs, _client), do: :skip
end
