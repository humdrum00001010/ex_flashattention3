defmodule FlashAttention3.Benchmark do
  @moduledoc false

  import Nx.Defn

  defn forward_chain(q, k, v, opts \\ []) do
    opts = keyword!(opts, chain_length: 64, causal: true)

    {output, _k, _v} =
      while {output = q, k, v}, _i <- 1..opts[:chain_length], unroll: true do
        {FlashAttention3.attention(output, k, v, causal: opts[:causal]), k, v}
      end

    output
  end

  defn forward_backward_chain(q, k, v, doutput, opts \\ []) do
    opts = keyword!(opts, chain_length: 64, causal: true)

    {q, k, v, _doutput} =
      while {q, k, v, doutput}, _i <- 1..opts[:chain_length], unroll: true do
        {dq, dk, dv} =
          grad({q, k, v}, fn {q, k, v} ->
            q
            |> FlashAttention3.attention(k, v, causal: opts[:causal])
            |> Nx.multiply(doutput)
            |> Nx.sum()
          end)

        {dq, dk, dv, doutput}
      end

    {q, k, v}
  end
end
