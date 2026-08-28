defmodule FA3TP.Benchmark do
  @moduledoc false

  import Nx.Defn

  defn forward_chain(q, k, v, opts \\ []) do
    opts =
      keyword!(opts,
        chain_length: 64,
        causal: true
      )

    {output, _k, _v} =
      while {output = q, k, v}, _i <- 1..opts[:chain_length], unroll: true do
        {output, _lse} =
          forward(output, k, v, causal: opts[:causal])

        {output, k, v}
      end

    output
  end

  defn forward_backward_chain(q, k, v, doutput, opts \\ []) do
    opts =
      keyword!(opts,
        chain_length: 64,
        causal: true
      )

    {q, k, v, _doutput} =
      while {q, k, v, doutput}, _i <- 1..opts[:chain_length], unroll: true do
        {dq, dk, dv} =
          grad({q, k, v}, fn {q, k, v} ->
            {output, _lse} =
              forward(q, k, v, causal: opts[:causal])

            output |> Nx.multiply(doutput) |> Nx.sum()
          end)

        {dq, dk, dv, doutput}
      end

    {q, k, v}
  end

  deftransformp forward(q, k, v, opts) do
    FA3TP.forward(q, k, v, opts)
  end
end
