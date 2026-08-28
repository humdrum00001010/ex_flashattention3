defmodule FA3TP.Benchmark do
  @moduledoc false

  import Nx.Defn

  defn forward_chain(q, k, v, opts \\ []) do
    opts =
      keyword!(opts,
        chain_length: 64,
        causal: true,
        call_target_name: "exla_fa3_forward",
        backward_call_target_name: "exla_fa3_backward",
        platforms: [:cuda]
      )

    {output, _k, _v} =
      while {output = q, k, v}, _i <- 1..opts[:chain_length], unroll: true do
        {output, _lse} =
          forward(output, k, v,
            causal: opts[:causal],
            call_target_name: opts[:call_target_name],
            backward_call_target_name: opts[:backward_call_target_name],
            platforms: opts[:platforms]
          )

        {output, k, v}
      end

    output
  end

  defn forward_backward_chain(q, k, v, doutput, opts \\ []) do
    opts =
      keyword!(opts,
        chain_length: 64,
        causal: true,
        call_target_name: "exla_fa3_forward",
        backward_call_target_name: "exla_fa3_backward",
        platforms: [:cuda]
      )

    {q, k, v, _doutput} =
      while {q, k, v, doutput}, _i <- 1..opts[:chain_length], unroll: true do
        {dq, dk, dv} =
          grad({q, k, v}, fn {q, k, v} ->
            {output, _lse} =
              forward(q, k, v,
                causal: opts[:causal],
                call_target_name: opts[:call_target_name],
                backward_call_target_name: opts[:backward_call_target_name],
                platforms: opts[:platforms]
              )

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
