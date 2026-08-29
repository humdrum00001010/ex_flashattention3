defmodule FlashAttention3 do
  @moduledoc """
  FlashAttention-3 for Nx, lowered to an XLA custom call on CUDA.

  `attention/4` is the operation. It takes BSHD `{batch, sequence, heads, dim}`
  tensors, is differentiable through `Nx.Defn.grad/2`, and is callable from
  `defn`. On a CUDA client with a supported dtype and head dimension it lowers
  to the native FA3 kernel; anywhere else it runs
  `FlashAttention3.Reference`, which defines the same operation but
  materializes the full score matrix.

  The native library must be loaded once before use, see
  `FlashAttention3.Native.load!/0`.

  ## Options

    * `:causal` - applies a causal mask. Requires equal q/k sequence lengths.
      Defaults to `false`.
    * `:softmax_scale` - defaults to `1/sqrt(head_dim)`.
  """

  import Nx.Defn
  import Nx.Defn.Kernel, only: [custom_grad: 3, stop_grad: 1]

  alias FlashAttention3.FFI

  @doc """
  Scaled dot-product attention over BSHD tensors.
  """
  deftransform attention(q, k, v, opts \\ []) do
    {output, _lse} = attention_with_lse(q, k, v, opts)
    output
  end

  @doc """
  Attention that also returns the FP32 log-sum-exp normalizer.

  The LSE is laid out as `{batch, heads, sequence}`. It is not differentiable;
  it is returned for callers that merge partial attention results across key
  and value chunks, such as ring or context-parallel attention.
  """
  deftransform attention_with_lse(q, k, v, opts \\ []) do
    opts = Keyword.validate!(opts, causal: false, softmax_scale: nil)
    causal = Keyword.fetch!(opts, :causal)
    softmax_scale = Keyword.get(opts, :softmax_scale) || default_scale(q)

    q
    |> FFI.forward(k, v, causal, softmax_scale)
    |> differentiate(q, k, v, causal, softmax_scale)
  end

  defp default_scale(q) do
    {_batch, _sequence, _heads, head_dim} = FlashAttention3.Shape.rank4!(q, "q")
    1.0 / :math.sqrt(head_dim)
  end

  defp differentiate({output, lse}, %{data: %Nx.Defn.Expr{}} = q, k, v, causal, softmax_scale) do
    graded =
      custom_grad(output, [q, k, v], fn doutput ->
        # Nx differentiates through an f32 scalar loss, so the cotangent
        # arriving here is f32 even when FA3's output is bf16/f16. The upstream
        # FA3 backward ABI requires dO to have the same element type as
        # Q/K/V/O.
        doutput = Nx.as_type(doutput, q.type)

        {dq, dk, dv} =
          FFI.backward(q, k, v, output, lse, doutput, causal, softmax_scale)

        [dq, dk, dv]
      end)

    {graded, stop_grad(lse)}
  end

  defp differentiate(result, _q, _k, _v, _causal, _softmax_scale), do: result
end
