defmodule FlashAttention3 do
  @moduledoc """
  FlashAttention-3 for Nx, as an XLA custom call.

  `attention/4` is the operation. It takes BSHD `{batch, sequence, heads, dim}`
  tensors, is differentiable through `Nx.Defn.grad/2`, and is callable from
  `defn`.

  This is a binding to the Hopper kernel, not an attention library. Compiled
  with EXLA it lowers to the native kernel on a CUDA client with BF16 or FP16
  and a head dimension of 128 or 256, and raises on any other client rather
  than substituting a score-matrix attention that would change the memory
  complexity of the model that called it.

  That refusal lives in `FlashAttention3.Block`, which only EXLA consults.
  Called eagerly or under `Nx.Defn.Evaluator` there is no such gate, and
  `FlashAttention3.DenseAttention` runs instead. It is correct and it is what
  the tests compare against, but it materializes the score matrix, so treat
  those paths as small-shape only.

  Loading the native library is the application's job. The FFI targets and the
  custom-call partitioner are registered by `libfa3_xla.so`'s static
  initializers, so the application loads it into the OS process once, before
  compiling anything that contains an FA3 call.

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
    softmax_scale = Keyword.get(opts, :softmax_scale) || 1.0 / :math.sqrt(Nx.axis_size(q, 3))

    q
    |> FFI.forward(k, v, causal, softmax_scale)
    |> attach_grad(q, k, v, causal, softmax_scale)
  end

  # Attaches the FA3 backward as the gradient of the output. `custom_grad/3`
  # and `stop_grad/1` annotate expressions, so an eager call has nothing to
  # annotate and passes through; its gradient, if taken at all, comes from
  # differentiating `DenseAttention` directly.
  defp attach_grad({output, lse}, %{data: %Nx.Defn.Expr{}} = q, k, v, causal, softmax_scale) do
    graded =
      custom_grad(output, [q, k, v], fn doutput ->
        # Nx differentiates through an f32 scalar loss, so the cotangent
        # arriving here is f32 even when FA3's output is bf16/f16. The upstream
        # FA3 backward ABI requires dO to have the same element type as
        # Q/K/V/O.
        doutput = Nx.as_type(doutput, q.type)

        {dq, dk, dv} = FFI.backward(q, k, v, output, lse, doutput, causal, softmax_scale)
        [dq, dk, dv]
      end)

    {graded, stop_grad(lse)}
  end

  defp attach_grad(result, _q, _k, _v, _causal, _softmax_scale), do: result
end
