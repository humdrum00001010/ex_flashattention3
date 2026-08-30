defmodule ExFlashAttention3.MixProject do
  use Mix.Project

  def project do
    [
      app: :ex_flashattention3,
      version: "0.1.0",
      elixir: "~> 1.20.2",
      elixirc_paths: elixirc_paths(Mix.env()),
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  def application do
    [extra_applications: [:logger]]
  end

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_env), do: ["lib"]

  defp deps do
    # `EXLA.CustomCall.Spec.mlir_attributes` is unreleased, so nx and exla come
    # from the branch carrying it. Both live in one repository, so each names
    # its `subdir`; without that Mix reads the repository root, which is itself
    # called nx. `override` is needed because exla declares nx from Hex. Pinned
    # by commit so a force-push cannot change what this builds against.
    [
      {:nx,
       github: "humdrum00001010/nx",
       ref: "62ecad529e24ae534eb4082e67153b0b3e12db73",
       subdir: "nx",
       override: true},
      {:exla,
       github: "humdrum00001010/nx",
       ref: "62ecad529e24ae534eb4082e67153b0b3e12db73",
       subdir: "exla"}
    ]
  end
end
