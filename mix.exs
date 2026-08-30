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
    # A sibling Nx checkout carrying `EXLA.CustomCall.Spec.mlir_attributes`,
    # which is unreleased. Points at the branch it lives on until it lands in
    # a release, at which point this becomes an ordinary version requirement.
    worktree = Path.expand("../nx-operation-attributes-pr", __DIR__)

    unless File.dir?(Path.join(worktree, "nx")) and File.dir?(Path.join(worktree, "exla")) do
      Mix.raise("expected an Nx worktree with nx/ and exla/ at #{worktree}")
    end

    [
      {:nx, path: Path.join(worktree, "nx"), override: true},
      {:exla, path: Path.join(worktree, "exla")}
    ]
  end
end
