defmodule FA3TPExperiment.MixProject do
  use Mix.Project

  def project do
    [
      app: :fa3_tp_experiment,
      version: "0.1.0",
      elixir: "~> 1.17",
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
    worktree =
      System.get_env("NX_MLIR_ATTRIBUTES_WORKTREE") ||
        Path.expand("../../nx-upstream-main", __DIR__)

    unless File.dir?(Path.join(worktree, "nx")) and File.dir?(Path.join(worktree, "exla")) do
      Mix.raise("""
      NX_MLIR_ATTRIBUTES_WORKTREE must point to the Nx worktree containing
      EXLA.CustomCall.Spec.mlir_attributes, got: #{worktree}
      """)
    end

    [
      {:nx, path: Path.join(worktree, "nx"), override: true},
      {:exla, path: Path.join(worktree, "exla")}
    ]
  end
end
