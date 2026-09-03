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
    # The custom-call metadata and library loader are unreleased, so both
    # applications follow the official repository at the commit containing
    # them. Pinning one revision keeps Nx and EXLA on the same API boundary.
    nx_ref = "509da1b5e28380bb60bd11d67dc669e4b68231df"

    [
      {:nx, github: "elixir-nx/nx", ref: nx_ref, subdir: "nx", override: true},
      {:exla, github: "elixir-nx/nx", ref: nx_ref, subdir: "exla"}
    ]
  end
end
