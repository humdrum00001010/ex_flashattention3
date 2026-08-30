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
