{
  model_registries: [
    {
      registry_name: "stock_models_" + std.toString(_idx),
      registry_url:  "https://huggingface.co/qywok/stock_models_" + std.toString(_idx)
    } for _idx in std.range(1, 10)
  ]
}
