jsonnet_train:
	jsonnet \
  --ext-str train_status="TRAIN_OFF" \
  ./contracts/train_config.jsonnet > train_config.json

jsonnet:
	jsonnet ./contracts/train_config.jsonnet > train_config.json

torch_cpu:
	pip install torch --index-url https://download.pytorch.org/whl/cpu

drop_workflows:
	rm .github/workflows/workflow_*_pipeline.yaml
