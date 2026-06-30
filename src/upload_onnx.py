import wandb

run = wandb.init(
    project="keyword-headline-generator",
    name="upload_onnx"
)

artifact = wandb.Artifact(
    "onnx_model",
    type="model",
    description="Exported ONNX model"
)

artifact.add_file(
    "outputs/onnx/vit5_encoder.onnx"
)

artifact.add_file(
    "outputs/onnx/headline_generator_int8.onnx"
)

run.log_artifact(artifact)

run.finish()