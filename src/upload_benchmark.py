import wandb

run = wandb.init(
    project="keyword-headline-generator",
    name="upload_benchmark"
)

artifact = wandb.Artifact(
    "benchmark",
    type="dataset",
    description="Inference benchmark"
)

artifact.add_file("benchmark.csv")

run.log_artifact(artifact)

run.finish()

print("Upload benchmark thành công!")