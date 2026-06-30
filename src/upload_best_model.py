import os
import wandb

run = wandb.init(
    project="keyword-headline-generator",
    name="upload_best_model"
)

artifact = wandb.Artifact(
    name="best_model",
    type="model",
    description="Best ViT5 model"
)

MODEL_DIR = "models/best_model"

ALLOWED_FILES = [
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer_config.json",
    "training_args.bin",
]

for file in ALLOWED_FILES:
    path = os.path.join(MODEL_DIR, file)

    if os.path.exists(path):
        artifact.add_file(path)

run.log_artifact(artifact)

run.finish()

print("Best model upload thành công!")