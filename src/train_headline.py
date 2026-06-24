import os
import wandb

from datasets import load_from_disk

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)

MODEL_NAME = "VietAI/vit5-base"

# =====================================================
# CHỈ CẦN SỬA PHẦN NÀY MỖI LẦN TRAIN
# =====================================================

RUN_NAME = "run_04_batch4_epoch5"

LEARNING_RATE = 5e-5
BATCH_SIZE = 4
EPOCHS = 5

# =====================================================


def main():

    print("=" * 60)
    print("ĐANG LOAD TOKENIZER...")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("=" * 60)
    print("ĐANG LOAD MODEL...")
    print("=" * 60)

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    print("=" * 60)
    print("ĐANG LOAD DATASET...")
    print("=" * 60)

    dataset = load_from_disk("data/tokenized/train_dataset")

    print(dataset)

    # Train thử
    # dataset = dataset.select(range(500))

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model
    )

    # ===============================
    # Thư mục lưu model của run hiện tại
    # ===============================

    SAVE_DIR = os.path.join("models", RUN_NAME)

    training_args = Seq2SeqTrainingArguments(

        output_dir=SAVE_DIR,

        overwrite_output_dir=True,

        num_train_epochs=EPOCHS,

        per_device_train_batch_size=BATCH_SIZE,

        gradient_accumulation_steps=2,

        learning_rate=LEARNING_RATE,

        weight_decay=0.01,

        warmup_steps=500,

        logging_steps=20,

        save_strategy="epoch",

        save_total_limit=2,

        predict_with_generate=False,

        fp16=True,

        torch_compile=False,

        report_to="wandb",

        dataloader_num_workers=0,
    )

    wandb.init(

        project="keyword-headline-generator",

        name=RUN_NAME,

        config={
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
        }

    )

    trainer = Seq2SeqTrainer(

        model=model,

        args=training_args,

        train_dataset=dataset,

        tokenizer=tokenizer,

        data_collator=data_collator,

    )

    print("=" * 60)
    print("BẮT ĐẦU TRAIN")
    print("=" * 60)

    trainer.train()

    print("=" * 60)
    print("ĐANG LƯU MODEL...")
    print("=" * 60)

    trainer.save_model(SAVE_DIR)

    tokenizer.save_pretrained(SAVE_DIR)

    print(f"\nModel đã lưu tại: {SAVE_DIR}")

    wandb.finish()

    print("=" * 60)
    print("HOÀN THÀNH!")
    print("=" * 60)


if __name__ == "__main__":
    main()