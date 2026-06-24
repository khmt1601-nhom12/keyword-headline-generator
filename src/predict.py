from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# ==================================
# Load model đã train
# ==================================

MODEL_PATH = "models/vit5"

print("=" * 60)
print("ĐANG LOAD MODEL...")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print("Device:", device)


# ==================================
# Hàm sinh tiêu đề
# ==================================

def generate_headline(keywords):

    inputs = tokenizer(
        keywords,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_length=64,
            num_beams=5,
            early_stopping=True,
            no_repeat_ngram_size=2
        )

    headline = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return headline


# ==================================
# Test
# ==================================

if __name__ == "__main__":

    while True:

        print("-" * 60)

        keywords = input("Nhập keywords (Enter để thoát): ")

        if keywords == "":
            break

        headline = generate_headline(keywords)

        print("\nHeadline:")
        print(headline)