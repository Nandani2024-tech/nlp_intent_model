# import re
# import unicodedata

# def normalize_text(text: str) -> str:
#     text = text.lower()
#     text = unicodedata.normalize("NFKD", text)
#     text = re.sub(r"[^a-zA-Z0-9\s₹]", "", text)
#     return text.strip()

# def tokenize(text: str):
#     return text.split()

# def preprocess(text: str):
#     text = normalize_text(text)
#     tokens = tokenize(text)
#     return {
#         "clean_text": text,
#         "tokens": tokens
#     }




import re

def normalize_text(text):
    text = text.lower()
    text = text.strip()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text

def tokenize(text):
    return text.split()

def preprocess_text(text):
    text = normalize_text(text)
    tokens = tokenize(text)
    return {
        "original": text,
        "tokens": tokens
    }

if __name__ == "__main__":
    sample = "Transfer 500 rupees to Rahul!"
    print(preprocess_text(sample))
