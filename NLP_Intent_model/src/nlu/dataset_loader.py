import json
import yaml

def load_intents(path="data/intents/intents.json"):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)

def load_entities(path="data/entities/entities.yaml"):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def load_multilingual(path="data/multilingual/multilingual_samples.json"):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)

if __name__ == "__main__":
    print("Intents:", load_intents())
    print("Entities:", load_entities())
    print("Multilingual Samples:", load_multilingual())
