"""
Small wrapper that exposes combined_nlu as an importable function
and a simple CLI for quick checks.
"""
import argparse
from src.nlu.entity_extractor import (
    combined_nlu,
    load_token_classifier_if_available,
    _SHARED_TOKEN_CLASSIFIER,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--use_tc", action="store_true", help="Use token-classifier if loaded")
    args = parser.parse_args()

    if args.use_tc and _SHARED_TOKEN_CLASSIFIER is None:
        # you may call load_token_classifier_if_available("models/token_classifier") beforehand
        try:
            load_token_classifier_if_available("models/token_classifier")
        except Exception as e:
            print("Could not load token classifier:", e)

    out = combined_nlu(args.text, use_token_classifier=args.use_tc)
    print(out)

if __name__ == "__main__":
    main()
