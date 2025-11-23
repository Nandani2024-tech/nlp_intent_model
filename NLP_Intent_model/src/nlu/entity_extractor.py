"""
Hybrid entity extraction for banking NLU.

- Quick reliable extractors (regex-based) for amounts, simple dates, UPI, last-4 digits.
- Token-classifier scaffold (HuggingFace) for fine-grained entities (PERSON, ACCOUNT, IFSC, etc.)
  - train_token_classifier() is provided but optional; it requires `datasets` and `transformers`.
- predict_entities(text) returns a merged list of entities with type, span, and value.
- combined_nlu(text) merges intent prediction (from intent_classifier.predict_intent)
  with entity extraction and returns a single NLU JSON.

Place at: src/nlu/entity_extractor.py
"""
import re
import os
import json
from typing import List, Dict, Tuple

# --- quick regex extractors (reliable for amounts/dates/upi/last4) ---
# Amounts like:
# - 1000 rupees, 500 rs, 250 inr
# - ₹5,000 or ₹ 5,000.50 or ₹5000
# - 5000, 1200.75 (standalone numbers)
# - 5k, 5K, 10k (short forms)
AMOUNT_PATTERN = re.compile(
    r"(?P<amt>"  # named group for clarity
    r"\d+(?:\.\d{1,2})?\s?(?:rupees|rs|inr)"  # e.g. 1000 rupees, 500 rs
    r"|(?:\u20B9|Rs)\s?\d{4,}(?:\.\d{1,2})?"  # e.g. ₹5000, Rs5000, ₹10000 (4+ digits after currency) - check longer first
    r"|(?:\u20B9|Rs)\s?\d{1,3}(?:[,\s]\d{2,3})+(?:\.\d{1,2})?"  # e.g. ₹5,000, Rs 5,000 (with comma/space separators)
    r"|\d{1,3}(?:[,\s]\d{2,3}){1,}(?:\.\d{1,2})?"  # e.g. 5,000 or 10,000.50 (must have comma/space)
    r"|\d+(?:\.\d{1,2})?\s*[kK](?:\s|$|[^\w])"  # e.g. 5k, 5K, 10k
    r"|\b\d{3,}(?:\.\d{1,2})?(?=\s|$|[^\d.,\w])"  # standalone numbers (3+ digits): 500, 1000, 5000
    r")",
    flags=re.IGNORECASE,
)
ACCOUNT_TYPE_PATTERN = re.compile(r"\b(savings?|current)\b", flags=re.IGNORECASE)
UPI_PATTERN = re.compile(r"\b[A-Za-z0-9.\-_]{2,256}@[A-Za-z]{2,}\b")
LAST4_PATTERN = re.compile(r"\b(?:\d{4})\b")
DATE_SIMPLE_PATTERN = re.compile(
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}(?=[^\d]|$)"  # Dates like 12/03/2023
    r"|\b\d{1,2}\s(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s?\d{2,4}(?=[^\d]|$)"  # 15 December
    r"|\b(?:today|tomorrow|yesterday)\b",  # Words for relative dates
    flags=re.IGNORECASE,
)



def _normalize_currency_glitches(text: str) -> str:
    """
    Fix common mojibake sequences that appear when ₹ is captured
    in Windows terminals configured with cp1252 (e.g., 'â‚¹').
    """
    if not text:
        return text
    replacements = {
        "\u00E2\u201A\u00B9": "\u20B9",  # â‚¹
        "\u00E2\u0082\u00B9": "\u20B9",  # â¹ (different interpretation)
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text



def _finditer_to_entities(matches, ent_type):
    ents = []
    for m in matches:
        span = (m.start(), m.end())
        text = m.group(0)
        ents.append({"entity": ent_type, "value": text, "start": span[0], "end": span[1]})
    return ents

def extract_amounts_regex(text):
    """
    Extract amount entities from text.
    Handles cases like:
    - 1000 rupees, 500 rs
    - ₹5000, ₹5,000
    - 5k, 5K, 10k
    - 500, 1000, 5000 (standalone numbers)
    """
    text = _normalize_currency_glitches(text or "")
    results = []
    matches = list(AMOUNT_PATTERN.finditer(text))

    i = 0
    while i < len(matches):
        m = matches[i]
        raw = m.group().strip()
        start = m.start()
        end = m.end()

        # Look ahead to merge split amounts like "100" + "0 rupees"
        if i + 1 < len(matches):
            nxt = matches[i + 1]
            raw_next = nxt.group()

            # If current chunk has no currency word but next does,
            # and they are adjacent / near each other, merge them.
            if (not re.search(r"rupees|rs|inr|[kK]", raw, flags=re.IGNORECASE)
                and re.search(r"rupees|rs|inr", raw_next, flags=re.IGNORECASE)
                and nxt.start() <= end + 1):
                start = start
                end = nxt.end()
                raw = text[start:end].strip()
                i += 1  # consume the next match as well

        # Normalize the amount
        normalized = normalize_amount(raw)
        
        results.append({
            "entity": "amount",
            "value": raw,
            "normalized": normalized,
            "start": start,
            "end": end
        })
        i += 1

    return results


def normalize_amount(raw_amount: str) -> str:
    """
    Normalize amount string to numeric value.
    Handles:
    - "5k" or "5K" → "5000"
    - "₹5000" → "5000"
    - "₹5,000" → "5000"
    - "500 rupees" → "500"
    - "500" → "500"
    """
    if not raw_amount:
        return ""
    
    # Convert to lowercase for easier processing
    text = _normalize_currency_glitches(raw_amount).lower().strip()
    
    # Check for 'k' or 'K' suffix (e.g., "5k" → 5000, "10k" → 10000)
    # This handles cases like "5k", "5K", "10.5k"
    if 'k' in text:
        # Extract number before 'k'
        k_match = re.search(r'(\d+(?:\.\d+)?)', text)
        if k_match:
            base_num = float(k_match.group(1))
            # Convert to integer (no decimals for k notation)
            normalized = str(int(base_num * 1000))
            return normalized
    
    # Remove currency symbols and words
    text = re.sub(r'[\u20B9$€£]', '', text)
    text = re.sub(r'\b(?:Rs|rupees|rs|inr)\b\s*', '', text, flags=re.IGNORECASE)
    
    # Remove commas and spaces from numbers (e.g., "5,000" → "5000")
    text = re.sub(r'[,\s]', '', text)
    
    # Extract only digits and decimal point
    normalized = re.sub(r'[^\d.]', '', text)
    
    # Handle empty result
    if not normalized:
        return ""
    
    # Handle cases where we might have multiple dots (shouldn't happen but safety check)
    if normalized.count('.') > 1:
        # Keep only the first decimal point
        parts = normalized.split('.')
        normalized = parts[0] + '.' + ''.join(parts[1:])
    
    # Remove trailing decimal point if no fractional part
    if normalized.endswith('.'):
        normalized = normalized[:-1]
    
    return normalized


def extract_dates_regex(text):
    results = []
    for m in DATE_SIMPLE_PATTERN.finditer(text):
        raw = m.group()
        value = raw.rstrip(".,)")
        results.append({
            "entity": "date",
            "value": value,
            "start": m.start(),
            "end": m.end(),   # original end
        })
    return results




def extract_upi_regex(text):
    results = []
    for m in UPI_PATTERN.finditer(text):
        results.append({
            "entity": "upi_id",
            "value": m.group(),
            "start": m.start(),
            "end": m.end()
        })
    return results


LOAN_ID_PATTERN = re.compile(r"\b[a-zA-Z]{2,10}\d{2,10}\b")  # Example pattern, tweak for your formats

def extract_loan_id_regex(text):
    results = []
    for m in LOAN_ID_PATTERN.finditer(text):
        results.append({
            "entity": "loan_id",
            "value": m.group(),
            "start": m.start(),
            "end": m.end()
        })
    return results



def extract_last4_regex(text):
    results = []
    for m in LAST4_PATTERN.finditer(text):
        results.append({
            "entity": "last4",
            "value": m.group(),
            "start": m.start(),
            "end": m.end()
        })
    return results


def extract_account_type_regex(text):
    results = []
    for m in ACCOUNT_TYPE_PATTERN.finditer(text or ""):
        raw = m.group()
        normalized = "savings" if raw.lower().startswith("sav") else "current"
        results.append({
            "entity": "account_type",
            "value": raw,
            "normalized": normalized,
            "start": m.start(),
            "end": m.end()
        })
    return results


# -------------------------
# Optional: token-classifier scaffolding (HuggingFace)
# -------------------------
# This part expects token-level labeled data (BIO format). It's provided as a ready scaffold.
# To run token-classifier training you will need `datasets` and `transformers`.
try:
    from transformers import (
        AutoTokenizer,
        AutoModelForTokenClassification,
        TrainingArguments,
        Trainer,
    )
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

class TokenClassifierWrapper:
    def __init__(self, model_name="distilbert-base-uncased", model_dir=None, label_list: List[str]=None):
        """
        label_list example: ["O", "B-PER", "I-PER", "B-ACCOUNT", "I-ACCOUNT", ...]
        model_dir: if provided will load model from disk
        """
        self.model_name = model_name
        self.model_dir = model_dir
        self.label_list = label_list
        self.tokenizer = None
        self.model = None
        if model_dir:
            self.load_model(model_dir)

    def load_model(self, model_dir: str):
        if not HF_AVAILABLE:
            raise RuntimeError("Transformers not available for token-classifier loading.")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
        # try to capture labels if provided
        # label mapping may be in config or files saved in model_dir

    def predict(self, text: str) -> List[Dict]:
        """Return token-level entities predicted by token-classifier (span merging included)."""
        if not HF_AVAILABLE or self.model is None or self.tokenizer is None:
            return []
        inputs = self.tokenizer(text, return_offsets_mapping=True, return_tensors="pt", truncation=True)
        offset_map = inputs.pop("offset_mapping")[0].tolist()
        device = "cuda" if (hasattr(self.model, "device") and str(self.model.device).startswith("cuda")) else "cpu"
        self.model.to(device)
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with __import__("torch").no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits.squeeze(0)
            preds = logits.argmax(-1).cpu().numpy().tolist()
        # try to get label mapping
        id2label = None
        try:
            id2label = self.model.config.id2label
        except Exception:
            id2label = {i: lab for i, lab in enumerate(self.label_list)} if self.label_list else None

        # convert token preds to spans
        entities = []
        current = None
        for idx, p in enumerate(preds):
            label = id2label.get(p, "O") if id2label else "O"
            start, end = offset_map[idx]
            if start == end:  # special tokens
                continue
            if label == "O":
                if current is not None:
                    entities.append(current)
                    current = None
            else:
                # label like B-PER or I-ACCOUNT
                if label.startswith("B-"):
                    if current is not None:
                        entities.append(current)
                    current = {"entity": label[2:], "start": start, "end": end, "value": text[start:end]}
                elif label.startswith("I-"):
                    if current is None:
                        current = {"entity": label[2:], "start": start, "end": end, "value": text[start:end]}
                    else:
                        current["end"] = end
                        current["value"] = text[current["start"]:end]
                else:
                    # fallback
                    if current is not None:
                        entities.append(current)
                    current = None
        if current is not None:
            entities.append(current)
        return entities

# -------------------------
# Combined prediction util
# -------------------------
# Try to import intent predictor
try:
    # prefer package import if project root is in PYTHONPATH
    from src.nlu.intent_classifier import predict_intent, load_trained_model  # type: ignore
except Exception:
    try:
        # fallback to direct import if file is in same folder (run tests accordingly)
        from intent_classifier import predict_intent, load_trained_model  # type: ignore
    except Exception:
        predict_intent = None
        load_trained_model = None

# Single shared token-classifier instance (if you load one)
_SHARED_TOKEN_CLASSIFIER = None

# Single shared intent classifier instance (model, tokenizer, id2label)
_SHARED_INTENT_MODEL = None
_SHARED_INTENT_TOKENIZER = None
_SHARED_INTENT_ID2LABEL = None

def load_intent_model_if_available(model_dir: str = "models/intent_classifier"):
    """Load and cache the intent classification model."""
    global _SHARED_INTENT_MODEL, _SHARED_INTENT_TOKENIZER, _SHARED_INTENT_ID2LABEL
    if _SHARED_INTENT_MODEL is not None:
        return _SHARED_INTENT_MODEL, _SHARED_INTENT_TOKENIZER, _SHARED_INTENT_ID2LABEL
    
    if load_trained_model is None:
        raise RuntimeError("Intent classifier module not available.")
    
    try:
        model, tokenizer, _, id2label = load_trained_model(model_dir)
        _SHARED_INTENT_MODEL = model
        _SHARED_INTENT_TOKENIZER = tokenizer
        _SHARED_INTENT_ID2LABEL = id2label
        return model, tokenizer, id2label
    except Exception as e:
        raise RuntimeError(f"Failed to load intent model from {model_dir}: {str(e)}")

def load_token_classifier_if_available(model_dir: str):
    global _SHARED_TOKEN_CLASSIFIER
    if not HF_AVAILABLE:
        raise RuntimeError("Transformers library required to load token-classifier.")
    tc = TokenClassifierWrapper(model_dir=model_dir)
    _SHARED_TOKEN_CLASSIFIER = tc
    return tc

def predict_entities(text: str, use_token_classifier: bool = False) -> List[Dict]:
    """
    Run hybrid extraction and return deduped list of entities with type, span, and value.
    Priority: dates > upi > amounts > last4 > token-classifier
    """
    text = text or ""
    ents = []
    
    # Extract dates first to protect them from other extractions
    date_ents = extract_dates_regex(text)
    date_spans = set()
    for de in date_ents:
        date_spans.update(range(de["start"], de["end"]))
    ents.extend(date_ents)
    
    # Extract UPI (shouldn't conflict with dates)
    ents.extend(extract_upi_regex(text))
    
    # Extract amounts, but exclude those that overlap with dates
    amount_ents = extract_amounts_regex(text)
    for ae in amount_ents:
        amount_span = set(range(ae["start"], ae["end"]))
        if not (amount_span & date_spans):
            ents.append(ae)
    
    # Extract last4, but exclude those that overlap with dates
    last4_ents = extract_last4_regex(text)
    for le in last4_ents:
        last4_span = set(range(le["start"], le["end"]))
        if not (last4_span & date_spans):
            ents.append(le)
    

    # Extract simple account types
    ents.extend(extract_account_type_regex(text))

    # Extract simple account types
    ents.extend(extract_account_type_regex(text))

    # Extract loan IDs -- ADD THIS RIGHT AFTER ACCOUNT TYPE EXTRACTION
    ents.extend(extract_loan_id_regex(text))

    if use_token_classifier and _SHARED_TOKEN_CLASSIFIER is not None:
        try:
            tc_ents = _SHARED_TOKEN_CLASSIFIER.predict(text)
            # token-classifier returns spans in character offsets; just extend
            ents.extend(tc_ents)
        except Exception:
            pass

    # deduplicate overlapping entities with priority: date > upi > amount > last4 > token-classifier
    # Priority order for entity types
    priority_order = {"date": 0, "upi_id": 1, "amount": 2, "last4": 3, "account_type": 4}
    
    ents_sorted = sorted(ents, key=lambda e: (
        e["start"],
        priority_order.get(e["entity"], 99),  # Lower priority number = higher priority
        -(e["end"] - e["start"])  # Longer entities first for same type
    ))
    final = []
    occupied = set()
    for e in ents_sorted:
        rng = set(range(e["start"], e["end"]))
        if rng & occupied:
            continue
        final.append(e)
        occupied |= rng
    return final

def _rule_based_intents(text: str) -> List[Dict]:
    """Simple heuristic fallback when ML intent model isn't available."""
    lowered = (text or "").lower()
    lowered = lowered.strip()
    if not lowered:
        return []
    
    def intent_entry(name: str, confidence: float) -> Dict:
        return {"intent": name, "confidence": confidence}
    
    rule_table = [
        ("money_transfer", ["transfer", "send", "pay", "upi", "to "], 0.92),
        ("balance_inquiry", ["balance", "account balance", "check balance", "how much money"], 0.85),
        ("loan_query", ["loan", "emi", "interest rate"], 0.8),
        ("set_reminder", ["remind", "reminder", "remind me", "set a reminder"], 0.78),
    ]
    
    for intent_name, keywords, conf in rule_table:
        if any(keyword in lowered for keyword in keywords):
            return [intent_entry(intent_name, conf)]
    
    # Continuation heuristics: plain numbers or UPI-like strings
    if re.search(r"\b\d+(?:k)?\b", lowered):
        return [intent_entry("money_transfer", 0.65)]
    if "@" in lowered:
        return [intent_entry("money_transfer", 0.65)]
    
    return []


def combined_nlu(text: str, top_k_intents: int = 1, use_token_classifier: bool = False) -> Dict:
    """
    Returns:
    {
        "text": text,
        "intents": [{"intent": <str>, "confidence": <float>}],
        "entities": [{"entity": <type>, "value": <str>, "start": <int>, "end": <int>}]
    }
    """
    nlu = {"text": text, "intents": [], "entities": []}
    intents_payload: List[Dict] = []
    # intent
    if predict_intent is not None:
        try:
            # Try to use cached model first
            if (
                _SHARED_INTENT_MODEL is not None
                and _SHARED_INTENT_TOKENIZER is not None
                and _SHARED_INTENT_ID2LABEL is not None
            ):
                intents = predict_intent(
                    text,
                    model=_SHARED_INTENT_MODEL,
                    tokenizer=_SHARED_INTENT_TOKENIZER,
                    id2label=_SHARED_INTENT_ID2LABEL,
                    top_k=top_k_intents,
                )
            else:
                # Try to load model if not cached
                try:
                    model, tokenizer, id2label = load_intent_model_if_available()
                    intents = predict_intent(
                        text,
                        model=model,
                        tokenizer=tokenizer,
                        id2label=id2label,
                        top_k=top_k_intents,
                    )
                except Exception as load_err:
                    import logging

                    logger = logging.getLogger(__name__)
                    logger.warning(f"Failed to load cached intent model: {load_err}. Using fallback.")
                    intents = predict_intent(text, top_k=top_k_intents)
            
            intents_payload = [{"intent": lab, "confidence": float(score)} for lab, score in intents]
        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.error(f"Intent prediction failed: {str(e)}", exc_info=True)
            intents_payload = []
    else:
        intents_payload = []

    if not intents_payload:
        intents_payload = _rule_based_intents(text)
    nlu["intents"] = intents_payload
    # entities
    ents = predict_entities(text, use_token_classifier=use_token_classifier)
    nlu["entities"] = ents
    return nlu

# -------------------------
# Optional: Trainer helper (brief scaffold)
# -------------------------
def train_token_classifier(dataset, model_name="distilbert-base-uncased", output_dir="models/token_classifier", epochs=3, batch_size=8, learning_rate=5e-5):
    """
    Minimal scaffold that expects a `datasets.Dataset` object with features:
      - tokens: List[str]
      - ner_tags: List[int]  (label ids)
    This function intentionally keeps the logic small; for detailed tokenization/label alignment see HuggingFace token classification examples.
    """
    if not HF_AVAILABLE:
        raise RuntimeError("Transformers not available. Install transformers and datasets.")
    from datasets import Dataset, ClassLabel, Sequence
    # NOTE: This function is an example; real usage requires careful token->label alignment
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    unique_labels = dataset.features["ner_tags"].feature.names if hasattr(dataset.features["ner_tags"].feature, "names") else None
    # Build model
    num_labels = len(unique_labels) if unique_labels else max(max(x) for x in dataset["ner_tags"]) + 1
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels)

    # Placeholder: user must align tokens -> subtokens before training
    # It's safer to use HF token-classification example: https://huggingface.co/docs/transformers/tasks/token_classification

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_strategy="epoch",
        save_strategy="epoch",
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset, eval_dataset=None)
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir

# -------------
# End of file
# -------------
