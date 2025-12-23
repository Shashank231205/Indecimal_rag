import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rag.prompt import build_prompt

# -----------------------------
# Constants
# -----------------------------
STOPWORDS = {
    "what","how","does","do","is","are","the","a","an","and",
    "or","to","of","during","when","that","which","use"
}

PACKAGE_TERMS = {"essential", "premier", "infinia", "pinnacle"}

INTENT_MAP = {
    "delay": "delay",
    "delays": "delay",
    "escrow": "escrow",
    "payment": "payment",
    "payments": "payment",
    "maintenance": "maintenance",
    "wallet": "wallet",
    "price": "price",
    "steel": "steel",
    "quality": "quality",
    "checkpoint": "quality",
    "transparency": "transparency",
    "visibility": "visibility",
}

NUMERIC_PATTERN = re.compile(
    r"(₹\s?[\d,]+|\d{1,3},\d{3}\s*/\s*\w+|\d+\s*/\s*\w+)",
    re.IGNORECASE
)

# -----------------------------
# Fact Extraction
# -----------------------------
def extract_facts(text: str):
    facts = []

    for line in text.splitlines():
        line = line.strip()

        # bullet points
        if line.startswith(("-", "•")):
            cleaned = line.lstrip("-• ").strip()
            if len(cleaned.split()) >= 4:
                facts.append(cleaned)

        # fallback factual sentences
        elif (
            len(line.split()) >= 6
            and not line.startswith("#")
            and not line.lower().startswith("purpose")
        ):
            facts.append(line)

    # deduplicate while preserving order
    return list(dict.fromkeys(facts))


# -----------------------------
# Generator
# -----------------------------
class Generator:
    """
    FINAL grounded generator.
    - Retrieval via embeddings
    - Python enforces grounding
    - LLM only paraphrases (never invents)
    """

    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.eval()

    def generate(self, context, question: str) -> str:
        if not context:
            return "This information is not available in the provided documents."

        q_lower = question.lower()
        q_words = {w for w in q_lower.split() if w not in STOPWORDS}

        # -----------------------------
        # Collect all facts
        # -----------------------------
        facts = []
        for c in context:
            facts.extend(extract_facts(c["content"]))

        if not facts:
            return "This information is not available in the provided documents."

        # -----------------------------
        # Package isolation
        # -----------------------------
        pkg = next((p for p in PACKAGE_TERMS if p in q_lower), None)
        if pkg:
            facts = [f for f in facts if pkg in f.lower()]
            if not facts:
                return "This information is not available in the provided documents."

        # -----------------------------
        # Intent-based routing (CRITICAL FIX)
        # -----------------------------
        intent = None
        for k, v in INTENT_MAP.items():
            if k in q_lower:
                intent = v
                break

        if intent:
            intent_facts = [f for f in facts if intent in f.lower()]
            if intent_facts:
                facts = intent_facts

        # -----------------------------
        # Keyword safety filter (soft)
        # -----------------------------
        keyword_facts = [
            f for f in facts
            if q_words & set(f.lower().split())
        ]
        if keyword_facts:
            facts = keyword_facts

        if not facts:
            return "This information is not available in the provided documents."

        # -----------------------------
        # Numeric questions → deterministic
        # -----------------------------
        for f in facts:
            m = NUMERIC_PATTERN.search(f)
            if m:
                return m.group(0)

        # -----------------------------
        # Short factual → no LLM
        # -----------------------------
        if len(facts) <= 3:
            return " ".join(facts)

        # -----------------------------
        # LLM phrasing (SAFE MODE)
        # -----------------------------
        prompt = build_prompt(
            [{"source": "document", "content": f} for f in facts[:3]],
            question
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=False,
                no_repeat_ngram_size=3
            )

        answer = self.tokenizer.decode(
            output[0],
            skip_special_tokens=True
        ).strip()

        if not answer or "not available" in answer.lower():
            return "This information is not available in the provided documents."

        return answer
