"""FinBERT scoring - the main Talk measure.

Two models by Huang, Wang & Yang (2023, Contemporary Accounting Research):
  * yiyanghkust/finbert-esg  : Environmental / Social / Governance / None
  * yiyanghkust/finbert-tone : Positive / Neutral / Negative

A sentence counts as positive environmental talk if FinBERT-ESG labels it
Environmental and FinBERT-tone labels it Positive.
"""
from __future__ import annotations

from typing import Callable, Sequence

ESG_MODEL = "yiyanghkust/finbert-esg"
TONE_MODEL = "yiyanghkust/finbert-tone"

# A classifier maps a list of sentences to a list of lower-case labels.
Classifier = Callable[[Sequence[str]], list[str]]


def _hf_classifier(model_name: str, device: int | None, batch_size: int) -> Classifier:
    import torch
    from transformers import pipeline

    if device is None:
        device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline(
        "text-classification",
        model=model_name,
        tokenizer=model_name,
        device=device,
        truncation=True,
        max_length=512,
    )

    def classify(sentences: Sequence[str]) -> list[str]:
        if not sentences:
            return []
        out = pipe(list(sentences), batch_size=batch_size)
        return [o["label"].lower() for o in out]

    return classify


class FinBertScorer:
    def __init__(
        self,
        esg: Classifier | None = None,
        tone: Classifier | None = None,
        device: int | None = None,
        batch_size: int = 64,
    ):
        self.esg = esg or _hf_classifier(ESG_MODEL, device, batch_size)
        self.tone = tone or _hf_classifier(TONE_MODEL, device, batch_size)

    def label(self, sentences: Sequence[str]) -> list[tuple[bool, str | None]]:
        """For each sentence: (is_environmental, tone or None)."""
        esg = self.esg(sentences)
        env_idx = [i for i, lab in enumerate(esg) if lab == "environmental"]
        tones = self.tone([sentences[i] for i in env_idx])
        result: list[tuple[bool, str | None]] = [(False, None)] * len(sentences)
        for i, t in zip(env_idx, tones):
            result[i] = (True, t)
        return result
