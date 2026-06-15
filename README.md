# Fine-tuned Small Models vs Large Zero-shot Models for Financial NLP

> **Research Question:** For financial sentiment classification, does a fine-tuned small model (110M params) outperform a large zero-shot model (7B params)?

## TL;DR — Key Finding
Fine-tuned FinBERT (110M) achieves **F1 0.8793**, outperforming Mistral-7B zero-shot (F1 0.6862) by **19.3 F1 points** at **64x less compute**.

## Results

| Model | Params | Strategy | Weighted F1 | Accuracy |
|-------|--------|----------|-------------|----------|
| FinBERT | 110M | Zero-shot | 0.7187 | 71% |
| Mistral-7B | 7B | Zero-shot | 0.6862 | 68% |
| Mistral-7B | 7B | Few-shot (3) | 0.7422 | 74% |
| **FinBERT** | **110M** | **Fine-tuned** | **0.8793** | **88%** |

## Three Research Findings

**Finding 1:** Fine-tuned small model beats zero-shot large model by 19.3 F1 points at 64x less compute.

**Finding 2:** Few-shot prompting improves Mistral-7B by 5.6 F1 points but still trails fine-tuned by 13.7 points.

**Finding 3:** Domain fine-tuning provides 22.4% relative improvement over zero-shot on same architecture.

## Practical Implication
- Classification tasks → Fine-tune small models (cheaper + more accurate)
- Generative QA tasks → Large models necessary
- Few-shot → Viable middle ground with zero training cost

## Dataset
- Twitter Financial News Sentiment (9,543 samples)
- Source: `zeroshot/twitter-financial-news-sentiment`

## Models
- `ProsusAI/finbert` (110M) — zero-shot + fine-tuned
- `mistralai/Mistral-7B-Instruct-v0.3` (7B) — zero-shot + few-shot

## Stack
Python · HuggingFace Transformers · PyTorch · scikit-learn · Matplotlib
