# Finsight
### Automated Bank Transaction Categorization Using Fine-Tuned DistilBERT

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co)

> FinSight transforms raw bank transaction strings like `AMZN MKTP US*2K4LP` 
> into meaningful spending categories and financial insights.

---

## Live Demo
🚀 **[Launch FinSight App](#)** 

---

## What it does
Upload a CSV of bank transactions and FinSight will:
- Categorize every transaction into 7 spending categories
- Generate a spending breakdown with percentages
- Calculate a Financial Health Score based on the 50/30/20 rule
- Flag unusual transactions as anomalies

**Categories:** Food & Dining · Transport · Shopping · Entertainment · Healthcare · Utilities · Financial Services

---

## Model Performance

| Model | Weighted F1 | Accuracy |
|---|---|---|
| Logistic Regression + TF-IDF | 0.5152 | 51% |
| Random Forest + TF-IDF | 0.5085 | 50% |
| **DistilBERT fine-tuned** | **0.6565** | **66%** |

---

## Architecture
