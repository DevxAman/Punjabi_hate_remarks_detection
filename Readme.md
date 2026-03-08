# 🛡️ Punjabi Hate Remarks Detection

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=for-the-badge&logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=for-the-badge&logo=huggingface)
![Accuracy](https://img.shields.io/badge/Accuracy-97.60%25-brightgreen?style=for-the-badge)
![F1 Score](https://img.shields.io/badge/F1%20Score-0.976-brightgreen?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge)

**A transformer-based hate speech detection system for Punjabi (Gurmukhi script) text, fine-tuned on Google MuRIL — achieving 97.60% accuracy on a custom-curated dataset of 1,384 real social media posts.**

[🤗 Live Demo](https://devxaman-punjabi-hate-detector.hf.space) • [📦 Model on HuggingFace](https://huggingface.co/DevxAman/punjabi-hate-speech-muril) • [🌐 Web App](https://punjabi-guard.vercel.app)

</div>

---

## 📌 Overview

Punjabi is spoken by over **100 million people worldwide**, yet automated hate speech detection tools for the language are virtually non-existent. This project fills that gap.

We fine-tuned **Google's MuRIL** (Multilingual Representations for Indian Languages) model on a custom-built Punjabi hate speech dataset scraped from real social media platforms. The result is a production-grade classifier that:

- ✅ Detects hate speech in **Gurmukhi script** with 97.6% accuracy
- ✅ Handles **code-mixed** Punjabi-English text
- ✅ Rejects unsupported scripts (Hindi, Arabic, Urdu) gracefully
- ✅ Deployed as a **live REST API** on HuggingFace Spaces
- ✅ Accessible via a **web frontend** at punjabi-guard.vercel.app

---

## 🚀 Live Demo

> Try it instantly — no setup required

🔗 **Web App:** [punjabi-guard.vercel.app](https://punjabi-guard.vercel.app)  
🔗 **API Endpoint:** [devxaman-punjabi-hate-detector.hf.space](https://devxaman-punjabi-hate-detector.hf.space)  
🔗 **Model Weights:** [huggingface.co/DevxAman/punjabi-hate-speech-muril](https://huggingface.co/DevxAman/punjabi-hate-speech-muril)

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | **97.60%** |
| **F1 Score (Weighted)** | **0.9759** |
| **F1 Score (Macro)** | **0.9757** |
| **Precision (Weighted)** | **97.63%** |
| **Recall (Weighted)** | **97.60%** |
| **Test Set Size** | 208 samples |
| **Misclassifications** | 5 out of 208 |

### Confusion Matrix (Test Set — n=208)

|  | Predicted: Non-Hate | Predicted: Hate |
|--|--|--|
| **Actual: Non-Hate** | 112 ✅ | 1 ❌ |
| **Actual: Hate** | 4 ❌ | 91 ✅ |

### Training Progress

| Epoch | Train Loss | Val Loss | Val Accuracy | Val F1 |
|-------|-----------|----------|-------------|--------|
| 1 | 0.5234 | 0.2461 | 90.87% | 0.9082 |
| 2 | 0.2187 | 0.1638 | 93.75% | 0.9370 |
| 3 | 0.1043 | 0.1184 | 95.19% | 0.9514 |
| **4** | **0.0578** | **0.0953** | **97.60%** | **0.9759** |

---

## 🗂️ Dataset

A custom Punjabi hate speech dataset was compiled from scratch — **no comparable public dataset exists for Punjabi.**

| Split | Total | Hate | Non-Hate |
|-------|-------|------|----------|
| Train | 968 | 441 | 527 |
| Validation | 208 | 95 | 113 |
| Test | 208 | 95 | 113 |
| **Total** | **1,384** | **631** | **753** |

- **Sources:** Facebook public pages, YouTube comments, Twitter/X posts
- **Script:** Gurmukhi (Punjabi)
- **Balance Ratio:** 0.84 (no oversampling required)
- **Annotation:** Manual labeling by native Punjabi speakers

---

## 🧠 Model Architecture

| Parameter | Value |
|-----------|-------|
| Base Model | `google/muril-base-cased` |
| Architecture | BERT (12 layers, 12 attention heads) |
| Vocabulary Size | 197,258 tokens |
| Trainable Parameters | 237,557,762 |
| Max Sequence Length | 128 tokens |
| Training Epochs | 4 |
| Learning Rate | 2e-5 |
| Batch Size | 16 |
| Optimizer | AdamW |
| Warmup Steps | 100 |
| Weight Decay | 0.01 |
| Training Hardware | Google Colab T4 GPU |
| Training Time | ~45 minutes |
| Framework | PyTorch + HuggingFace Transformers 4.40 |

**Why MuRIL over mBERT?**  
MuRIL was specifically pre-trained on 17 Indian languages including Punjabi (Gurmukhi script), giving it far superior tokenization and semantic understanding of Gurmukhi text compared to standard multilingual BERT.

---

## 🔌 API Usage

### Health Check
```bash
GET https://devxaman-punjabi-hate-detector.hf.space/health
```

### Predict
```bash
POST https://devxaman-punjabi-hate-detector.hf.space/predict
Content-Type: application/json

{
  "text": "ਪੰਜਾਬ ਦੇ ਲੋਕ ਮਿਹਨਤੀ ਨੇ"
}
```

### Response
```json
{
  "label": "NON-HATE",
  "confidence": 0.729,
  "probabilities": {
    "NON-HATE": 0.729,
    "HATE": 0.271
  }
}
```

### Error Codes
| Code | Meaning |
|------|---------|
| `200` | Successful prediction |
| `400` | Empty or missing input text |
| `422` | Unsupported language script (Hindi, Arabic, Urdu) |

---

## 🛠️ Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/DevxAman/Punjabi_hate_remarks_detection.git
cd Punjabi_hate_remarks_detection

# 2. Install dependencies
pip install torch transformers flask

# 3. Download model from HuggingFace
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained("DevxAman/punjabi-hate-speech-muril")
model = AutoModelForSequenceClassification.from_pretrained("DevxAman/punjabi-hate-speech-muril")

# 4. Run inference
inputs = tokenizer("ਪੰਜਾਬ ਦੇ ਲੋਕ ਮਿਹਨਤੀ ਨੇ", return_tensors="pt", truncation=True, max_length=128)
outputs = model(**inputs)
```

---

## 🏗️ System Architecture

```
User Input (Punjabi Text)
        │
        ▼
┌───────────────────┐
│  Language Filter  │  ← Rejects Hindi, Arabic, Urdu (HTTP 422)
│  (Unicode Check)  │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│   Preprocessing   │  ← Normalize, clean, remove noise
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ MuRIL Tokenizer   │  ← WordPiece, max_len=128, attention mask
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  MuRIL Classifier │  ← Fine-tuned, 237M params
└────────┬──────────┘
         │
         ▼
   HATE / NON-HATE
   + Confidence Score
```

---

## 🚢 Deployment

| Component | Technology | URL |
|-----------|-----------|-----|
| Model Weights | HuggingFace Hub | [DevxAman/punjabi-hate-speech-muril](https://huggingface.co/DevxAman/punjabi-hate-speech-muril) |
| Backend API | Flask + Docker (HF Spaces) | [devxaman-punjabi-hate-detector.hf.space](https://devxaman-punjabi-hate-detector.hf.space) |
| Frontend UI | Vanilla HTML/CSS/JS (Vercel) | [punjabi-guard.vercel.app](https://punjabi-guard.vercel.app) |

---

## 👥 Team

**BTech Final Year Project — Computer Science & Engineering**  
**Guru Nanak Dev Engineering College (GNDEC), Ludhiana**  
**March 2026**

| Name | URN |
|------|-----|
| Amandeep Singh | 2203396 |
| Arshpreet Kaur | 2203407 |
| Natasha Pal | 2203508 |

**Project Guide:** Dr. Parminder Singh  
**HOD:** Dr. Kiran Jyoti

---

## 📄 License

This project is licensed under the MIT License.
