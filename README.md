# 🎫 Support Ticket Classifier

> **Assessment Project** — Automatic routing of support tickets using Machine Learning  
> **Author:** Shivanshu Shukla | **Stack:** Python · scikit-learn · TF-IDF · Streamlit

---

## 📌 Problem Statement

Support teams receive hundreds of tickets daily across different departments — Billing, Technical, HR, and General. Manually triaging each ticket is slow and error-prone. This project builds an ML-powered classifier that:

- Reads the **subject + body** of a ticket
- Automatically assigns it to the correct category
- Returns a **confidence score**
- Flags low-confidence tickets for **human review**
- Tags tickets as **URGENT or NORMAL** using keyword rules

---

## 🗂️ Project Structure

```
Assessment/
├── data/
│   └── tickets.csv          # 50 labeled support tickets (4 categories)
├── ticket_classifier.py     # Main ML pipeline + CLI demo
├── app.py                   # Streamlit web demo
├── requirements.txt
└── README.md
```

---

## 🧠 Approach

### 1. Text Preprocessing
- Lowercase + remove punctuation/numbers
- Combine subject and body (both carry signal)
- Minimal stopword removal — TF-IDF handles frequency-based down-weighting

### 2. Feature Engineering — TF-IDF
**Why TF-IDF over simple Bag-of-Words?**
- **TF (Term Frequency):** rewards words that appear often in this specific ticket
- **IDF (Inverse Document Frequency):** penalises words common across all tickets
- Using **unigrams + bigrams** captures phrases like *"payment failed"*, *"not working"*
- `sublinear_tf=True` applies log dampening to avoid high-frequency word dominance

### 3. Models Trained
| Model | Why chosen |
|-------|-----------|
| **Naive Bayes** | Fast, works great on sparse TF-IDF, excellent baseline for text |
| **Logistic Regression** | Handles correlated features better, typically outperforms NB on small datasets |

Both models are trained and compared. The best one is selected automatically.

### 4. Evaluation
- **Accuracy** — overall correctness
- **Precision / Recall / F1** — per-category breakdown
- **Confusion matrix** — visualises where the model gets confused

---

## ✅ Bonus Features Implemented

| Feature | Description |
|---------|-------------|
| **Confidence score** | Returns probability % alongside predicted category |
| **Human review threshold** | Confidence < 60% → routed to manual review queue |
| **Priority tagging** | Keywords like *"urgent", "down", "not working"* → URGENT tag |
| **Streamlit demo** | Live input box, category probabilities, performance charts |
| **Reflection note** | See below |

---

## 🚀 How to Run

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the CLI classifier (with interactive demo)
```bash
python ticket_classifier.py
```

### Run the Streamlit web app
```bash
streamlit run app.py
```

---

## 📊 Results

The classifier achieves **~90–100% accuracy** on the held-out test set (10 samples from 50).  
Logistic Regression consistently outperforms Naive Bayes on this dataset due to its ability to model feature correlations better.

Sample predictions on new tickets:

| Ticket (truncated) | Predicted | Confidence | Priority |
|--------------------|-----------|------------|----------|
| "I was charged twice this month..." | Billing | 95%+ | URGENT |
| "App crashes on every startup..." | Technical | 90%+ | NORMAL |
| "Need to apply for paternity leave..." | HR | 88%+ | NORMAL |
| "Tell me about your enterprise pricing..." | General | 78%+ | NORMAL |
| "Webhook not firing after deployment — urgent" | Technical | 85%+ | URGENT |

---

## 💭 Reflection Note

> **What would I improve with more data or time?**
>
> 1. **More training data** — 50 tickets is tiny. With 500–5000 samples per category, accuracy would likely hit 95%+ and generalise better to edge cases.
> 2. **Better embeddings** — TF-IDF misses semantic meaning (e.g. "can't login" vs "authentication issue"). Sentence-BERT or a fine-tuned DistilBERT would capture this.
> 3. **Active learning loop** — tickets flagged for human review could be labelled and fed back to retrain the model automatically over time.
> 4. **Multi-label classification** — some tickets span categories (e.g. "My salary was deducted but I haven't received a receipt" → both HR and Billing).
> 5. **Better edge-case handling** — tickets with very vague text (e.g. "I need help") could be handled with a dedicated "Unknown" class and always routed to human review.

---

## 🏷️ Tags
`#billing` `#technical` `#hr` `#general` `#tfidf` `#naive-bayes` `#logistic-regression` `#nlp` `#python` `#scikit-learn`
