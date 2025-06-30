# Hate Speech Detection with HateBERT

A project fine-tuning **HateBERT**—a BERT variant pretrained on abusive and hate-speech data—for robust hate speech and offensive language detection.

---

## 📊 Dataset

**Hate Speech and Offensive Language Dataset**  
- **Total tweets:** 24,783  
- **Classes:**  
  - **0 – Hate Speech:** 1,430 examples  
  - **1 – Offensive Language:** 19,190 examples  
  - **2 – Neither:** remainder  

Each record contains:  
- `tweet` – raw text of the Twitter post  
- `class` – numeric label (0/1/2)  

---

## 1. Data Preparation

1. **Cleaning & Filtering**  
   - Drop missing or duplicate tweets  
   - Lowercase, remove URLs, mentions (`@…`), hashtags (`#…`), punctuation & special characters  

2. **Label Mapping**  
   - Standardize into three classes:  
     `0 = hate_speech`, `1 = offensive_language`, `2 = neither`

3. **Data Augmentation**  
   - Synonym-based augmentation for underrepresented hate speech  
   - New counts after augmentation:  
     - Hate Speech → 8,321  
     - Offensive Language → 19,190  
     - Neither → 1,430  
---

## 2. Model Selection & Tokenization

- **Encoder:**  
  - [`HateBERT`][hb] (RoBERTa-base fine-tuned on hate speech data)  
  - Loaded via Hugging Face `transformers`  

- **Tokenizer:**  
  - `AutoTokenizer.from_pretrained("HateBERT")`  
  - Settings:  
    - `max_length=128`  
    - `padding='max_length'`, `truncation=True`  

- **Dataset & DataLoader:**  
  1. Wrap cleaned tweets + labels in a custom `torch.utils.data.Dataset`  
  2. Tokenize on-the-fly in `__getitem__()`  
  3. Use `DataLoader(..., batch_size=32, shuffle=True)` for train/val/test  

---

## 3. Training Strategy

1. **Data Splitting**  
   - Train / Val / Test:  
     - **Train:** 20 073 examples  
     - **Val:**   2 231 examples  
     - **Test:**  2 479 examples  
   - Stratified split to preserve class ratios.

2. **Loss & Class Weights**  
   - Weighted cross-entropy to counter class imbalance  
   - Class weights inversely proportional to class frequencies.

3. **Optimizer & Scheduler**  
   - AdamW (`lr=2e-5`, `weight_decay=0.01`)  
   - Linear warm-up over first 10% of steps, then linear decay.

4. **Training Loop**  
   ```python
   for epoch in range(EPOCHS):
       train_epoch(model, train_loader)
       val_metrics = eval_model(model, val_loader)
       if val_metrics["f1"] > best_f1:
           save_checkpoint(model)
- Early stopping on validation F1 plateau (patience=3)

- **Metrics Calculation**  
  - Accuracy, Precision, Recall, F1‐score  
  - Report both **macro** and **weighted** averages  
  - Confusion matrix visualized via `sklearn.metrics.ConfusionMatrixDisplay`
 
## 4. Results

- **Baseline (no augmentation)**  
  - Accuracy: 0.84  
  - Precision (macro / weighted): 0.65 / 0.87  
  - Recall    (macro / weighted): 0.71 / 0.84  
  - F1-score  (macro / weighted): 0.67 / 0.85  
- **With Synonym-Based Augmentation**  
  - Accuracy: 0.90  
  - Precision (macro / weighted): 0.90 / 0.90  
  - Recall    (macro / weighted): 0.90 / 0.90  
  - F1-score  (macro / weighted): 0.90 / 0.90  
- **Visualizations**  
  - Confusion matrices before / after augmentation  
  - Comparison bar-chart of Precision / Recall / F1  

