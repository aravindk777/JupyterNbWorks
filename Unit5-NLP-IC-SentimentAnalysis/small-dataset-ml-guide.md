# Sentiment Analysis on 234 Reviews: Complete Guide

## Question 1: Which Models Apply for 234 Reviews?

### ✅ Models You CAN Use (With Caveats)

| Model Type | Can Use? | Expected Performance | Recommendation |
|------------|----------|---------------------|----------------|
| **Logistic Regression** | ✅ Yes | 60-75% accuracy | ✅ Use with cross-validation |
| **Naïve Bayes** | ✅ Yes | 60-70% accuracy | ✅ Works with small data |
| **LSTM** | ⚠️ Technically yes | 45-60% accuracy | ❌ Will severely overfit |
| **BERT/Transformers** | ⚠️ Fine-tuning only | 70-80% accuracy | ⚠️ Use pre-trained + fine-tune |
| **VADER** | ✅ Yes (no training) | 75-85% accuracy | ✅ **Best for this size** |

### Reality Check for 234 Reviews

```
Traditional ML (LogReg, NB):
├─ Pro: Can work with small datasets
├─ Pro: Fast training
├─ Con: Performance limited by data size
└─ Result: 60-75% accuracy (with good practices)

Deep Learning (LSTM):
├─ Pro: Powerful when enough data
├─ Con: Needs 10,000+ samples to work well
├─ Con: Will memorize 234 samples
└─ Result: 45-60% accuracy (severe overfitting)

Transformers (BERT):
├─ Pro: Pre-trained knowledge helps
├─ Pro: Can fine-tune on small data
├─ Con: Complex setup
└─ Result: 70-80% with fine-tuning (best DL option)

VADER (Rule-based):
├─ Pro: No training needed
├─ Pro: Pre-built on millions of samples
├─ Con: Not customizable
└─ Result: 75-85% accuracy (best for 234 samples)
```

---

## Question 2: Can You Use Traditional Models on 234 Reviews?

### ✅ YES - But with Proper Techniques

**You absolutely CAN use Logistic Regression and Naïve Bayes**, but you must:

1. **Use Cross-Validation** (not simple train/test split)
2. **Limit features** (avoid overfitting)
3. **Use regularization** (especially for LogReg)
4. **Report realistic metrics** (cross-validation scores)

### How to Do It Right

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Assume you have 234 reviews
# df has columns: 'review_text' and 'sentiment'

# Step 1: Encode labels
le = LabelEncoder()
y = le.fit_transform(df['sentiment'])

# Step 2: Vectorize with LIMITED features
vectorizer = TfidfVectorizer(
    max_features=100,      # SMALL for 234 samples!
    ngram_range=(1, 1),    # Only unigrams
    min_df=2,              # Must appear in 2+ docs
    max_df=0.8             # Ignore very common words
)

X = vectorizer.fit_transform(df['review_text'])

print(f"Feature matrix: {X.shape}")  # Should be (234, 100)

# Step 3: Logistic Regression with Cross-Validation
log_reg = LogisticRegression(
    C=0.1,                 # Strong regularization
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)

# Use 5-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_lr = cross_val_score(log_reg, X, y, cv=cv, scoring='accuracy')

print("\nLogistic Regression:")
print(f"  CV Scores: {cv_scores_lr}")
print(f"  Mean: {cv_scores_lr.mean():.4f} (+/- {cv_scores_lr.std():.4f})")

# Step 4: Naïve Bayes with Cross-Validation
nb = MultinomialNB(alpha=1.0)  # Laplace smoothing
cv_scores_nb = cross_val_score(nb, X, y, cv=cv, scoring='accuracy')

print("\nNaïve Bayes:")
print(f"  CV Scores: {cv_scores_nb}")
print(f"  Mean: {cv_scores_nb.mean():.4f} (+/- {cv_scores_nb.std():.4f})")
```

### Expected Results

```
Logistic Regression CV: 0.60-0.75 accuracy
Naïve Bayes CV:        0.58-0.72 accuracy

High variance expected due to small dataset!
```

### Why Traditional Models Can Work

**Advantages for Small Data:**
1. ✅ Fewer parameters than deep learning
2. ✅ Less prone to overfitting (with regularization)
3. ✅ Faster training
4. ✅ More interpretable
5. ✅ Work reasonably with 200+ samples

**Key Principle:**
> With 234 samples, use simple models with strong regularization

---

## Question 3: Other Models Besides VADER?

### ✅ Yes - Multiple Options

#### Option A: Pre-trained Transformers (Recommended)

```python
from transformers import pipeline

# No training needed - uses pre-trained model
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# Analyze all 234 reviews
results = sentiment_analyzer(df['review_text'].tolist())

df['predicted_sentiment'] = [r['label'] for r in results]
df['confidence'] = [r['score'] for r in results]

# Expected accuracy: 80-90%
```

#### Option B: TextBlob (Simpler than VADER)

```python
from textblob import TextBlob

def textblob_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    
    if polarity > 0.1:
        return 'positive'
    elif polarity < -0.1:
        return 'negative'
    else:
        return 'neutral'

df['sentiment'] = df['review_text'].apply(textblob_sentiment)

# Expected accuracy: 70-80%
```

#### Option C: Fine-tuned BERT (Best Deep Learning)

```python
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.model_selection import train_test_split
import torch

# Prepare data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['review_text'].tolist(),
    y.tolist(),
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Load pre-trained model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3  # positive, neutral, negative
)

# Tokenize
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)

# Create dataset
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

train_dataset = SentimentDataset(train_encodings, train_labels)
val_dataset = SentimentDataset(val_encodings, val_labels)

# Training arguments for SMALL dataset
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,        # Limited epochs
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

# Expected accuracy: 70-85% (best deep learning option for small data)
```

### Comparison for 234 Reviews

| Model | Training Time | Accuracy | Pros | Cons |
|-------|--------------|----------|------|------|
| **VADER** | None | 75-85% | No training, fast | Not customizable |
| **TextBlob** | None | 70-80% | Very simple | Less accurate |
| **Pre-trained BERT** | None | 80-90% | Best accuracy | Slower inference |
| **Fine-tuned BERT** | ~10 min | 70-85% | Adapts to your data | Complex setup |
| **LogReg** | <1 min | 60-75% | Simple, fast | Limited by data size |
| **Naïve Bayes** | <1 min | 58-72% | Works on small data | Assumes independence |
| **LSTM** | ~5 min | 45-60% | Can learn sequences | Severe overfitting |

---

## Question 4: How to Evaluate VADER Results?

### Step-by-Step Evaluation Guide

#### Step 1: Compare VADER with Ground Truth (If Available)

```python
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming you have true labels
true_labels = df['sentiment_actual']  # Your labeled data
vader_predictions = df['sentiment_vader']  # VADER predictions

# Calculate metrics
accuracy = accuracy_score(true_labels, vader_predictions)
f1 = f1_score(true_labels, vader_predictions, average='weighted')

print(f"VADER Performance:")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  F1-Score: {f1:.4f}")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(true_labels, vader_predictions))

# Confusion matrix
cm = confusion_matrix(true_labels, vader_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['negative', 'neutral', 'positive'],
            yticklabels=['negative', 'neutral', 'positive'])
plt.title('VADER Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
```

#### Step 2: Analyze Confidence Scores

```python
import pandas as pd
import numpy as np

# Analyze VADER compound scores
df['compound_score'] = df['review_text'].apply(
    lambda x: analyzer.polarity_scores(x)['compound']
)

# Statistics by sentiment
print("\nCompound Score Statistics by Sentiment:")
for sentiment in ['positive', 'neutral', 'negative']:
    subset = df[df['sentiment_vader'] == sentiment]
    print(f"\n{sentiment.upper()}:")
    print(f"  Mean:   {subset['compound_score'].mean():.4f}")
    print(f"  Median: {subset['compound_score'].median():.4f}")
    print(f"  Std:    {subset['compound_score'].std():.4f}")
    print(f"  Range:  [{subset['compound_score'].min():.4f}, {subset['compound_score'].max():.4f}]")

# Visualize distribution
plt.figure(figsize=(12, 4))

for i, sentiment in enumerate(['negative', 'neutral', 'positive'], 1):
    plt.subplot(1, 3, i)
    subset = df[df['sentiment_vader'] == sentiment]
    plt.hist(subset['compound_score'], bins=20, alpha=0.7)
    plt.title(f'{sentiment.capitalize()} Reviews')
    plt.xlabel('Compound Score')
    plt.ylabel('Frequency')
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
```

#### Step 3: Find Misclassifications

```python
# Identify cases where VADER disagrees with true labels
misclassified = df[df['sentiment_actual'] != df['sentiment_vader']]

print(f"\nMisclassifications: {len(misclassified)} / {len(df)} ({len(misclassified)/len(df)*100:.1f}%)")

print("\nSample Misclassified Reviews:")
for idx, row in misclassified.head(10).iterrows():
    print(f"\nReview: {row['review_text'][:80]}...")
    print(f"  True:      {row['sentiment_actual']}")
    print(f"  Predicted: {row['sentiment_vader']}")
    print(f"  Compound:  {row['compound_score']:.4f}")
```

#### Step 4: Threshold Analysis

```python
# Try different thresholds for classification
thresholds = [
    (0.05, -0.05),   # Default VADER
    (0.1, -0.1),     # More conservative
    (0.2, -0.2),     # Very conservative
    (0.0, 0.0)       # Any positive/negative
]

results = []

for pos_threshold, neg_threshold in thresholds:
    def classify_with_threshold(compound):
        if compound >= pos_threshold:
            return 'positive'
        elif compound <= neg_threshold:
            return 'negative'
        else:
            return 'neutral'
    
    predictions = df['compound_score'].apply(classify_with_threshold)
    acc = accuracy_score(df['sentiment_actual'], predictions)
    f1 = f1_score(df['sentiment_actual'], predictions, average='weighted')
    
    results.append({
        'pos_threshold': pos_threshold,
        'neg_threshold': neg_threshold,
        'accuracy': acc,
        'f1_score': f1
    })

results_df = pd.DataFrame(results)
print("\nThreshold Optimization:")
print(results_df)

# Find best threshold
best = results_df.loc[results_df['f1_score'].idxmax()]
print(f"\nBest Thresholds: positive={best['pos_threshold']}, negative={best['neg_threshold']}")
print(f"Best F1-Score: {best['f1_score']:.4f}")
```

#### Step 5: Per-Class Analysis

```python
from sklearn.metrics import precision_recall_fscore_support

# Get per-class metrics
precision, recall, f1, support = precision_recall_fscore_support(
    df['sentiment_actual'],
    df['sentiment_vader'],
    labels=['negative', 'neutral', 'positive']
)

metrics_df = pd.DataFrame({
    'Sentiment': ['negative', 'neutral', 'positive'],
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1,
    'Support': support
})

print("\nPer-Class Performance:")
print(metrics_df.to_string(index=False))

# Identify weakest class
weakest_class = metrics_df.loc[metrics_df['F1-Score'].idxmin()]
print(f"\nWeakest Class: {weakest_class['Sentiment']} (F1={weakest_class['F1-Score']:.4f})")
```

#### Step 6: Compare with Other Models

```python
# Compare VADER with traditional models
from sklearn.model_selection import cross_val_predict

# Get predictions from different models
vader_pred = df['sentiment_vader']

# Logistic Regression predictions
lr_pred = cross_val_predict(log_reg, X, y, cv=5)
lr_pred_labels = le.inverse_transform(lr_pred)

# Naïve Bayes predictions
nb_pred = cross_val_predict(nb, X, y, cv=5)
nb_pred_labels = le.inverse_transform(nb_pred)

# Compare all models
comparison = pd.DataFrame({
    'Model': ['VADER', 'Logistic Regression', 'Naïve Bayes'],
    'Accuracy': [
        accuracy_score(df['sentiment_actual'], vader_pred),
        accuracy_score(df['sentiment_actual'], lr_pred_labels),
        accuracy_score(df['sentiment_actual'], nb_pred_labels)
    ],
    'F1-Score (Weighted)': [
        f1_score(df['sentiment_actual'], vader_pred, average='weighted'),
        f1_score(df['sentiment_actual'], lr_pred_labels, average='weighted'),
        f1_score(df['sentiment_actual'], nb_pred_labels, average='weighted')
    ]
})

print("\nModel Comparison:")
print(comparison.to_string(index=False))

# Visualize
plt.figure(figsize=(10, 5))
x = np.arange(len(comparison))
width = 0.35

plt.subplot(1, 2, 1)
plt.bar(x, comparison['Accuracy'], width)
plt.xticks(x, comparison['Model'], rotation=45, ha='right')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.ylim([0, 1])

plt.subplot(1, 2, 2)
plt.bar(x, comparison['F1-Score (Weighted)'], width)
plt.xticks(x, comparison['Model'], rotation=45, ha='right')
plt.ylabel('F1-Score')
plt.title('Model F1-Score Comparison')
plt.ylim([0, 1])

plt.tight_layout()
plt.show()
```

---

## Complete Workflow: Meeting Your Requirements

### For Your Assignment with 234 Reviews

```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns

# === STEP 1: Traditional Models ===
print("="*80)
print("TRADITIONAL MODELS (Logistic Regression, Naïve Bayes)")
print("="*80)

# Prepare data
le = LabelEncoder()
y = le.fit_transform(df['sentiment'])

# Vectorize
vectorizer = TfidfVectorizer(max_features=100, min_df=2, max_df=0.8)
X = vectorizer.fit_transform(df['review_text'])

# Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Logistic Regression
lr = LogisticRegression(C=0.1, max_iter=1000, class_weight='balanced')
lr_scores = cross_val_score(lr, X, y, cv=cv, scoring='accuracy')
lr_f1 = cross_val_score(lr, X, y, cv=cv, scoring='f1_weighted')

print(f"\nLogistic Regression:")
print(f"  Accuracy: {lr_scores.mean():.4f} (+/- {lr_scores.std():.4f})")
print(f"  F1-Score: {lr_f1.mean():.4f} (+/- {lr_f1.std():.4f})")

# Naïve Bayes
nb = MultinomialNB(alpha=1.0)
nb_scores = cross_val_score(nb, X, y, cv=cv, scoring='accuracy')
nb_f1 = cross_val_score(nb, X, y, cv=cv, scoring='f1_weighted')

print(f"\nNaïve Bayes:")
print(f"  Accuracy: {nb_scores.mean():.4f} (+/- {nb_scores.std():.4f})")
print(f"  F1-Score: {nb_f1.mean():.4f} (+/- {nb_f1.std():.4f})")

# === STEP 2: VADER (Baseline) ===
print("\n" + "="*80)
print("VADER SENTIMENT ANALYSIS (Baseline)")
print("="*80)

analyzer = SentimentIntensityAnalyzer()

def vader_sentiment(text):
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        return 'positive'
    elif compound <= -0.05:
        return 'negative'
    else:
        return 'neutral'

df['vader_pred'] = df['review_text'].apply(vader_sentiment)

vader_acc = (df['vader_pred'] == df['sentiment']).mean()
vader_f1 = f1_score(df['sentiment'], df['vader_pred'], average='weighted')

print(f"\nVADER:")
print(f"  Accuracy: {vader_acc:.4f}")
print(f"  F1-Score: {vader_f1:.4f}")

# === STEP 3: Transformer (Optional) ===
print("\n" + "="*80)
print("PRE-TRAINED TRANSFORMER (DistilBERT)")
print("="*80)

from transformers import pipeline

try:
    sentiment_pipe = pipeline("sentiment-analysis")
    transformer_results = sentiment_pipe(df['review_text'].tolist())
    # Note: May need mapping if model outputs different labels
    print("Transformer model loaded successfully")
except:
    print("⚠️  Transformer model not available (requires internet/GPU)")

# === STEP 4: Summary Comparison ===
print("\n" + "="*80)
print("MODEL COMPARISON SUMMARY")
print("="*80)

summary = pd.DataFrame({
    'Model': ['Logistic Regression', 'Naïve Bayes', 'VADER'],
    'Accuracy': [
        lr_scores.mean(),
        nb_scores.mean(),
        vader_acc
    ],
    'F1-Score': [
        lr_f1.mean(),
        nb_f1.mean(),
        vader_f1
    ],
    'Std Dev': [
        lr_scores.std(),
        nb_scores.std(),
        0.0  # VADER has no variance (not trained)
    ]
})

print(summary.to_string(index=False))

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].bar(summary['Model'], summary['Accuracy'])
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Model Accuracy Comparison')
axes[0].set_ylim([0, 1])
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')

axes[1].bar(summary['Model'], summary['F1-Score'])
axes[1].set_ylabel('F1-Score')
axes[1].set_title('Model F1-Score Comparison')
axes[1].set_ylim([0, 1])
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.show()

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("\nFor 234 reviews:")
print("✅ Traditional models (LogReg, NB) are viable with cross-validation")
print("✅ VADER provides competitive baseline without training")
print("⚠️  Deep learning (LSTM) will overfit severely - not recommended")
print("✅ Fine-tuned BERT is feasible but complex for this size")
```

---

## Final Recommendations

### For Your Assignment

**✅ DO THIS:**

1. **Use ALL required models** but report honestly:
   - Logistic Regression: Use with cross-validation
   - Naïve Bayes: Use with cross-validation
   - VADER: Use as strong baseline
   - BERT: Fine-tune pre-trained model OR use pre-trained directly
   - LSTM: Mention it will overfit, show results as comparison

2. **Report metrics properly:**
   ```
   Model              | Accuracy | F1-Score | Notes
   -------------------|----------|----------|------------------
   Logistic Regression| 0.65     | 0.64     | Cross-validation
   Naïve Bayes        | 0.62     | 0.61     | Cross-validation
   VADER              | 0.78     | 0.77     | No training needed
   BERT (fine-tuned)  | 0.72     | 0.71     | 3 epochs
   LSTM               | 0.52     | 0.48     | Severe overfitting
   ```

3. **In your report, explain:**
   - Dataset is small (234 reviews)
   - Traditional models viable with regularization
   - Deep learning limited by data size
   - VADER performs best due to pre-training
   - Cross-validation used to combat small sample

**❌ DON'T DO THIS:**
- Don't use simple train/test split (too small)
- Don't claim LSTM works well (it won't)
- Don't report inflated metrics
- Don't ignore the data size limitation

### Bottom Line

**Yes, you can and should try traditional models on 234 reviews** - they will work reasonably well with proper cross-validation. VADER will likely perform best, but showing all models demonstrates understanding of when each approach is appropriate.

