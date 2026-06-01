

# NLP Text Classification – Movie Genre Prediction

## 1. Task Description

The goal of this assignment is to build a machine learning model that predicts whether a movie belongs to the **Drama genre** based on its **overview text**.

This is a supervised text classification task using classical Natural Language Processing (NLP) techniques such as TF-IDF and machine learning classifiers.

---

## 2. Pipeline Overview

```text
Load Data → Pre-processing → TF-IDF Vectorization → Classification → Evaluation (Cross-Validation)
```

---

## 3. Step 1: Import Libraries

```python
import csv
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
```

### Explanation

| Library        | Purpose                                            |
| -------------- | -------------------------------------------------- |
| pandas         | Load and process dataset                           |
| TfidfVectorizer | Convert text into numerical TF-IDF features        |
| MultinomialNB  | Baseline classifier (Naive Bayes)                  |
| Pipeline       | Combines preprocessing and model steps             |
| cross_val_score | Evaluates model performance using cross-validation |

---

## 4. Step 2: Load Dataset

```python
data = pd.read_csv(
    'data/Movie_Overview_Classification.csv',
    encoding='utf8',
    delimiter=',',
    quotechar='|',
    quoting=csv.QUOTE_MINIMAL
)
```

### Explanation

The dataset is loaded using `pandas.read_csv()`.

Important parameters:

* `encoding='utf8'`: supports special characters
* `delimiter=','`: CSV column separation
* `quotechar='|'`: handles text fields correctly
* `quoting=csv.QUOTE_MINIMAL`: only quotes necessary fields

---

## 5. Step 3: Pre-processing

```python
data = data.dropna(subset=['overview', 'genre_Drama'])

X_text = data['overview']
y_genre = data['genre_Drama']
```

### Explanation

* Missing values in `overview` and `genre_Drama` are removed
* `X_text`: input features (movie descriptions)
* `y_genre`: binary target label

  * 1 = Drama
  * 0 = Not Drama

---

## 6. Step 4: Baseline Model

```python
baseline_model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", MultinomialNB())
])
```

### Explanation

This baseline model uses:

* **TF-IDF Vectorization**: Converts text into weighted word features
* **Multinomial Naive Bayes**: Probabilistic classifier commonly used for text classification

This model serves as a simple reference performance.

---

## 7. Step 5: Model Evaluation (Baseline)

```python
baseline_scores = cross_val_score(
    baseline_model,
    X_text,
    y_genre,
    cv=5,
    scoring='accuracy'
)

print("Baseline Accuracy:", baseline_scores.mean())
```

### Result

*Baseline Accuracy*: 0.6648

---

## 8. Step 6: Optimized Model

```python
optimized_model = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1,2),
        min_df=3,
        max_df=0.85,
        sublinear_tf=False,
        binary=True,
        smooth_idf=True
    )),
    ("clf", MultinomialNB())
])
```

### Explanation

The model was improved through TF-IDF feature tuning. The classifier remains unchanged (Multinomial Naive Bayes), allowing the impact of feature engineering to be evaluated independently from the classification algorithm.
The optimized TF-IDF configuration includes:
- unigram and bigram features (ngram_range=(1,2))
- removal of rare terms (min_df=3)
- filtering of extremely common terms (max_df=0.85)
- binary term representation (binary=True)
- IDF smoothing (smooth_idf=True)


#### Improved TF-IDF settings:

• ngram_range=(1,2)
  Includes both single words (unigrams) and word pairs (bigrams).

• min_df=3
  Removes very rare terms that appear in fewer than three documents.

• max_df=0.85
  Removes extremely frequent terms that appear in more than 85% of documents.

• binary=True
  Uses only the presence or absence of a term instead of raw frequency counts.

• smooth_idf=True
  Applies smoothing to IDF values to avoid division-by-zero issues.


The optimized model uses TF-IDF with unigram and bigram features (ngram_range=(1,2)) and removes very rare terms (min_df=2). This increased the cross-validation accuracy from 66.48% to 66.98%, resulting in an improvement of approximately 0.5 percentage points.
#### Classifier:

Multinomial Naive Bayes

• MultinomialNB is a probabilistic classifier commonly used for text classification.
• It works particularly well with TF-IDF and bag-of-words representations.
• The classifier remains the same as in the baseline model so that the impact of TF-IDF optimization can be evaluated separately.
---

## 9. Step 7: Model Evaluation (Optimized)

```python
optimized_scores = cross_val_score(
    optimized_model,
    X_text,
    y_genre,
    cv=5,
    scoring='accuracy'
)

print("Optimized Accuracy:", optimized_scores.mean())
```

### Result

**Optimized Accuracy:** `0.6768`

---

## 10. Step 8: Results Comparison

```python
print("Improvement:",
      optimized_scores.mean() - baseline_scores.mean())
```

### Result
| Model                            | Accuracy |
| -------------------------------- | -------- |
| Baseline TF-IDF + MultinomialNB  | 0.6648   |
| Optimized TF-IDF + MultinomialNB | 0.6768   |


## 11. Discussion
This improvement indicates that the additional TF-IDF feature engineering helped the classifier extract more useful information from the movie overviews.
The classifier itself remained unchanged (Multinomial Naive Bayes), meaning that the performance gain can be attributed to the modified text representation rather than a different learning algorithm.
The results demonstrate that feature engineering can have a significant impact on text classification performance, even when using the same classifier.
---

## 12. Key Concepts

| Concept          | Explanation                                |
| ---------------- | ------------------------------------------ |
| Bag-of-Words     | Represents text as word frequency features |
| TF-IDF           | Weighs words based on importance in corpus |
| Naive Bayes      | Probabilistic classifier for text data     |
| Cross-validation | Robust method for model evaluation         |

---

## 13. Conclusion
This assignment demonstrates a complete NLP text classification pipeline using TF-IDF vectorization and machine learning techniques.
The results show that:

- Multinomial Naive Bayes is an effective baseline classifier for text classification tasks.
- TF-IDF parameter tuning can improve model performance.
- Feature engineering plays an important role in NLP applications.
- Cross-validation provides a reliable estimate of model performance.
- The optimized model achieved an accuracy of 67.68%, outperforming the baseline model (66.48%) by approximately 1.20 percentage points.
