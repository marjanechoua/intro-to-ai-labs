

# First Question: Which Columns Actually Influence Revenue?

Problem: There are null values and 0 values in `budget`.

* Does 0 mean a real budget of 0, or is it missing?
* Can we really trust this column?

We have inputs as movie information: **budget, genres, actors, release date, language, etc.**
The output is **revenue**.

The goal: learn the relationships between movie properties and revenue.

---

## Steps to Fulfill This Exercise

1. Problem Definition
2. Understanding the Dataset
3. Data Quality and Challenges
4. Feature Ideas and Hypotheses
5. Data Preprocessing
6. Feature Engineering
7. Model Training
8. Model Validation
9. Prediction and Submission
10. Reflection / Possible Improvements

---

## Problem Definition

Inputs: movie information (`budget`, `genres`, `actors`, `release_date`, `language`, etc.)
Output: `revenue`

This is a **supervised machine learning regression task** because the target value is continuous.

---

## Understanding the Dataset

We first need to understand **which columns may influence revenue**.

Questions to ask for each column:

* What does this column represent?
* Is it numeric, categorical, or textual?
* Are there missing values?
* Is it useful for prediction?

| Type                      | Examples                    |
| ------------------------- | --------------------------- |
| Numeric                   | budget, popularity, runtime |
| Text                      | overview, tagline           |
| Categories                | original_language           |
| Nested lists/dictionaries | genres, cast, crew          |

| Column                  | Type      | Missing values | Useful for revenue prediction |
| ----------------------- | --------- | -------------- | ----------------------------- |
| `belongs_to_collection` | list/dict | yes            | maybe                         |
| `budget`                | numeric   | yes            | yes                           |
| `genres`                | list/dict | yes            | maybe                         |
| `popularity`            | numeric   | yes            | yes                           |
| `production_companies`  | list/dict | yes            | maybe                         |
| `production_countries`  | list/dict | yes            | maybe                         |
| `release_date`          | date/text | yes            | maybe                         |
| `runtime`               | numeric   | yes            | maybe                         |
| `cast`                  | list/dict | yes            | maybe                         |
| `crew`                  | list/dict | yes            | maybe                         |

---

## Data Quality and Challenges

Some columns contain **missing or inconsistent values**.

* `budget` sometimes contains 0.

  * Unclear if 0 means “actual budget 0” or “missing”.
  * This affects reliability.

Other challenges:

* Missing values
* Nested JSON-like structures
* Inconsistent formats
* High-cardinality categorical data

---

## Feature Ideas and Hypotheses

* `budget` → likely strong influence
* `popularity` → likely strong influence
* `genres` → may influence audience size
* Famous actors → may increase popularity
* `release_date` → seasonal effects possible
* `runtime` → possible weak influence
* `language` → may affect international audience reach

---

## Data Preprocessing

Before training a model, the dataset must be **cleaned and standardized**:

* Handle missing values
* Investigate suspicious zeros
* Parse nested JSON-like strings
* Standardize formats

---

## Feature Engineering

Many columns contain complex nested structures, which are **hard for models to use directly**.

Examples:

Original:

```text
"[{'id': 35, 'name': 'Comedy'}]"
```

Transformations:

* Extract genre names
* Create binary indicators (`is_comedy = 1`)
* Extract release year from `release_date`
* Count number of cast members
* Create `has_collection`
* Count production companies

**Key principle:**
If it’s not numeric → convert it or extract something numeric

Examples:

```python
# Number of genres
data["genres_count"] = data["genres"].fillna("[]").apply(len)

# Is English
data["is_english"] = (data["original_language"] == "en").astype(int)

# Number of cast members
data["cast_count"] = data["cast"].fillna("[]").apply(len)
```

---

## Numeric Transformation

Most ML models require numeric input:

* Categorical → encode
* Dates → transform into year/month
* Text → simplify into numeric features

---

## Model Training

The model learns patterns like:

* Higher budgets often increase revenue
* Franchise movies may earn more
* Some genres may generally perform better

---

## Model Validation

* Use **cross-validation** to check generalization
* Regression performance measured with **RMSE**

[
\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}
]

Lower RMSE → better performance

---

## Feature Importance & Lessons Learned

* Added `budget_log` and `release_year` → barely changed RMSE (weak predictors)
* Added `genres` + `is_empty_genre` → RMSE slightly worse (~83.95M → ~84.10M)

  * **Lesson:** Feature quality matters more than quantity
* Added `genre_count` → small improvement
* Added `is_in_collection` → big improvement (~79.3M RMSE)
* Added `cast_size` → RMSE improved (~78.3M)

  * Captures blockbuster films with large ensembles
* Features like `production_companies` or `keywords` → noisy, did not help

### Feature Importance Example

```text
budget_log                0.439
budget_popularity_ratio   0.153
release_year              0.095
runtime                   0.081
popularity                0.075
cast_size                 0.056
```

* Most important: `budget_log`
* Second: `budget_popularity_ratio` (interaction feature)
* Weak features (`is_action`, `is_english`, `is_in_collection`) still added small improvements

**Log-transforming revenue** also improved RMSE:

```python
y = np.log1p(data["revenue"])
```

* Reduces effect of extreme outliers
* RMSE improved from ~2.62 → ~2.13

**Key insights:**

* Feature engineering matters more than feature quantity
* Feature interactions can improve predictions
* Even weak features contribute small but useful signals

---

## New Features: `director_freq` and `top_disney`

### Director

* Extracted director from `crew` column
* Converted to numeric via **frequency**: `director_freq`
* Frequent directors → likely experienced/commercially active

```python
director_freq = data["director"].value_counts()
data["director_freq"] = data["director"].map(director_freq)
```

### Production Company: Disney

* Created binary feature `top_disney`
* 1 if Disney involved, 0 otherwise
* Captures signal for blockbuster production
* No target leakage

```python
data["top_disney"] = data["production_companies"].apply(lambda x: int("Disney" in get_companies(x)))
```

### Results

RMSE: 2.1347

```text
budget_log                0.4359
budget_popularity_ratio   0.1499
release_year              0.0901
runtime                   0.0784
popularity                0.0718
cast_size                 0.0526
release_month             0.0413
keyword_count             0.0384
director_freq             0.0226
genre_count               0.0190
top_disney                0.0000
```

* `director_freq` → small but meaningful signal
* `top_disney` → negligible importance, but interpretable


