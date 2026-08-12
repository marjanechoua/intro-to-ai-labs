
# AI, Machine Learning & Data Science Cheat Sheet

## 1. AI landscape

### Artificial Intelligence

Broad field of creating systems that perform tasks associated with intelligence.

Common areas include:

* **Machine Learning** → learn patterns from data
* **Knowledge-based AI** → explicit knowledge, rules, RDF, knowledge graphs
* **Natural Language Processing** → process human language
* **Computer Vision** → process images/video
* **Complex Event Processing** → analyze streams of events
* **Ethical AI** → fairness, bias, responsible AI

### When to use ML?

Use ML when:

* there is enough data
* the relationship is difficult to write as explicit rules
* patterns can be learned from examples
* predictions or classifications are needed

Example:

```text
Movie features → predict revenue
Email text → predict spam/not spam
Image → predict digit
```

Don't necessarily use ML when simple deterministic rules solve the problem better.

```python
if temperature > 30:
    print("Hot")
```

---

# 2. Python & Pandas

## Lambda

Useful for applying a simple function.

```python
df["B"] = df["A"].apply(lambda x: 2 ** x)
```

For:

```text
0 → 1
1 → 2
2 → 4
3 → 8
```

Important:

```python
2 ** x
```

means **2 to the power of x**, not square root.

---

## List comprehension

Compact way to create/process lists.

```python
result = [x * 2 for x in range(5)]
```

Equivalent to:

```python
result = []

for x in range(5):
    result.append(x * 2)
```

Example:

```python
sum([2 ** x - 1 for x in range(4)])
```

---

## `range()`

```python
range(4)
```

gives:

```text
0, 1, 2, 3
```

The ending value is **not included**.

---

# 3. Machine Learning Workflow

Typical workflow:

```text
Load data
   ↓
Explore / understand data
   ↓
Clean data
   ↓
Feature engineering
   ↓
Split X and y
   ↓
Train model
   ↓
Validate
   ↓
Predict
   ↓
Evaluate / deploy
```

Example:

```python
data = pd.read_csv("data.csv")

X = data[features]
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
```

---

# 4. X and y

## Features: X

Input information.

```text
age
income
budget
popularity
runtime
```

## Target: y

What you want to predict.

```text
price
revenue
survived
spam
genre
```

Conceptually:

[
y = f(X)
]

The model tries to learn an approximation:

[
\hat{y} = g(X)
]

---

#  5. Types of Machine Learning

## Supervised Learning

Training data has inputs **and known labels/targets**.

```text
X → y
```

Examples:

* house features → house price
* email → spam/not spam
* Titanic passenger → survived/not survived

### Classification

Predict a category.

```text
spam / not spam
cat / dog
survived / died
```

### Regression

Predict a number.

```text
house price
movie revenue
temperature
```

---

## Unsupervised Learning

No target labels.

The model searches for structure or patterns.

Examples:

* clustering customers
* grouping similar documents
* dimensionality reduction

```text
X → discover patterns
```

---

## Reinforcement Learning

An **agent interacts with an environment**.

```text
Agent
  ↓ action
Environment
  ↓ reward / penalty
Agent learns better actions
```

Examples:

* game-playing AI
* robotics
* autonomous decision-making

Key idea:

> Learn through rewards and penalties.

---

# 6. Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(X, y)
```

The model:

```text
TRAINING DATA
     ↓
model.fit()
     ↓
learn patterns
     ↓
TEST DATA
     ↓
model.predict()
     ↓
evaluate generalization
```

### Why not train and test on the same data?

Because the model may simply memorize training examples.

Testing on unseen data estimates **generalization ability**.

---

# 7. Cross-Validation

Example: 5-fold cross-validation.

```text
Round 1: [TEST][TRAIN][TRAIN][TRAIN][TRAIN]
Round 2: [TRAIN][TEST][TRAIN][TRAIN][TRAIN]
Round 3: [TRAIN][TRAIN][TEST][TRAIN][TRAIN]
Round 4: [TRAIN][TRAIN][TRAIN][TEST][TRAIN]
Round 5: [TRAIN][TRAIN][TRAIN][TRAIN][TEST]
```

Final score:

```text
average(score1, score2, ..., score5)
```

### Advantages

* more reliable evaluation
* every sample is eventually used for validation
* useful for smaller datasets

### Disadvantages

* slower
* computationally expensive
* trains the model multiple times

---

# 8. `fit()`, `transform()`, `fit_transform()`, `predict()`

This is extremely useful to remember.

## `fit()`

**Learn something from data.**

For preprocessing:

```python
scaler.fit(X_train)
```

Learns statistics such as mean/std.

For ML:

```python
model.fit(X_train, y_train)
```

Learns the relationship:

```text
X → y
```

---

## `transform()`

Apply something already learned.

```python
X_test = scaler.transform(X_test)
```

The test data uses the statistics learned from training data.

---

## `fit_transform()`

Do both:

```text
fit
↓
learn

transform
↓
apply
```

Example:

```python
X_train = vectorizer.fit_transform(X_train)
```

The vectorizer:

1. learns the vocabulary
2. transforms the text into numbers

---

## `predict()`

Use the trained model to predict outputs.

```python
predictions = model.predict(X_test)
```

---

### Golden rule

```python
X_train = transformer.fit_transform(X_train)

X_test = transformer.transform(X_test)
```

❌ Avoid:

```python
X_test = transformer.fit_transform(X_test)
```

This can cause **data leakage**.

---

# 9. Data Leakage

Data leakage happens when information from outside the training process incorrectly influences training.

Example:

```python
scaler.fit(all_data)
```

before splitting.

The model indirectly receives information about the test data.

Better:

```python
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
```

Rule:

> Learn preprocessing parameters only from training data.

---

# 10. Overfitting

```text
Training performance: 99%
Test performance:     65%
```

Possible overfitting.

The model learned:

* useful patterns ✅
* noise ❌
* overly specific training examples ❌

Result:

> Poor generalization to unseen data.

---

# 11. Classification Metrics

Confusion matrix:

| Actual / Predicted | Positive | Negative |
| ------------------ | -------- | -------- |
| Positive           | TP       | FN       |
| Negative           | FP       | TN       |

## Accuracy

[
Accuracy =
\frac{TP + TN}{TP + TN + FP + FN}
]

Overall percentage correct.

---

## Recall

[
Recall =
\frac{TP}{TP + FN}
]

Question:

> Of all actual positives, how many did we find?

Important when **missing a positive is dangerous**.

Examples:

* cancer detection
* fraud detection
* fire detection

---

## Precision

[
Precision =
\frac{TP}{TP + FP}
]

Question:

> When the model says positive, how often is it correct?

Important when **false positives are expensive or dangerous**.

Example:

* smartphone fingerprint security

---

# 12. Regression Metrics

## MAE — Mean Absolute Error

[
MAE =
\frac{1}{n}
\sum |y-\hat{y}|
]

Average absolute error.

---

## MSE — Mean Squared Error

[
MSE =
\frac{1}{n}
\sum(y-\hat{y})^2
]

Large errors receive stronger penalties.

---

## RMSE

[
RMSE = \sqrt{MSE}
]

Useful because it returns to the original unit.

Example:

```text
Predicting house prices:

MSE → €²
RMSE → €
```

---

# 13. Feature Engineering

Creating new useful features.

Example from Titanic:

```python
data["FamilySize"] = data["SibSp"] + data["Parch"]

data["IsAlone"] = (
    data["FamilySize"] == 0
).astype(int)
```

Instead of only:

```text
SibSp
Parch
```

we create:

```text
FamilySize
IsAlone
```

The goal:

> Give the model more useful representations of the data.

---

# 14. Random Forest

Random Forest consists of many decision trees.

```text
Tree 1 → prediction
Tree 2 → prediction
Tree 3 → prediction
...
        ↓
combine predictions
```

Classification:

```text
majority vote
```

Regression:

```text
average prediction
```

Example:

```python
RandomForestClassifier()
RandomForestRegressor()
```

Important parameters:

```python
n_estimators
max_depth
min_samples_leaf
random_state
```

---

# 15. Natural Language Processing

Typical NLP pipeline:

```text
Text
 ↓
Tokenization
 ↓
POS tagging
 ↓
Lemmatization
 ↓
Named Entity Recognition
```

## Tokenization

```text
"Leonardo painted Mona Lisa"
```

↓

```text
Leonardo
painted
Mona
Lisa
```

---

## POS tagging

Assign grammatical roles.

```text
painted → VERB
Leonardo → PROPN / NOUN
```

---

## Lemmatization

Reduce words to their base form.

```text
painted → paint
running → run
```

---

## Named Entity Recognition

Recognize entities.

```text
Leonardo → PERSON
Paris → LOCATION
```

---

# 16. Bag of Words

Texts:

```text
Text 1: cat likes milk
Text 2: dog likes milk
```

Vocabulary:

```text
cat | dog | likes | milk
```

| Text   | cat | dog | likes | milk |
| ------ | --: | --: | ----: | ---: |
| Text 1 |   1 |   0 |     1 |    1 |
| Text 2 |   0 |   1 |     1 |    1 |

Bag of Words ignores word order.

---

# 🔤 17. N-Grams

For:

```text
Leonardo painted Mona Lisa
```

Bigrams (`n = 2`):

```text
Leonardo painted
painted Mona
Mona Lisa
```

---

# 18. TF-IDF

Measures how important a word is.

## Term Frequency

[
TF(t,d)=
\frac{\text{count of }t\text{ in }d}
{\text{number of tokens in }d}
]

## Inverse Document Frequency

[
IDF(t,D)=
\log
\left(
\frac{|D|}
{\text{documents containing }t}
\right)
]

## TF-IDF

[
TFIDF = TF \times IDF
]

Important intuition:

```text
Common word → low IDF
Rare word → high IDF
```

If a word appears in every document:

[
IDF = \log(1)=0
]

Therefore:

[
TFIDF=0
]

---

# 19. Neural Networks

Basic neuron:

[
z = \sum w_ix_i+b
]

Then activation:

[
output=\sigma(z)
]

---

## ReLU

[
ReLU(x)=\max(0,x)
]

Examples:

```text
ReLU(5) = 5
ReLU(-3) = 0
```

Why useful?

Without activation functions, multiple layers would still behave essentially like a linear transformation.

Activation functions introduce **non-linearity**, allowing neural networks to learn more complex patterns.

---

#  20. Computer Vision & CNNs

Typical image preparation:

```python
X = X / 255.0
```

Normalize pixel values:

```text
0–255
↓
0–1
```

Reshape:

```python
X.reshape(-1, 28, 28, 1)
```

Meaning:

```text
number of images
height = 28
width = 28
channels = 1
```

Typical CNN:

```text
Input
 ↓
Conv2D
 ↓
MaxPooling
 ↓
Conv2D
 ↓
MaxPooling
 ↓
Flatten
 ↓
Dense
 ↓
Softmax
```

### Main functions

**Conv2D** → detects patterns/features

**MaxPooling** → reduces spatial size

**Flatten** → converts feature maps into a vector

**Dense** → combines learned features

**Softmax** → outputs probabilities for classes

---

# 21. One-Hot Encoding

Labels:

```text
red
green
blue
```

Encoding:

```text
red   → [1,0,0]
green → [0,1,0]
blue  → [0,0,1]
```

Each class gets its own position.

---

#  22. Knowledge Graphs / RDF

RDF consists of triples:

```text
Subject → Predicate → Object
```

Example:

```text
:Book1 :author :Author1
```

```text
Subject:  Book1
Predicate: author
Object:   Author1
```

Another:

```text
:Author1 :name "J.R.R. Tolkien"
```

Objects can be:

* resources/entities
* literals such as strings or numbers

---

# 23. SPARQL

SPARQL queries knowledge graphs.

Basic structure:

```sparql
SELECT ?variable
WHERE {
    pattern
}
```

Variables start with:

```text
?
```

Example:

```sparql
SELECT ?title
WHERE {
    ?movie a :Movie ;
           :title ?title .
}
```

Meaning:

> Find all things that are movies and return their titles.

---

## SPARQL joins

This is a very important concept.

Example:

```sparql
SELECT ?movie
WHERE {
    ?movie :hasDirector ?director .
    ?director :bornIn :London .
}
```

The variable:

```text
?director
```

appears in both patterns.

That automatically creates a **join**.

Conceptually:

```text
Movie
 ↓ hasDirector
Director
 ↓ bornIn
London
```

The shared variable connects the two RDF patterns.

---

# 24. Flask, Jinja, NLP & Knowledge Graph Architecture

Your application followed a nice architecture:

```text
User
 ↓
HTML form
 ↓
Flask
 ↓
Python logic
 ↓
NLP / SPARQL
 ↓
Result
 ↓
Jinja template
 ↓
HTML response
```

---

## Flask

Handles HTTP requests.

```python
@app.route("/", methods=["GET", "POST"])
def index():
```

Receives the user request.

---

## User input

```python
user_text = request.form.get("description", "")
```

Gets the text entered into the HTML form.

---

## NLP

```python
genres = predict_genres(user_text)
```

Processes the text and predicts genres.

---

## Knowledge Graph

```python
results = graph.query(query)
```

Runs SPARQL against the RDF graph.

---

## Jinja

Passes Python variables into HTML:

```python
return render_template(
    "index.html",
    genres=genres
)
```

In HTML:

```html
{% for genre in genres %}
    {{ genre }}
{% endfor %}
```

---

# 25. Complex Event Processing

Often used for streams of events.

Examples:

* stock prices
* sensor data
* clicks
* transactions

## Moving Average

For window size 3:

[
MA_i=
\frac{x_{i-2}+x_{i-1}+x_i}{3}
]

Example:

```text
Values:

4, 6, 8, 10
```

Moving averages:

```text
-
-
(4+6+8)/3 = 6
(6+8+10)/3 = 8
```

Pandas:

```python
df.rolling(window=3).mean()
```

---

# 26. AI Ethics & Bias

## Data Bias

Bias originating from the data.

Example:

Training data contains mostly one demographic group.

---

## Algorithmic Bias

Bias introduced by the algorithm or system design.

---

## Inductive Bias

Assumptions used by the learning algorithm.

This is **not necessarily bad**.

Example:

A linear regression model assumes a linear relationship.

---

## Statistical Bias

Difference between an estimator's expected value and the true value.

---

### Important principle

AI is **not automatically objective**.

AI systems can reproduce or amplify:

* biased training data
* historical inequalities
* biased assumptions
* measurement errors

---

# 27. A useful project structure

A typical ML project:

```text
project/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── app.py
├── model.py
├── preprocessing.py
├── nlp.py
│
├── templates/
│   ├── base.html
│   └── index.html
│
└── static/
```

Try to separate:

```text
Data processing
ML logic
NLP logic
Knowledge graph logic
Web application
HTML/UI
```

That makes projects easier to understand and maintain.

---

#  The core pattern to remember

A huge amount of AI programming can be reduced to this:

```text
DATA
 ↓
PREPROCESS
 ↓
FEATURES (X)
 ↓
TARGET (y)
 ↓
TRAIN
 ↓
VALIDATE
 ↓
PREDICT
 ↓
EVALUATE
 ↓
DEPLOY
```

And for your web/AI applications:

```text
USER INPUT
 ↓
FLASK
 ↓
AI / NLP / ML / KNOWLEDGE GRAPH
 ↓
RESULT
 ↓
JINJA
 ↓
USER
```

