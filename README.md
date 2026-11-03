# Machine Learning for Beginners: A Detailed Theory Guide

> Based on the roadmap **"Day 2: Preparing Machine Learning Models: Classification | Regression | Clustering"**.
> Every concept is explained with **What** it is, **Why** it matters, **How** it works, and a **small example**.
> Every model comes with a detailed explanation of **how it works** and **all its important hyperparameters**.

---

## Table of Contents

0. [How to use this guide](#0-how-to-use-this-guide)
1. [Foundations you must know first](#1-foundations-you-must-know-first)
2. [Phase 1: Classification](#2-phase-1-classification)
   - 2.1 [The models](#21-the-classification-models)
   - 2.2 [Classification metrics](#22-classification-metrics)
   - 2.3 [Class imbalance and SMOTE](#23-class-imbalance-and-smote)
   - 2.4 [GridSearchCV and RandomizedSearchCV](#24-gridsearchcv-and-randomizedsearchcv)
   - 2.5 [Phase 1 checklist and full code](#25-phase-1-checklist-and-full-code)
3. [Phase 2: Regression and optimization](#3-phase-2-regression-and-optimization)
   - 3.1 [Regression models](#31-regression-models)
   - 3.2 [Regression metrics](#32-regression-metrics)
   - 3.3 [Overfitting, underfitting, regularization](#33-overfitting-underfitting-and-regularization)
   - 3.4 [Gradient descent](#34-gradient-descent)
4. [Phase 3: Clustering](#4-phase-3-clustering)
5. [Phase 4: Visual interpretation](#5-phase-4-visual-interpretation)
6. [Glossary](#6-glossary)
7. [Resources](#7-resources)

---

## 0. How to use this guide

**Goal of the roadmap:** by the end you should be able to

- Train, tune and compare **3 or more classification models** without data leakage.
- Choose and interpret the right **metrics** (classification, regression, clustering).
- **Explain every choice out loud**: why this split, this scaler, this model, this metric.
- Produce a clean notebook with **commented charts**.

**Suggested order:** read Section 1 completely (it is the base of everything), then follow Phases 1 to 4 in order. After each model, run the small code example yourself. Reading is not enough: ML is learned by running code.

**Libraries used:** `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `imbalanced-learn`, `xgboost`, `joblib`.

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost joblib
```

---

## 1. Foundations you must know first

### 1.1 What is Machine Learning?

**What:** Machine Learning (ML) is a way of making a computer **learn patterns from data** instead of being told explicit rules by a programmer.

**Why:** Some problems have rules too complex to write by hand. Nobody can write `if/else` rules that recognize every spam email, but a model can learn from 100,000 labeled emails.

**How:** You give the algorithm many examples. It adjusts internal numbers (called **parameters**) so that its predictions get closer to reality. This adjustment process is called **training** (or *fitting*).

**Example:**

| Classic programming | Machine learning |
|---|---|
| Rules + Data → Answers | Data + Answers → Rules (the model) |
| You write `if "free money" in email: spam` | You show 10,000 spam and non-spam emails; the model finds the rules itself |

---

### 1.2 Types of learning and the three problems of this roadmap

| Type | Do we have the "right answer" (label)? | Problem types in this roadmap |
|---|---|---|
| **Supervised learning** | Yes | **Classification** (predict a category) and **Regression** (predict a number) |
| **Unsupervised learning** | No | **Clustering** (find natural groups) |

- **Classification:** the output is a **category**. Examples: spam / not spam, disease / healthy, fraud / normal, iris species (3 classes).
  - *Binary* = 2 classes. *Multiclass* = 3 or more classes.
- **Regression:** the output is a **continuous number**. Examples: house price, temperature, salary.
- **Clustering:** there is **no target**. The algorithm groups similar items together. Example: grouping customers by behavior without knowing the groups in advance.

---

### 1.3 Dataset vocabulary

| Word | Meaning | Example (house dataset) |
|---|---|---|
| **Sample / row / observation** | One example | One house |
| **Feature / variable / column** | An input property | Surface, number of rooms, city |
| **Target / label** (`y`) | What we want to predict | Price |
| **`X`** | The table of all features | All columns except price |
| **`y`** | The vector of targets | The price column |

```python
import pandas as pd
df = pd.read_csv("houses.csv")
X = df.drop(columns="price")   # features
y = df["price"]                # target
```

---

### 1.4 Parameters vs hyperparameters (very important)

| | Parameters | Hyperparameters |
|---|---|---|
| **Who sets them?** | The **algorithm**, during training | **You**, before training |
| **Examples** | Weights `w` of linear regression, split thresholds in a tree | `max_depth`, `n_neighbors`, `C`, `learning_rate` |
| **How are they found?** | By optimization (e.g., gradient descent) | By trial: **GridSearchCV / RandomizedSearchCV** (Section 2.4) |

**Analogy:** hyperparameters are the *settings of the oven* (temperature, time) that you choose. Parameters are what the *cake* becomes after baking.

---

### 1.5 Train/test split

**What:** Splitting the dataset in two: a **training set** (the model learns from it) and a **test set** (kept hidden to evaluate the model at the very end).

**Why:** A model can simply *memorize* the training data and look perfect on it. To know whether it works on **new, unseen data**, you must test it on data it has never seen. The test set simulates the future.

**How:**

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% for the test set
    random_state=42,     # makes the split reproducible
    stratify=y           # classification: keep class proportions in both sets
)
```

**Key arguments:**

- `test_size`: fraction for the test (typical 0.2 or 0.25).
- `random_state`: any fixed integer gives the same split each run (reproducibility).
- `stratify=y`: **for classification**. If 10% of your data is positive, both train and test will have about 10% positives. Without it, the test set might get by chance almost no positives.
- For **regression** you do not stratify (the target is continuous).

**Golden rule:** the test set is used **once**, at the very end, for the final score. If you tune your model while looking at the test score, the test set stops being "unseen" and your final score becomes too optimistic.

---

### 1.6 Data leakage

**What:** Data leakage happens when information from outside the training data (usually from the test set or from the future) **sneaks into the training process**.

**Why it matters:** The model looks excellent during development but fails in the real world. It is one of the most common and most dangerous mistakes in ML.

**Common leakage examples:**

1. **Scaling before splitting.** If you compute the mean and standard deviation on the *whole* dataset, the training process "knows" something about the test data.
2. **SMOTE before splitting / before cross-validation.** Synthetic points get created from neighbors that end up in validation folds.
3. **A feature that is the answer in disguise.** Example: predicting "customer left" using the feature "cancellation date". The roadmap warns: *"a feature that is too perfect can reveal leakage"*.
4. **Choosing hyperparameters using the test set.**

**How to avoid it:** always `split first`, then do **everything else** (imputation, scaling, encoding, SMOTE, training) **using only the training data**, ideally inside a `Pipeline` (Section 1.8).

---

### 1.7 Feature scaling

**What:** Putting numeric features on a comparable scale.

**Why:** Some models compute **distances** (KNN, SVM) or use **gradient descent** (logistic regression, linear models with regularization). If one feature ranges from 0 to 100,000 (salary) and another from 0 to 1 (a ratio), the large one dominates the computation. Tree-based models only compare a feature to a threshold, so they **do not need** scaling.

**How:**

| Scaler | Formula | Result |
|---|---|---|
| `StandardScaler` | `z = (x - mean) / std` | Mean 0, std 1 (the most common choice) |
| `MinMaxScaler` | `x' = (x - min) / (max - min)` | Values between 0 and 1 |

**Example:** ages `[20, 30, 40]` → mean = 30, std ≈ 8.16 → scaled `[-1.22, 0, 1.22]`.

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)  # learn mean/std on TRAIN only
X_test_s  = scaler.transform(X_test)       # apply the SAME mean/std to test
```

> `fit` = learn the statistics. `transform` = apply them. **Never `fit` on test data.**

**Scaling table from the roadmap:**

| Needs scaling | Does not need scaling |
|---|---|
| Logistic regression, KNN (crucial), SVM, Ridge/Lasso, KMeans (mandatory), PCA | Decision tree, Random Forest, Gradient Boosting / XGBoost |

---

### 1.8 Pipeline

**What:** A `Pipeline` chains steps (scaler → model) into one object.

**Why:** It guarantees that, during cross-validation, each step is fitted **only on the training part of each fold**. This is the cleanest protection against leakage. It also makes the code shorter and the model easy to save.

**How:**

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  LogisticRegression(max_iter=1000))
])
pipe.fit(X_train, y_train)          # scaler.fit_transform then model.fit
pipe.predict(X_test)                # scaler.transform then model.predict
```

Each step has a **name** (`"scaler"`, `"model"`). Names matter for hyperparameter search: `model__C` means "parameter `C` of the step named `model`" (double underscore).

---

### 1.9 Cross-validation (CV)

**What:** Instead of one train/validation split, you split the training data into **k folds** (e.g., 5), train on 4 folds and validate on the 5th, and rotate until each fold has been the validation set once. The final score is the **average** (plus the standard deviation).

**Why:** A single split can be lucky or unlucky. CV gives a **more reliable and stable** estimate, and the standard deviation tells you how much the score varies.

**How (5-fold):**

```
Fold 1: [VAL][TR ][TR ][TR ][TR ]  → score 0.91
Fold 2: [TR ][VAL][TR ][TR ][TR ]  → score 0.89
Fold 3: [TR ][TR ][VAL][TR ][TR ]  → score 0.93
Fold 4: [TR ][TR ][TR ][VAL][TR ]  → score 0.90
Fold 5: [TR ][TR ][TR ][TR ][VAL]  → score 0.92
                         mean = 0.91, std ≈ 0.014
```

**Variants:**

- **`StratifiedKFold`**: each fold keeps the class proportions. **Use it for classification** (the roadmap says "stratified cross-validation").
- **`KFold`**: plain folds. **Use it for regression** (you cannot stratify a continuous target).

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro")
print(f"{scores.mean():.3f} +/- {scores.std():.3f}")
```

---

### 1.10 Baseline

**What:** A deliberately dumb model used as a reference (e.g., always predict the most frequent class, or always predict the mean).

**Why:** If your fancy model gets 92% but the baseline already gets 90%, your model is barely useful. A score has no meaning without a reference.

**How:**

```python
from sklearn.dummy import DummyClassifier, DummyRegressor
baseline = DummyClassifier(strategy="most_frequent")   # classification
baseline_reg = DummyRegressor(strategy="mean")         # regression
```

---

### 1.11 Overfitting, underfitting, and the bias/variance trade-off

| | **Underfitting** | **Good fit** | **Overfitting** |
|---|---|---|---|
| Model complexity | Too simple | Right | Too complex |
| Train score | Bad | Good | Excellent |
| Test score | Bad | Good | Bad |
| Analogy | Student who barely studied | Student who understood the lesson | Student who memorized the exact exercises but fails new ones |
| Error type | High **bias** | Balanced | High **variance** |

- **Bias** = error from wrong/too-simple assumptions.
- **Variance** = sensitivity to the particular training data (the model changes a lot if data changes slightly).
- **Trade-off:** making a model more complex reduces bias but increases variance. The goal is the sweet spot.

**Quick diagnosis:**

- Train 0.99, validation 0.70 → **overfitting** → simplify the model, regularize, get more data.
- Train 0.65, validation 0.64 → **underfitting** → more complex model, better features.

---

### 1.12 The standard workflow (memorize this)

```
1. Look at the data (EDA), define the target
2. Split: train / test  (stratify for classification)
3. Build a Pipeline: preprocessing -> (SMOTE if needed) -> model
4. Baseline + several models, compared with cross-validation on TRAIN
5. Tune hyperparameters of the best candidates (GridSearchCV / RandomizedSearchCV)
6. ONE final evaluation on the TEST set
7. Interpret with charts, save the model (joblib)
```

---

## 2. Phase 1: Classification

**Goal:** compare several models rigorously, measure their performance correctly, handle imbalance, and tune hyperparameters.

### 2.1 The classification models

Summary table from the roadmap:

| Model | Idea | Hyperparameters to tune | Scaling? |
|---|---|---|---|
| Logistic Regression | Linear boundary, probabilities via sigmoid / softmax | `C`, `penalty`, `class_weight` | Yes |
| KNN | Vote of the k nearest neighbors | `n_neighbors`, `weights`, `metric` | Yes (crucial) |
| SVM (SVC) | Maximum-margin hyperplane, kernels | `C`, `kernel`, `gamma` | Yes |
| Decision Tree | if / else rules, very interpretable | `max_depth`, `min_samples_leaf` | No |
| Random Forest | Average of many trees (bagging) | `n_estimators`, `max_depth`, `max_features` | No |
| Gradient Boosting / XGBoost | Successive trees correcting errors | `learning_rate`, `n_estimators`, `max_depth` | No |

Below, each model is explained in depth.

---

#### 2.1.1 Logistic Regression

**What:** Despite the name, it is a **classification** model. It draws a **linear boundary** between classes and outputs a **probability** that a sample belongs to a class.

**Why use it:** It is fast, simple, interpretable (each coefficient shows a feature's influence), and a very strong first model to try after a baseline.

**How it works, step by step:**

1. Compute a linear score: `z = w1*x1 + w2*x2 + ... + b`
2. Convert the score to a probability with the **sigmoid** function:

   `p = 1 / (1 + e^(-z))`

   The sigmoid squashes any number into the range (0, 1). `z = 0` → `p = 0.5`; large positive `z` → `p` near 1; large negative `z` → `p` near 0.
3. Decide the class using a **threshold** (default 0.5): `p >= 0.5` → class 1.
4. **Training:** find `w` and `b` that minimize the **log-loss** (cross-entropy) using gradient descent (Section 3.4):

   `loss = -[ y*log(p) + (1-y)*log(1-p) ]`

   This loss is large when the model is confidently wrong and small when it is confidently right.

**Multiclass:** instead of sigmoid, **softmax** gives one probability per class, and all probabilities sum to 1:

`p_k = e^(z_k) / sum_j e^(z_j)`

**Small example (pass an exam from hours studied):** suppose training found `z = -4 + 1.0 * hours`.

| Hours | z | p = sigmoid(z) | Prediction |
|---|---|---|---|
| 2 | -2 | 0.12 | Fail |
| 4 | 0 | 0.50 | Borderline |
| 5 | 1 | 0.73 | Pass |
| 8 | 4 | 0.98 | Pass |

Softmax example: scores `[2, 1, 0]` → `e^` = `[7.39, 2.72, 1.00]` → sum 11.11 → probabilities `[0.665, 0.245, 0.090]`.

**Hyperparameters:**

| Hyperparameter | Meaning | Effect |
|---|---|---|
| `C` | **Inverse** of regularization strength (default 1.0) | **Small C** = strong regularization = simpler model (risk of underfitting). **Large C** = weak regularization = complex model (risk of overfitting). Search on a log scale: `0.001, 0.01, 0.1, 1, 10, 100` |
| `penalty` | Type of regularization: `'l2'` (default, shrinks weights), `'l1'` (sets some weights to exactly 0 = feature selection), `'elasticnet'` (mix), `None` | Controls how weights are penalized |
| `class_weight` | `None` or `'balanced'` (or a dict) | `'balanced'` gives rare classes a larger weight in the loss: helps with imbalance |
| `solver` | Optimization algorithm: `'lbfgs'` (default), `'liblinear'`, `'saga'` ... | **Must be compatible with the penalty**: `l1` needs `liblinear` or `saga`; `lbfgs` supports only `l2`/`None` |
| `max_iter` | Max iterations (default 100) | Increase (e.g., 1000) if you see a `ConvergenceWarning` |

> Note: newer scikit-learn versions are changing how `penalty` is specified. If you get a deprecation warning, check the current documentation of `LogisticRegression`.

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(C=1.0, penalty="l2", class_weight="balanced", max_iter=1000)
```

**Strengths:** fast, interpretable, gives probabilities. **Weaknesses:** only linear boundaries (unless you add feature engineering); needs scaling.

---

#### 2.1.2 K-Nearest Neighbors (KNN)

**What:** To classify a new point, look at the **k closest training points** and take a **majority vote**.

**Why use it:** It is the simplest model to understand, makes no assumption about data shape, and works well on small datasets with meaningful distances.

**How it works:**

1. Choose `k` (e.g., 3).
2. Compute the **distance** from the new point to **every** training point (default: Euclidean, `sqrt(sum((a-b)^2))`).
3. Take the k nearest points.
4. Predict the most common class among them.

KNN has **no real training**: it just stores the data ("lazy learner") and does all the work at prediction time, which makes prediction slow on large datasets.

**Example:** neighbors of a new point: 2 red, 1 blue (k = 3) → prediction **red**.

**Why scaling is crucial (numeric example):**

| Person | Age | Salary |
|---|---|---|
| A (query) | 30 | 3000 |
| B | 60 | 3100 |
| C | 31 | 5000 |

Unscaled: distance(A,B) = sqrt(30² + 100²) ≈ **104**; distance(A,C) = sqrt(1² + 2000²) ≈ **2000**. KNN says "B is much closer to A than C", even though B is 30 years older. The salary completely drowns the age. After `StandardScaler`, both features count fairly.

**Hyperparameters:**

| Hyperparameter | Meaning | Effect |
|---|---|---|
| `n_neighbors` (k) | Number of neighbors (default 5) | **Small k** (1) = very flexible, noisy, overfits. **Large k** = smooth, may underfit. Prefer odd values in binary problems to avoid ties |
| `weights` | `'uniform'` (all votes equal) or `'distance'` (closer neighbors count more) | `'distance'` is often better when classes overlap |
| `metric` | Distance type: `'minkowski'` (default), `'euclidean'`, `'manhattan'`, ... | Manhattan can be more robust in high dimensions |
| `p` | Power of Minkowski: 1 = Manhattan, 2 = Euclidean | Used with `metric='minkowski'` |
| `algorithm` | `'auto'`, `'kd_tree'`, `'ball_tree'`, `'brute'` | Speed only, not results |

**Strengths:** simple, no training. **Weaknesses:** slow predictions on big data, sensitive to scaling and irrelevant features, struggles in very high dimensions ("curse of dimensionality").

---

#### 2.1.3 Support Vector Machine (SVM / SVC)

**What:** SVM finds the **hyperplane (boundary) that separates the classes with the largest possible margin**. The margin is the empty space between the boundary and the closest points of each class. Those closest points are the **support vectors**: only they define the boundary.

**Why use it:** Very effective on medium-sized datasets and in high dimensions. With **kernels** it can learn non-linear boundaries.

**How it works:**

1. Among all lines that separate the classes, choose the one **farthest from both classes** (largest margin). A wider margin usually generalizes better.
2. Real data is rarely perfectly separable, so SVM allows some mistakes (**soft margin**). The hyperparameter `C` controls the compromise between "wide margin" and "few training errors".
3. For **non-linear** data, the **kernel trick** implicitly maps points to a higher-dimensional space where they become separable, without computing that space explicitly.

**Kernel intuition (1D example):** points on a line: blue at `-3, -2`, red at `-1, 0, 1`, blue at `2, 3`. No single threshold separates red from blue. Add a second feature `x²`: red points have `x² ≤ 1`, blue points have `x² ≥ 4`. Now a horizontal line (`x² = 2.5`) separates them perfectly. A kernel does this kind of mapping automatically.

**Hyperparameters:**

| Hyperparameter | Meaning | Effect |
|---|---|---|
| `C` | Penalty for misclassified points (default 1.0) | **Large C** = tries hard to classify every training point = narrow margin = overfitting risk. **Small C** = wider margin, tolerates errors = smoother |
| `kernel` | `'linear'`, `'rbf'` (default), `'poly'`, `'sigmoid'` | `linear` for roughly linear data/high dimensions; `rbf` is the best general-purpose choice |
| `gamma` | Reach of a single point's influence (for `rbf`, `poly`, `sigmoid`). `'scale'` (default), `'auto'`, or a number | **Large gamma** = each point influences only its close neighborhood = wiggly boundary = overfitting. **Small gamma** = smooth boundary = underfitting |
| `degree` | Degree of the polynomial (only `kernel='poly'`) | Higher = more complex |
| `probability` | `True` to enable `predict_proba` | Slower (internal CV); needed for ROC-AUC via probabilities (or use `decision_function`) |
| `class_weight` | `'balanced'` for imbalance | Same idea as in logistic regression |

RBF kernel formula: `K(a, b) = exp(-gamma * ||a - b||²)`: similarity is 1 for identical points and decays toward 0 with distance.

**Strengths:** powerful, effective in high dimensions. **Weaknesses:** slow on large datasets (> ~50k rows), needs scaling, no direct probabilities, hyperparameters (`C`, `gamma`) must be tuned together.

---

#### 2.1.4 Decision Tree

**What:** A flowchart of **if / else questions** on features that ends in a prediction at the leaves.

**Why use it:** It is the most **interpretable** model: you can read and even draw the decision logic. It handles numeric and categorical-encoded features, needs **no scaling**, and captures non-linear relationships.

**How it works:**

1. At each node, the tree tests many possible splits (`feature <= threshold`) and picks the one that makes the resulting groups **as pure as possible**.
2. **Purity** is measured by **Gini impurity**: `Gini = 1 - sum(p_k²)`, where `p_k` is the fraction of class k in the node. (Alternative: **entropy** `-sum(p_k * log2(p_k))`.) Gini = 0 means a perfectly pure node; for 2 classes, the worst value is 0.5.
3. Repeat recursively on each child until a stop condition (max depth, minimum samples, pure node).
4. A leaf predicts the majority class of its training samples.

**Numeric example:** a node has 10 samples: 5 "yes" and 5 "no". `Gini = 1 - (0.5² + 0.5²) = 0.5`.
A split on `age > 30` creates a left node (4 yes, 1 no) and a right node (1 yes, 4 no).
Each has `Gini = 1 - (0.8² + 0.2²) = 0.32`. Weighted Gini after the split = 0.32. **Gain = 0.5 - 0.32 = 0.18**. The tree picks the split with the largest gain.

**Danger:** a fully grown tree memorizes the training data (each leaf = one sample) → **overfitting**. Hyperparameters exist to limit this.

**Hyperparameters:**

| Hyperparameter | Meaning | Effect |
|---|---|---|
| `max_depth` | Maximum depth of the tree (default None = unlimited) | The **most important** one. Small = simple (underfit). None/large = overfit. Typical values 3 to 15 |
| `min_samples_leaf` | Minimum samples required in a leaf | Larger = smoother, more general tree |
| `min_samples_split` | Minimum samples needed to split a node | Larger = fewer splits |
| `criterion` | `'gini'` (default), `'entropy'`, `'log_loss'` | Rarely changes results much |
| `max_features` | Number of features considered per split | Adds randomness; used heavily in forests |
| `class_weight` | `'balanced'` | Handles imbalance |
| `ccp_alpha` | Cost-complexity pruning strength | Larger = more pruning = simpler tree |

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=42).fit(X_train, y_train)
```

**Strengths:** interpretable, no scaling, non-linear. **Weaknesses:** overfits easily, unstable (small data change → different tree).

---

#### 2.1.5 Random Forest

**What:** An **ensemble** of many decision trees whose predictions are **combined** (vote for classification, average of probabilities).

**Why use it:** A single tree has high variance. Averaging many *different* trees cancels out their individual errors, giving a much more stable and accurate model with little tuning. It is often an excellent default.

**How it works (bagging + feature randomness):**

1. **Bootstrap:** for each tree, draw a random sample of rows **with replacement** from the training set (same size as the original). About 63% of the distinct rows appear; the rest (~37%) are **out-of-bag (OOB)** and can be used as a free validation set.
2. **Random features:** at each split, the tree may only consider a random subset of features (`max_features`, default `sqrt(n_features)` for classification). This makes the trees **different from each other** (decorrelated).
3. Train each tree **independently** (can be parallel).
4. **Predict:** every tree votes; the majority wins.

**Analogy:** asking 200 slightly different experts and taking the majority opinion, instead of trusting one expert.

**Hyperparameters:**

| Hyperparameter | Meaning | Effect |
|---|---|---|
| `n_estimators` | Number of trees (default 100) | More trees = more stable, slower. Gains flatten after a few hundred; it will **not** overfit by adding trees |
| `max_depth` | Depth of each tree | Limits complexity |
| `max_features` | Features considered per split: `'sqrt'`, `'log2'`, an int, or a fraction | Smaller = more diverse trees, less correlation; too small = weak trees |
| `min_samples_leaf` / `min_samples_split` | Same as in a tree | Regularization |
| `bootstrap` | Use bootstrap samples (default True) | False = each tree sees all the data |
| `oob_score` | `True` to compute the out-of-bag score | Free validation estimate |
| `class_weight` | `'balanced'` or `'balanced_subsample'` | Imbalance |
| `n_jobs` | `-1` = use all CPU cores | Speed |
| `random_state` | Seed | Reproducibility |

**Feature importance:** `rf.feature_importances_` shows which features reduced impurity the most (see Section 5).

**Strengths:** robust, accurate, little tuning, no scaling. **Weaknesses:** less interpretable than one tree, larger/slower model, can be biased toward high-cardinality features.

---

#### 2.1.6 Gradient Boosting / XGBoost

**What:** Another tree ensemble, but trees are built **one after the other**, each new tree focusing on **correcting the errors of the previous ones**.

**Why use it:** On tabular data, boosting is often the **most accurate** family of models (XGBoost, LightGBM, CatBoost win many competitions).

**How it works:**

1. Start with a simple prediction (e.g., the average of the target; for classification, the log-odds of the positive class).
2. Compute the **errors** (residuals, or more generally the negative gradient of the loss).
3. Train a **small tree** (shallow, "weak learner") to predict those errors.
4. Add it to the model, but only **a fraction** of it: `prediction_new = prediction_old + learning_rate * tree_output`.
5. Repeat for `n_estimators` rounds. Each round reduces the remaining error (a gradient descent in the space of functions, linking to Section 3.4).

**Numeric example (regression):** house prices (in k$): actual `[100, 200, 300]`.

- Round 0: predict the mean = 200 for all → residuals `[-100, 0, +100]`.
- Round 1: a small tree learns residuals (suppose perfectly). With `learning_rate = 0.1`, predictions become `[190, 200, 210]` → new residuals `[-90, 0, +90]`.
- Round 2: new tree on `[-90, 0, 90]` → predictions `[181, 200, 219]` ...

After many rounds, predictions approach `[100, 200, 300]`. A small learning rate = small careful steps (needs more trees, but generalizes better).

**Hyperparameters (XGBoost names; scikit-learn's `GradientBoostingClassifier` has similar ones):**

| Hyperparameter | Meaning | Effect |
|---|---|---|
| `learning_rate` (eta) | Shrinkage of each tree's contribution (typical 0.01 to 0.3) | Small = slow, robust, needs more trees. **Trade-off with `n_estimators`**: lower the first, raise the second |
| `n_estimators` | Number of boosting rounds | Too many = overfitting (unlike Random Forest). Use **early stopping** on a validation set |
| `max_depth` | Depth of each tree (typical 3 to 8) | Boosting uses **shallow** trees. Deeper = more complex |
| `subsample` | Fraction of rows used per tree (e.g., 0.8) | < 1 adds randomness, reduces overfitting |
| `colsample_bytree` | Fraction of features per tree | Same idea for features |
| `min_child_weight` | Minimum "weight" in a leaf | Larger = more conservative |
| `gamma` (XGBoost) | Minimum loss reduction required to split | Larger = more conservative |
| `reg_lambda` / `reg_alpha` | L2 / L1 regularization of leaf weights | Reduce overfitting |
| `scale_pos_weight` | Ratio negative/positive for imbalance (binary) | XGBoost's version of class weighting |
| `early_stopping_rounds` | Stop if validation score does not improve for N rounds | Automatically finds a good number of trees |

```python
from xgboost import XGBClassifier
xgb = XGBClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=4,
    subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42
)
```

**Strengths:** top accuracy on tabular data, handles complex interactions. **Weaknesses:** more hyperparameters, sensitive to overfitting if poorly tuned, trees are sequential (harder to parallelize than a forest), less interpretable.

---

#### 2.1.7 Quick comparison: which model when?

| Situation | Good first choice |
|---|---|
| Need an interpretable, fast baseline | Logistic Regression |
| Very small data, simple geometry | KNN |
| Medium data, many features, possibly non-linear | SVM (RBF) |
| Need explainable rules | Decision Tree (shallow) |
| Strong, low-effort default | Random Forest |
| Maximum accuracy on tabular data | Gradient Boosting / XGBoost |

---

### 2.2 Classification metrics

**Why metrics matter:** "How good is the model?" has no single answer. Different mistakes have different costs. Choosing the wrong metric can make a useless model look great.

#### Running example (use it to understand every formula)

A spam filter tested on **1000 emails**, of which **100 are spam** (positive class) and 900 are normal. The result:

| | **Predicted spam** | **Predicted normal** |
|---|---|---|
| **Actually spam** | **TP = 80** | **FN = 20** |
| **Actually normal** | **FP = 30** | **TN = 870** |

- **TP (True Positive):** spam correctly flagged.
- **TN (True Negative):** normal correctly left alone.
- **FP (False Positive):** normal email wrongly flagged as spam (**Type I error**).
- **FN (False Negative):** spam that slipped through (**Type II error**).

#### Confusion matrix

**What:** the table above. **Why:** it shows **where** the model makes mistakes, not only how many. **Always produce it** (roadmap rule).

```python
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test, cmap="Blues")
```

For multiclass, it is an N×N table: the diagonal is correct predictions; off-diagonal cells show which classes get confused with which.

#### Accuracy

- **Formula:** `(TP + TN) / total` = (80 + 870) / 1000 = **0.95**
- **Meaning:** the fraction of all predictions that are correct.
- **Use only when classes are balanced.**
- **Trap:** a useless model that always says "normal" scores **900/1000 = 0.90 accuracy** while catching **0 spam**. This is why accuracy is dangerous with imbalance.

#### Precision

- **Formula:** `TP / (TP + FP)` = 80 / (80 + 30) = **0.727**
- **Meaning:** "Of everything I **flagged** as positive, how much was truly positive?"
- **Prefer when false positives are costly.** Example: spam filter. Sending a real email (a job offer) to the spam folder is bad.

#### Recall (sensitivity, TPR)

- **Formula:** `TP / (TP + FN)` = 80 / (80 + 20) = **0.80**
- **Meaning:** "Of all the **real** positives, how many did I catch?"
- **Prefer when false negatives are costly.** Examples: cancer screening (missing a sick patient), fraud detection.

**Precision vs recall trade-off:** lowering the decision threshold flags more cases → recall goes up, precision goes down. Raising it does the opposite. You cannot maximize both.

#### F1-score

- **Formula:** `2 * P * R / (P + R)` = 2 × 0.727 × 0.80 / (0.727 + 0.80) = **0.762**
- **Meaning:** the **harmonic mean** of precision and recall. It is dragged down by whichever of the two is low (unlike the arithmetic mean). A model with P = 1.0 and R = 0.01 has an arithmetic mean of 0.505 but F1 ≈ 0.02.
- **Use for:** an overall compromise, and for imbalanced classes.

#### ROC curve and ROC-AUC

- **What:** the ROC curve plots **TPR (= recall)** on the y-axis against **FPR = FP / (FP + TN)** on the x-axis, for **every possible threshold**.
- **AUC** = area under that curve:
  - **1.0** = perfect, **0.5** = random guessing (the diagonal line), **< 0.5** = worse than random (labels probably inverted).
  - Interpretation: AUC is the probability that a randomly chosen positive gets a higher score than a randomly chosen negative.
- **Why:** it is **independent of the threshold**, so it is great to compare models globally.
- **Limit:** with very high imbalance, ROC-AUC can look optimistic because FPR is divided by a huge number of negatives.

#### PR curve and PR-AUC

- **What:** plots **Precision vs Recall** for every threshold. PR-AUC is the area (often computed as *average precision*).
- **Why:** it ignores true negatives, so it focuses on the rare positive class. **Prefer it with strong imbalance** (fraud: 0.1% positives). The random baseline is not 0.5 but the positive rate (e.g., 0.001).

#### Classification report and averaging for multiclass

```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```

It prints precision, recall, F1 and **support** (number of real samples) per class. For multiclass, you must **average** across classes:

| Average | How | When |
|---|---|---|
| `macro` | Plain mean of per-class scores; every class counts equally | Rare classes matter (used in the roadmap: `f1_macro`) |
| `weighted` | Mean weighted by class size | Reflect the dataset distribution |
| `micro` | Pool all TP/FP/FN first | Overall performance (equals accuracy for single-label multiclass) |

#### Metric chooser

| Business situation | Metric |
|---|---|
| Balanced classes | Accuracy (plus F1) |
| False alarms are costly | Precision |
| Missing positives is costly | Recall |
| Imbalanced and need a compromise | F1 / `f1_macro` |
| Compare models regardless of threshold | ROC-AUC |
| Strong imbalance | PR-AUC |
| Always | Confusion matrix |

---

### 2.3 Class imbalance and SMOTE

**What is imbalance:** one class is much rarer than the others (e.g., 2% fraud vs 98% normal).

**Why it is a problem:** the model can reach 98% accuracy by ignoring the rare class. It sees so few positive examples that it barely learns them, yet the positive class is usually the one we care about.

#### SMOTE: Synthetic Minority Over-sampling Technique

**What:** Instead of simply duplicating minority samples, SMOTE **creates new synthetic minority samples** by **interpolating** between a minority point and one of its nearest minority neighbors.

**How it works:**

1. Pick a minority sample `x`.
2. Find its `k` nearest minority neighbors (`k_neighbors=5` by default).
3. Randomly choose one neighbor `x_nn`.
4. Create a new point on the **line segment** between them: `x_new = x + λ * (x_nn - x)`, with `λ` random in [0, 1].

**Example:** `x = (1, 1)`, `x_nn = (3, 2)`, `λ = 0.5` → `x_new = (1,1) + 0.5 * (2, 1) = (2, 1.5)`.

**Key parameters (`imblearn.over_sampling.SMOTE`):**

| Parameter | Meaning |
|---|---|
| `sampling_strategy` | Target ratio (`'auto'` = balance to the majority class; or a float / dict) |
| `k_neighbors` | Number of neighbors used for interpolation (default 5); must be smaller than the number of minority samples |
| `random_state` | Reproducibility |

(For datasets with categorical columns use `SMOTENC`; plain SMOTE works on numeric data only.)

#### The rules to remember (from the roadmap) and the reasons

1. **SMOTE only on the train set, never on the test set.**
   *Why:* the test set must reflect reality. If you add synthetic points to it, you evaluate on fake data and your score is meaningless.
2. **Use `imblearn.pipeline.Pipeline` (not scikit-learn's).**
   *Why:* the imblearn pipeline applies SMOTE **only during `fit`**, and **inside each CV fold on the training part only**. The validation fold stays untouched. If you apply SMOTE once before cross-validation, synthetic points built from neighbors of validation samples leak into training = **data leakage**, and your CV scores are inflated.
3. **SMOTE comes after encoding and scaling.**
   *Why:* it works by computing distances and interpolating, so it needs **numeric** features on a **comparable scale**.
4. **Alternatives:**
   - `class_weight='balanced'`: no new data; the loss weights rare classes more. Often the simplest first try.
   - **Under-sampling:** remove majority samples (fast, but throws information away).
   - **Threshold tuning:** keep the model, but lower the probability threshold for the positive class to increase recall.
5. **Evaluate with F1 / recall / PR-AUC, not accuracy.**

```python
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

pipe = ImbPipeline([
    ("scaler", StandardScaler()),
    ("smote",  SMOTE(random_state=42)),
    ("model",  RandomForestClassifier(random_state=42))
])
# cross_val_score / GridSearchCV on this pipe apply SMOTE correctly inside each fold
```

---

### 2.4 GridSearchCV and RandomizedSearchCV

**What:** Automatic **hyperparameter search**. For each candidate combination, it trains the model with cross-validation and records the score, then keeps the best.

**Why:** Hyperparameters strongly affect performance and the best values are not known in advance. Searching manually is slow and biased.

#### GridSearchCV

Tests **every combination** in a grid. Exhaustive but expensive.

**Cost example:** grid with `C` (3 values) × `kernel` (2) × `gamma` (2) = **12 combinations**. With 5-fold CV = **60 model fits**. Adding one more hyperparameter with 5 values → 300 fits. The cost explodes quickly.

#### RandomizedSearchCV

Draws `n_iter` **random** combinations from lists or **distributions**. Much cheaper when the space is large, and often finds almost as good a result because usually only a few hyperparameters really matter.

#### Important points from the roadmap

- **Both use cross-validation.**
- **Inside a pipeline, parameter names use `step__parameter`** (double underscore): e.g., `"model__C"`, `"smote__k_neighbors"`.
- **`scoring`** must match your goal (e.g., `"f1_macro"`, `"recall"`, `"average_precision"`). The default (accuracy) is a bad choice with imbalance.
- **`refit=True`** (default): after the search, the best combination is **re-trained on the entire training set** and stored in `best_estimator_`.
- **After the search: use the test set exactly once** for the final score.

**Key arguments:**

| Argument | Meaning |
|---|---|
| `estimator` | Your model or pipeline |
| `param_grid` / `param_distributions` | Values to try (dict) |
| `cv` | Number of folds or a splitter object (use `StratifiedKFold`) |
| `scoring` | Metric to maximize |
| `n_iter` | (Randomized only) Number of random combinations |
| `n_jobs` | `-1` = parallelize |
| `refit` | Retrain best model on the full training set |
| `random_state` | (Randomized) reproducibility |

**Useful attributes:** `best_params_`, `best_score_` (mean CV score), `best_estimator_`, `cv_results_` (full table).

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import loguniform

pipe = Pipeline([("scaler", StandardScaler()), ("model", SVC())])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 1) Grid search
grid = GridSearchCV(
    pipe,
    param_grid={"model__C": [0.1, 1, 10],
                "model__kernel": ["linear", "rbf"],
                "model__gamma": ["scale", 0.01]},
    scoring="f1_macro", cv=cv, n_jobs=-1, refit=True
)
grid.fit(X_train, y_train)
print(grid.best_params_, grid.best_score_)

# 2) Randomized search (continuous distributions on a log scale)
rand = RandomizedSearchCV(
    pipe,
    param_distributions={"model__C": loguniform(1e-2, 1e2),
                         "model__gamma": loguniform(1e-4, 1e0)},
    n_iter=30, scoring="f1_macro", cv=cv, n_jobs=-1, random_state=42
)
rand.fit(X_train, y_train)

# 3) ONE final test
best = rand.best_estimator_
print("Final test score:", best.score(X_test, y_test))
```

**Common mistake:** `best_score_` is the CV score on the training data (slightly optimistic because you picked the max). The **test score** is the honest estimate.

**Saving the best model:**

```python
import joblib
joblib.dump(best, "models/best.joblib")      # save
model = joblib.load("models/best.joblib")    # load later
```

---

### 2.5 Phase 1 checklist and full code

**Checklist:**

- [ ] Baseline + 3 models compared with **stratified cross-validation**.
- [ ] Classification report, confusion matrix and ROC produced **and interpreted**.
- [ ] SMOTE in an `ImbPipeline`, with an explanation of why **only on train**.
- [ ] GridSearchCV and RandomizedSearchCV run, then **a single final test**.
- [ ] Best model saved: `joblib.dump(best, 'models/best.joblib')`.

**Full working example (copy and run):**

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, RocCurveDisplay
import matplotlib.pyplot as plt

# 1. Data + split (stratified)
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# 2. Models (scaled ones inside a Pipeline)
models = {
    "Baseline":  DummyClassifier(strategy="most_frequent"),
    "LogReg":    Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=1000))]),
    "KNN":       Pipeline([("sc", StandardScaler()), ("m", KNeighborsClassifier(n_neighbors=5))]),
    "SVM":       Pipeline([("sc", StandardScaler()), ("m", SVC(probability=True))]),
    "Tree":      DecisionTreeClassifier(max_depth=4, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "GradBoost": GradientBoostingClassifier(random_state=42),
}

# 3. Compare with stratified CV on TRAIN only
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, m in models.items():
    s = cross_val_score(m, X_train, y_train, cv=cv, scoring="f1_macro")
    print(f"{name:13s} F1-macro = {s.mean():.3f} +/- {s.std():.3f}")

# 4. Final evaluation of the chosen model on TEST (once)
best = models["RandomForest"].fit(X_train, y_train)
print(classification_report(y_test, best.predict(X_test)))
ConfusionMatrixDisplay.from_estimator(best, X_test, y_test, cmap="Blues")
RocCurveDisplay.from_estimator(best, X_test, y_test)
plt.show()
```

---

## 3. Phase 2: Regression and optimization

**Goal:** predict a **continuous value**, measure the error with MSE and friends, and understand the **optimization algorithm** hiding behind almost every model.

### 3.1 Regression models

#### 3.1.1 Linear Regression

**What:** Models the target as a **weighted sum of the features**: `ŷ = w1*x1 + w2*x2 + ... + b`. With one feature it is the best-fitting straight line.

**Why use it:** The simplest, fastest and most interpretable regression model. Each weight says "if this feature increases by 1 unit, the prediction changes by `w` units, other features fixed".

**How it works:** it finds `w` and `b` that minimize the **sum of squared errors** (Ordinary Least Squares = minimizing the MSE). This can be solved directly by a formula (the *normal equation*) or iteratively with **gradient descent** (Section 3.4).

**Numeric example:** house surface `x` (in 100 m²) and price `y` (in 100k$): points `(1, 2), (2, 4), (3, 5)`.

- mean(x) = 2, mean(y) = 3.667
- `w = Σ(x - x̄)(y - ȳ) / Σ(x - x̄)² = 3.0 / 2.0 = 1.5`
- `b = ȳ - w * x̄ = 3.667 - 1.5 * 2 = 0.667`
- Model: `ŷ = 1.5x + 0.667`. For `x = 4` → `ŷ = 6.67`.
- Residuals (y − ŷ): `-0.167, +0.333, -0.167` (they sum to zero).

**Hyperparameters:** almost none (`fit_intercept`, `positive`). That is its limit: it cannot control complexity, which is why **Ridge** and **Lasso** exist. It is sensitive to outliers and correlated features (multicollinearity).

#### 3.1.2 Ridge Regression (L2 penalty)

**What:** Linear regression + a penalty on the **squared size of the weights**.

`Loss = MSE + alpha * Σ w²`

**Why:** When there are many features, correlated features, or little data, plain linear regression gives huge unstable weights and overfits. The penalty forces weights to stay **small**.

**How:** The optimizer now faces a trade-off: fit the data **and** keep weights small. A bigger `alpha` means a stronger penalty. Ridge **shrinks weights toward zero but never exactly to zero**: all features stay in the model.

**Example:** two nearly identical features. Plain regression may give `w1 = +500, w2 = -498` (huge, canceling). Ridge gives something like `w1 = 1.1, w2 = 0.9`: stable and sensible.

#### 3.1.3 Lasso Regression (L1 penalty)

**What:** Linear regression + a penalty on the **absolute size of the weights**.

`Loss = MSE + alpha * Σ |w|`

**Why / How:** The L1 penalty has a special geometry that pushes some weights to **exactly 0**. So Lasso does **automatic feature selection**: useless features are removed. Great when you suspect many features are irrelevant, and for a sparse, interpretable model.

**Ridge vs Lasso:**

| | Ridge (L2) | Lasso (L1) |
|---|---|---|
| Penalty | `Σ w²` | `Σ abs(w)` |
| Weights | Small, non-zero | Some exactly 0 |
| Best when | Many useful correlated features | Many useless features, need selection |

**Shared hyperparameters:**

| Hyperparameter | Meaning | Effect |
|---|---|---|
| `alpha` | Regularization strength (default 1.0) | **alpha = 0** → plain linear regression. **Small** → weak penalty (overfit risk). **Large** → strong penalty (underfit risk; Lasso zeros everything). Search on a log scale: `0.001 … 100` |
| `fit_intercept` | Learn `b` | Usually True |
| `max_iter`, `tol` | Optimizer settings (Lasso) | Raise `max_iter` on convergence warnings |
| `solver` | (Ridge) solver choice | Usually `'auto'` |
| `l1_ratio` | (**ElasticNet** = L1 + L2 mix) 0 = Ridge, 1 = Lasso | Combines both advantages |

> **Scaling is mandatory** for Ridge/Lasso: the penalty treats all weights equally, so features must be on the same scale or it would unfairly penalize features measured in small units.
>
> **Naming warning:** `alpha` here is the **regularization strength**. In gradient descent (Section 3.4), `alpha` is the **learning rate**. Same Greek letter, two different things.

#### 3.1.4 RandomForestRegressor

**What:** The same forest idea as in classification, but each tree predicts a **number** (the mean of the targets in its leaf) and the forest **averages** all the trees.

**Why:** It captures **non-linear relationships and interactions** automatically, needs no scaling and is robust to outliers in features.

**How:** bootstrap samples + random features per split + average of predictions. Splits minimize the **MSE** (`criterion='squared_error'`) instead of Gini.

**Hyperparameters:** the same as the classifier: `n_estimators`, `max_depth`, `max_features` (for regression the default is all features, so tuning it often helps), `min_samples_leaf`, `bootstrap`, `oob_score`, `n_jobs`, `random_state`.

**Limitation:** it **cannot extrapolate**. A tree predicts averages of training values, so it never outputs values beyond the range seen in training (a linear model can).

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor

models = {
    "Linear": Pipeline([("sc", StandardScaler()), ("m", LinearRegression())]),
    "Ridge":  Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0))]),
    "Lasso":  Pipeline([("sc", StandardScaler()), ("m", Lasso(alpha=0.1))]),
    "RF":     RandomForestRegressor(n_estimators=200, random_state=42),
}
```

---

### 3.2 Regression metrics

**Running example:** true values `y = [3, 5, 2, 7]`, predictions `ŷ = [2.5, 5, 4, 8]`.
Errors (ŷ − y): `[-0.5, 0, 2, 1]`.

| Metric | Formula | Calculation | Result | Notes |
|---|---|---|---|---|
| **MSE** | `(1/n) Σ (y − ŷ)²` | (0.25 + 0 + 4 + 1) / 4 | **1.3125** | Squares errors → punishes **big errors** heavily; unit is the **square** of the target unit (e.g., $²). Differentiable, hence the standard training loss |
| **RMSE** | `√MSE` | √1.3125 | **1.146** | Back to the **same unit** as the target: easy to interpret ("on average off by about 1.15") |
| **MAE** | `(1/n) Σ abs(y − ŷ)` | (0.5 + 0 + 2 + 1) / 4 | **0.875** | **Robust to outliers**, treats all errors proportionally |
| **R²** | `1 − SS_res / SS_tot` | see below | **0.644** | Fraction of variance explained |

R² details: `SS_res = Σ(y − ŷ)² = 5.25`; mean(y) = 4.25; `SS_tot = Σ(y − ȳ)² = 14.75`; `R² = 1 − 5.25/14.75 = 0.644`.

- `R² = 1` → perfect. `R² = 0` → no better than always predicting the mean. **`R² < 0`** → worse than predicting the mean (possible on test data).
- **MSE vs MAE:** if one huge error would be a disaster (e.g., engineering safety), prefer MSE/RMSE. If outliers are just noise you don't want to dominate, prefer MAE.
- Always compare to a baseline (`DummyRegressor(strategy="mean")` has R² ≈ 0).

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)
```

In scikit-learn scoring strings, errors are negated (higher = better): `"neg_mean_squared_error"`, `"neg_mean_absolute_error"`, `"r2"`.

---

### 3.3 Overfitting, underfitting and regularization

**Recap in regression terms:** fit polynomials of increasing degree to noisy data from a curve.

| Degree | What happens | Train error | Test error |
|---|---|---|---|
| 1 (straight line) | Too simple, misses the curve | High | High → **underfitting** |
| 3 | Captures the true shape | Low | Low → **good** |
| 15 | Wiggles through every noisy point | ~0 | Very high → **overfitting** |

**Regularization (what / why / how):**

- **What:** adding a penalty on weight size to the loss (Ridge/Lasso).
- **Why:** huge weights are typical of overfitted models; limiting them forces **smoother, simpler** functions.
- **How:** `alpha` controls how much simplicity is rewarded versus fit. Tune it with cross-validation (`RidgeCV`, `LassoCV`, or `GridSearchCV` with `Pipeline`).

**Cross-validation for regression:** use plain `KFold(n_splits=5, shuffle=True, random_state=42)`. **No stratification** because the target is continuous.

---

### 3.4 Gradient descent

#### 3.4.1 The fundamental principle

**What:** An iterative algorithm to **minimize a cost (loss) function** by repeatedly moving the parameters a little in the direction that reduces the cost the fastest.

**Why:** For most models (logistic regression, neural networks, boosting) there is **no closed-form formula** for the best parameters. Gradient descent is the general tool that trains them. The roadmap calls it "the optimization algorithm hiding behind almost every model".

**How:** The **gradient** is the vector of partial derivatives of the cost with respect to each parameter. It points toward the **steepest increase** of the cost, so we move in the **opposite** direction:

```
w = w - alpha * gradient
```

`alpha` is the **learning rate** (step size).

**Analogy:** you are on a foggy mountain and want to reach the valley. You feel the slope under your feet (the gradient) and take a step downhill. The step length is the learning rate.

**Numeric example:** minimize `J(w) = (w − 3)²`. Gradient = `2(w − 3)`. Start `w = 0`.

| alpha | Iteration 1 | Iteration 2 | Iteration 3 | Behavior |
|---|---|---|---|---|
| **0.1** | 0 → 0.60 | → 1.08 | → 1.46 | Smooth, steady convergence toward 3 |
| **0.5** | 0 → 3.0 | (stays) | | Jumps to the minimum in one step (lucky case) |
| **1.1** | 0 → 6.6 | → −1.32 | → 8.18 | **Diverges**: each step overshoots farther away |
| **0.001** | 0 → 0.006 | → 0.012 | | Correct but **painfully slow** |

(Each step multiplies the distance to the minimum by `1 − 2*alpha`; if its absolute value is above 1, it diverges.)

**For linear regression**, with `ŷ = w*x + b` and `J = (1/n) Σ (ŷ − y)²`:

```
dJ/dw = (2/n) Σ (ŷ − y) * x
dJ/db = (2/n) Σ (ŷ − y)
```

**From-scratch implementation with 3 learning rates (roadmap checklist item):**

```python
import numpy as np
import matplotlib.pyplot as plt

# synthetic data: y = 4 + 3x + noise
rng = np.random.default_rng(42)
x = rng.uniform(0, 10, 100)
y = 4 + 3 * x + rng.normal(0, 2, 100)

# standardize x -> faster, more stable convergence
x = (x - x.mean()) / x.std()

def gradient_descent(x, y, alpha, n_iter=100):
    w, b, history = 0.0, 0.0, []
    n = len(x)
    for _ in range(n_iter):
        y_pred = w * x + b
        error = y_pred - y
        history.append(np.mean(error ** 2))          # cost (MSE)
        w -= alpha * (2 / n) * np.sum(error * x)     # update w
        b -= alpha * (2 / n) * np.sum(error)         # update b
    return w, b, history

for alpha in [0.001, 0.1, 0.9]:
    _, _, hist = gradient_descent(x, y, alpha)
    plt.plot(hist, label=f"alpha = {alpha}")

plt.xlabel("Iteration"); plt.ylabel("Cost (MSE)")
plt.yscale("log"); plt.legend(); plt.title("Cost vs iterations")
plt.show()
```

**Reading the cost curve:** it should **decrease then flatten**. Too small alpha: still decreasing slowly after all iterations. Good alpha: fast drop then plateau. Too large alpha: oscillates or **rises** (divergence).

#### 3.4.2 Variants

| Variant | Data used per update | Characteristics |
|---|---|---|
| **Batch** | The **whole** dataset | Exact gradient, stable and smooth, but **slow** for large data |
| **Stochastic (SGD)** | **1** random example | Very fast updates but **noisy** path (zig-zags); noise can even help escape bad spots |
| **Mini-batch** | A small batch (e.g., 32, 64) | **Standard compromise**: efficient on GPU/vectorized code, less noisy than SGD. Used in deep learning |

Vocabulary: one **epoch** = one full pass over the dataset. With 1000 samples and batch size 100, one epoch = 10 updates.

**Scikit-learn tools:** `SGDRegressor` and `SGDClassifier` train linear models with SGD.

| Hyperparameter | Meaning |
|---|---|
| `loss` | Which loss to minimize (`'squared_error'`, `'log_loss'`, `'hinge'`...) |
| `learning_rate` / `eta0` | Learning-rate schedule and initial value |
| `alpha` | **Regularization strength** here (not the learning rate!) |
| `penalty` | `'l2'`, `'l1'`, `'elasticnet'` |
| `max_iter`, `tol` | Max epochs and stopping tolerance |
| `early_stopping` | Stop when the validation score stops improving |

#### 3.4.3 Practical rules

- **Alpha too big** → cost oscillates or diverges. **Alpha too small** → very slow convergence.
- **Standardize the features** → gradient descent converges much faster. *Why:* with unscaled features, the cost surface is a long narrow valley (elliptical contours) and the path zig-zags; with scaled features the contours are rounder and the path goes almost straight to the minimum.
- **Plot cost vs iterations** and interpret it (see above).
- **Link with the rest:** logistic regression, neural networks and boosting are all trained with gradient descent (or a variant). Understanding it once explains many models.

**Checklist Phase 2:**

- [ ] I compare Linear, Ridge, Lasso and RandomForest with **MSE, RMSE, MAE, R²**.
- [ ] I can explain overfitting / underfitting and regularization.
- [ ] I coded a gradient descent and plotted the cost curve with **3 values of alpha**.

---

## 4. Phase 3: Clustering

**Goal:** group data **without labels** and evaluate the quality of the groups.

**What is clustering:** unsupervised learning that puts **similar** samples in the same group and **different** ones in different groups, with no target to guide it.

**Why:** customer segmentation, grouping documents, anomaly detection, exploring unknown data structure.

**Important:** all clustering algorithms based on distance need **scaling** (KMeans: mandatory).

| Algorithm | Principle | Key points |
|---|---|---|
| KMeans | Partition into k groups around centroids | Choose k (elbow + silhouette), scaling mandatory, spherical clusters |
| DBSCAN | Clusters by density | `eps` and `min_samples`, detects outliers, k not fixed |
| Hierarchical | Progressive merging of points | Dendrogram, choose the number of clusters by cutting the tree |

---

### 4.1 KMeans

**What:** splits data into exactly **k** clusters; each cluster is represented by its **centroid** (the mean of its points).

**How it works (Lloyd's algorithm):**

1. **Initialize** k centroids (smartly with `k-means++`, which spreads them apart).
2. **Assign** each point to the **nearest centroid**.
3. **Update** each centroid to the **mean** of its assigned points.
4. Repeat 2 and 3 until assignments stop changing (convergence).

**Objective:** minimize the **inertia** = sum of squared distances from each point to its centroid.

**Numeric example (1D):** points `[1, 2, 9, 10]`, k = 2, initial centroids 1 and 2.

- Assign: 1→c1; 2→c2; 9→c2; 10→c2. Update: c1 = 1, c2 = (2+9+10)/3 = 7.
- Assign: 1→c1; 2 (|2−1|=1 vs |2−7|=5)→c1; 9→c2; 10→c2. Update: c1 = 1.5, c2 = 9.5.
- Assign again: no change → **converged**. Clusters `{1,2}` and `{9,10}`; inertia = 4 × 0.5² = 1.0.

**Hyperparameters:**

| Hyperparameter | Meaning | Effect |
|---|---|---|
| `n_clusters` (k) | Number of clusters | The main choice (see elbow/silhouette) |
| `init` | `'k-means++'` (default) or `'random'` | k-means++ gives better, more stable starts |
| `n_init` | Number of different initializations tried (best kept) | KMeans can get stuck in a poor local solution; more runs = safer |
| `max_iter`, `tol` | Iteration limit, convergence tolerance | Rarely changed |
| `random_state` | Seed | Reproducibility |

**Limits:** you must choose k; assumes **roughly spherical, similar-size** clusters; sensitive to **outliers** (they pull centroids); needs scaling; cannot find ring or crescent shapes.

```python
from sklearn.cluster import KMeans
km = KMeans(n_clusters=3, init="k-means++", n_init=10, random_state=42).fit(X_scaled)
labels = km.labels_          # cluster of each sample
inertia = km.inertia_
```

#### Choosing k: the elbow method

- **What:** plot inertia for k = 2…8.
- **Why:** inertia always decreases as k grows (k = n gives 0), so we look for the point where adding clusters **no longer gives a big gain**.
- **How:** find the "elbow" where the curve bends. It is subjective, so confirm with silhouette.

#### Silhouette score

- **What:** for each point, compare the average distance to its own cluster (`a`, cohesion) with the average distance to the nearest other cluster (`b`, separation):

  `s = (b − a) / max(a, b)`

- **Range:** **−1 to 1**. Close to **1** = well matched to its cluster and far from others. Near **0** = on the border between clusters. **Negative** = probably in the wrong cluster.
- **Example:** a = 1, b = 5 → s = (5−1)/5 = **0.8** (very good).
- The **average silhouette** over all points is the score: **higher = better**. Choose the k with the highest value (and check no cluster is poor).

```python
from sklearn.metrics import silhouette_score
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled)
    print(k, round(km.inertia_, 1), round(silhouette_score(X_scaled, km.labels_), 3))
```

---

### 4.2 DBSCAN

**What:** Density-Based Spatial Clustering of Applications with Noise. Clusters are **dense regions** separated by sparse regions.

**Why:** unlike KMeans it does **not need k**, finds **arbitrary shapes** (rings, crescents) and flags **outliers** as noise.

**How it works:** every point is one of three types, based on the neighborhood of radius `eps`:

- **Core point:** has at least `min_samples` points (itself included) within distance `eps`.
- **Border point:** within `eps` of a core point but not core itself.
- **Noise point:** neither → labeled **−1**.

Clusters grow by connecting core points that are within `eps` of each other (plus their border points).

**Hyperparameters:**

| Hyperparameter | Meaning | Effect |
|---|---|---|
| `eps` | Neighborhood radius | **Too small** → everything becomes noise / many tiny clusters. **Too large** → clusters merge into one. Pick with a **k-distance plot** (sort each point's distance to its k-th neighbor and look for the elbow) |
| `min_samples` | Points needed to form a dense region | Larger = more noise, more conservative. Rule of thumb: `≥ n_features + 1`, often `2 × n_features` |
| `metric` | Distance (default Euclidean) | |

**Limits:** struggles when clusters have **very different densities**; `eps` is sensitive; needs scaling; has no `predict` for new points.

```python
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=0.5, min_samples=5).fit(X_scaled)
labels = db.labels_                      # -1 = noise
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
```

---

### 4.3 Hierarchical (agglomerative) clustering

**What:** Builds a **tree of merges**. Start with every point alone, then repeatedly **merge the two closest clusters** until one cluster remains.

**Why:** you don't fix k in advance; the **dendrogram** shows the structure at all levels and lets you choose the number of clusters afterward.

**How:** at each step, merge the pair of clusters with the smallest "distance", defined by the **linkage**:

| Linkage | Distance between clusters | Tendency |
|---|---|---|
| `ward` (default) | Merge that **minimally increases the within-cluster variance** | Compact, similar-size clusters (Euclidean only). Usually the best default |
| `complete` | **Maximum** distance between members | Compact clusters |
| `average` | Average distance between all pairs | Balanced |
| `single` | **Minimum** distance between members | Elongated "chains" (chaining effect) |

**Example (1D):** points `1, 2, 6, 7, 15`. Merge (1,2) at height 1 and (6,7) at height 1; then `{1,2}` with `{6,7}` at a larger height; finally 15 joins at the top at a much larger height. **Cutting the dendrogram** just below the large jump gives clusters `{1,2,6,7}` and `{15}`, or higher/lower cuts give other counts. **A long vertical line in the dendrogram = a natural place to cut.**

**Hyperparameters:** `n_clusters` (or `distance_threshold` to cut by height), `linkage`, `metric`.

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

agg = AgglomerativeClustering(n_clusters=3, linkage="ward").fit(X_scaled)
Z = linkage(X_scaled, method="ward")
dendrogram(Z, truncate_mode="level", p=5); plt.show()
```

**Limits:** slow and memory-hungry on large datasets (quadratic), merges cannot be undone.

---

### 4.4 Evaluating clusters

| Situation | Metric | Meaning |
|---|---|---|
| **No labels** | **Silhouette** (−1 to 1) | Higher = more cohesive and separated |
| **No labels** | **Davies-Bouldin** | Average similarity between each cluster and its most similar one. **Lower = better** (0 is the best) |
| **Known labels** (for validation) | **Adjusted Rand Index (ARI)** | Agreement between clusters and true classes, **corrected for chance**: 1 = perfect, ≈ 0 = random, can be negative. It ignores the actual label names (cluster "0" can match class "B") |

```python
from sklearn.metrics import davies_bouldin_score, adjusted_rand_score
print(davies_bouldin_score(X_scaled, labels))
print(adjusted_rand_score(y_true, labels))   # only if you know the true classes
```

### 4.5 PCA for visualization

**What:** Principal Component Analysis rotates the data to new axes (**principal components**) ordered by how much **variance** they capture. Keeping the first 2 compresses many features into 2 numbers.

**Why:** we cannot plot 30 dimensions. A 2D PCA projection lets us **see** the clusters.

**How:** it finds the direction of maximum variance (PC1), then the next direction orthogonal to it with the most remaining variance (PC2), and so on. `explained_variance_ratio_` tells how much information each component keeps (if PC1 + PC2 = 0.70, the 2D picture shows 70% of the variance).

**Rules:** **scale first** (otherwise large-unit features dominate), and PCA is used **only to visualize** here: cluster in the original (scaled) space, then **color the PCA plot by cluster label**.

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X_scaled)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="viridis", s=20)
plt.xlabel("PC1"); plt.ylabel("PC2")
print("Explained variance:", pca.explained_variance_ratio_.sum())
```

### 4.6 The roadmap trick: clustering a classification dataset

**What / why:** take a labeled dataset (e.g., Iris), **throw away the target**, cluster the features, then **compare the clusters with the true classes**. It tells you whether the natural structure of the data matches the labels, and it is a safe way to practice because you can verify the result.

```python
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

X, y = load_iris(return_X_y=True)
Xs = StandardScaler().fit_transform(X)
labels = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(Xs)

print(pd.crosstab(y, labels))                 # true class vs cluster
print("ARI:", adjusted_rand_score(y, labels))
```

**Checklist Phase 3:**

- [ ] Elbow curve + silhouette for k from 2 to 8, **choice of k justified**.
- [ ] 2D PCA visualization of the clusters.
- [ ] I can explain the **limits of KMeans** and what **DBSCAN** brings.

---

## 5. Phase 4: Visual interpretation

> Producing a chart is not enough. You must **read it out loud and draw a conclusion**. Prepare a template sentence for each visual.

| Visual | What it shows | How to interpret | Example sentence |
|---|---|---|---|
| **Confusion matrix** | Errors per class | Strong diagonal = good; find the confused classes | "The model detects 80 of 100 spam emails; the main error is 30 normal emails flagged as spam." |
| **ROC / PR curve** | Performance across thresholds | The closer to the top-left corner, the better; AUC ≈ 0.5 = random | "AUC = 0.97, so the model separates the classes very well regardless of the threshold." |
| **Feature importance** | Weight of features (trees) or coefficients (logreg) | Shows which variables drive the decision; a feature that is **too perfect** may reveal **leakage** | "`worst radius` is the most influential feature, which is medically plausible." |
| **Learning curve** | Train / validation score vs data size | **Big gap** = overfitting; **both low** = underfitting | "The train score is 1.0 but validation is 0.85: overfitting, more data or regularization may help." |
| **Loss curve (CNN, gradient descent)** | Cost per epoch | Should decrease; **validation rising** = overfitting | "Validation loss rises after epoch 12 while train loss keeps falling: I should stop early." |
| **Residuals and actual vs predicted (regression)** | Model errors | **Residuals centered around 0 with no structure** = good model | "Residuals are random around zero, but they widen for high prices: the model is less reliable there." |
| **Elbow, silhouette, PCA 2D (clustering)** | Choice of k, group separation | Well separated clusters = clear structure; **overlap = weak clusters** | "The silhouette peaks at k = 3 and PCA shows three distinct groups." |
| **Model comparison barplot** | CV scores per model | **Include the standard deviation**: a small difference is not necessarily significant | "RF 0.96 ± 0.01 and SVM 0.95 ± 0.02 overlap, so they are statistically close." |

#### How to produce the main visuals

```python
import numpy as np, matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.inspection import permutation_importance

# --- Feature importance (tree models) ---
imp = best.feature_importances_
idx = np.argsort(imp)[::-1][:10]
plt.barh(np.array(feature_names)[idx][::-1], imp[idx][::-1]); plt.title("Top 10 features")

# --- Permutation importance (works for ANY model, computed on test) ---
r = permutation_importance(best, X_test, y_test, n_repeats=10, random_state=42)
# r.importances_mean : the score drop when a feature is shuffled

# --- Learning curve ---
sizes, tr, va = learning_curve(best, X_train, y_train, cv=StratifiedKFold(5),
                               train_sizes=np.linspace(0.1, 1.0, 8), scoring="f1_macro")
plt.plot(sizes, tr.mean(axis=1), label="train")
plt.plot(sizes, va.mean(axis=1), label="validation"); plt.legend()

# --- Residuals (regression) ---
res = y_test - y_pred
plt.scatter(y_pred, res); plt.axhline(0, color="r"); plt.xlabel("Predicted"); plt.ylabel("Residual")

# --- Model comparison with std ---
plt.bar(names, means, yerr=stds, capsize=4)
```

**Notes on specific visuals:**

- **Feature importance (trees):** based on impurity decrease; can favor high-cardinality features. **Permutation importance** (shuffle one feature and measure the score drop) is model-agnostic and more reliable. **SHAP** (bonus) explains each individual prediction.
- **Learning curve:** if the validation curve is still rising at the right edge, more data would help.
- **Why a very important "perfect" feature is suspicious:** if one feature alone gives almost perfect accuracy, ask whether it contains the answer (e.g., an ID generated after the event) → possible **leakage**.

---

## 6. Glossary

| Term | Simple definition |
|---|---|
| **Feature** | Input variable (a column) |
| **Target / label** | What we predict |
| **Parameter** | Value learned by the model during training |
| **Hyperparameter** | Setting chosen by you before training |
| **Overfitting** | Memorizes training data, fails on new data |
| **Underfitting** | Too simple to capture the pattern |
| **Bias / variance** | Error from oversimplification / sensitivity to training data |
| **Regularization** | Penalty on large weights to prevent overfitting |
| **Cross-validation** | Repeated train/validation splits for reliable scores |
| **Data leakage** | Information from test/future entering training |
| **Pipeline** | Chain of preprocessing + model steps |
| **Threshold** | Probability cut-off to decide the positive class |
| **Epoch** | One full pass over the training data |
| **Learning rate** | Step size in gradient descent |
| **Inertia** | Sum of squared distances to centroids (KMeans) |
| **Centroid** | The mean point of a cluster |
| **Ensemble** | Combination of several models |
| **Bagging** | Train models on bootstrap samples in parallel, then average (Random Forest) |
| **Boosting** | Train models sequentially, each fixing the previous errors (XGBoost) |
| **Outlier** | An abnormal, extreme sample |

---

## 7. Resources

**Phase 1: Classification**

- scikit-learn, Model evaluation (all metrics): https://scikit-learn.org/stable/modules/model_evaluation.html
- scikit-learn, Cross-validation: https://scikit-learn.org/stable/modules/cross_validation.html
- scikit-learn, Hyperparameter tuning (Grid / Randomized): https://scikit-learn.org/stable/modules/grid_search.html
- scikit-learn, ConfusionMatrixDisplay: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html
- scikit-learn, multiclass ROC (example): https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
- imbalanced-learn, Over-sampling (SMOTE guide): https://imbalanced-learn.org/stable/over_sampling.html
- imbalanced-learn, SMOTE (API): https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html
- imbalanced-learn, Pipeline: https://imbalanced-learn.org/stable/references/generated/imblearn.pipeline.Pipeline.html
- Kaggle Learn, Intro to Machine Learning: https://www.kaggle.com/learn/intro-to-machine-learning
- Google ML Crash Course: https://developers.google.com/machine-learning/crash-course
- StatQuest (videos: logistic regression, SVM, random forest, ROC, SMOTE...): https://www.youtube.com/@statquest

**Phase 2: Regression and optimization**

- scikit-learn, Linear models (Ridge, Lasso): https://scikit-learn.org/stable/modules/linear_model.html
- scikit-learn, SGD: https://scikit-learn.org/stable/modules/sgd.html
- scikit-learn, Regression metrics: https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics
- 3Blue1Brown, gradient descent (videos): https://www.3blue1brown.com/topics/neural-networks
- Coursera, Machine Learning Specialization (Andrew Ng): https://www.coursera.org/specializations/machine-learning-introduction

**Phase 3: Clustering**

- scikit-learn, Clustering overview: https://scikit-learn.org/stable/modules/clustering.html
- scikit-learn, KMeans silhouette analysis (example): https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
- scikit-learn, PCA: https://scikit-learn.org/stable/modules/decomposition.html#pca
- StatQuest (videos: KMeans, hierarchical clustering, PCA): https://www.youtube.com/@statquest

**Phase 4: Visual interpretation**

- Seaborn tutorial: https://seaborn.pydata.org/tutorial.html
- Kaggle Learn, Data Visualization: https://www.kaggle.com/learn/data-visualization
- scikit-learn, Learning curves: https://scikit-learn.org/stable/modules/learning_curve.html
- scikit-learn, Permutation importance: https://scikit-learn.org/stable/modules/permutation_importance.html
- SHAP documentation (bonus): https://shap.readthedocs.io/

---

*Good luck. Remember the three golden rules: **split first**, **never touch the test set until the end**, and **always explain your choices**.*
