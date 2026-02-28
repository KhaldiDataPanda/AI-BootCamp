# Homework 2 — Basics of Machine Learning
### Case Study: **MedPredict — A Clinical Data Analysis System**

---

## Background

You have joined **MedPredict**, a healthcare analytics startup. The company works with clinical datasets to help doctors understand how patient measurements relate to disease progression. Your goal is to build a full ML pipeline — from loading and exploring the data, all the way to training, visualizing, and evaluating a **Simple Linear Regression** model.

You will work with the **Diabetes Dataset** built into `sklearn`, which contains 10 medical features measured from 442 patients, and a target variable representing **disease progression one year after baseline**.

Each part of this homework builds on the previous one. Work through them in order.

> **Load the dataset once at the top of your notebook and reuse it throughout:**
> ```python
> from sklearn.datasets import load_diabetes
> import pandas as pd
> import numpy as np
> import matplotlib.pyplot as plt
>
> data = load_diabetes()
> df = pd.DataFrame(data.data, columns=data.feature_names)
> df['progression'] = data.target
> ```

---

## Part 1 — Data Exploration

Before building any model, a data scientist must understand the dataset.

### Task 1.1 — Inspect the Dataset

1. Print the **shape** of `df` (number of rows and columns).
2. Print the **column names**.
3. Print the **first 5 rows** using `.head()`.
4. Print basic statistics using `.describe()`.

<details>
<summary>💡 Hint</summary>

```python
print(df.shape)
print(df.columns.tolist())
df.head()
df.describe()
```

- `.shape` returns a tuple `(rows, columns)`.
- `.describe()` gives count, mean, std, min, max, and quartile values for each column.

</details>

---

### Task 1.2 — Key Variable Summary

Create individual Python variables to store the following and print each with a descriptive label:

| Variable | What to store |
|---|---|
| `num_patients` | Total number of rows |
| `num_features` | Number of feature columns (exclude `progression`) |
| `avg_progression` | Mean value of the `progression` column |
| `max_progression` | Maximum value of the `progression` column |
| `min_progression` | Minimum value of the `progression` column |

**Expected output (example):**
```
Number of Patients: 442
Number of Features: 10
Average Disease Progression: 152.13
Max Disease Progression: 346.0
Min Disease Progression: 25.0
```

<details>
<summary>💡 Hint</summary>

```python
num_patients    = len(df)
num_features    = len(df.columns) - 1   # exclude the target column
avg_progression = df['progression'].mean()
```

Use `round(value, 2)` to format your printed output cleanly.

</details>

---

## Part 2 — Conditions & Data Labeling

The medical team wants to categorize patients based on their measurements.

### Task 2.1 — BMI Category Check

The dataset's `bmi` column is **standardized** (mean ≈ 0). A patient with `bmi > 0` is above-average BMI; a patient with `bmi <= 0` is at or below average.

Pick the **first patient** (row index 0) and write an `if / else` block that prints:
- `"Above average BMI — monitor diet."` if `bmi > 0`
- `"At or below average BMI."` otherwise

<details>
<summary>💡 Hint</summary>

```python
bmi_patient_0 = df['bmi'].iloc[0]
if bmi_patient_0 > 0:
    ...
```

Use `.iloc[0]` to access the first row's value by position.

</details>

---

### Task 2.2 — Disease Severity Classification

Write an `if / elif / else` block that classifies a patient's disease progression score into a severity level. Use the variable `score = df['progression'].iloc[0]`:

| Condition | Severity |
|---|---|
| `score >= 250` | `"High Severity"` |
| `score >= 150` | `"Moderate Severity"` |
| `score >= 75` | `"Mild Severity"` |
| Below 75 | `"Low Severity"` |

Print the score and its corresponding severity label.

<details>
<summary>💡 Hint</summary>

```python
score = df['progression'].iloc[0]
if score >= 250:
    severity = "High Severity"
elif score >= 150:
    ...
```

Chain conditions from the **highest** threshold down to avoid overlap.

</details>

---

### Task 2.3 — Combined Condition Check

Using **logical operators**, write a condition for a given patient (use index 0) that checks:

> The patient has **above average BMI** (`bmi > 0`) **AND** a high disease progression (`progression > 150`)

- If both are true → `"High-risk patient — escalate to specialist."`
- If only one is true → `"Patient requires monitoring."`
- If neither → `"Patient appears stable."`

<details>
<summary>💡 Hint</summary>

```python
bmi   = df['bmi'].iloc[0]
score = df['progression'].iloc[0]

if bmi > 0 and score > 150:
    ...
elif bmi > 0 or score > 150:
    ...
else:
    ...
```

</details>

---

## Part 3 — Loops & Feature Analysis

The team wants a summary report for all numerical features.

### Task 3.1 — Feature Statistics with a `for` Loop

Loop through all **feature columns** (every column except `progression`) and for each one print:
- The feature name
- Its **mean** value (rounded to 4 decimal places)
- Its **standard deviation** (rounded to 4 decimal places)

**Expected output format:**
```
age  | Mean:  0.0000 | Std: 0.0476
bmi  | Mean:  0.0000 | Std: 0.0476
...
```

<details>
<summary>💡 Hint</summary>

```python
feature_cols = data.feature_names   # list of the 10 feature names

for col in feature_cols:
    mean = round(df[col].mean(), 4)
    std  = round(df[col].std(), 4)
    print(f"{col:<5} | Mean: {mean:>8.4f} | Std: {std:>8.4f}")
```

</details>

---

### Task 3.2 — Manual Min/Max Search (Loops + Conditions)

Without using `max()`, `min()`, or `.max()` / `.min()` pandas methods, write a loop that finds:
1. The **highest** progression score in the dataset.
2. The **lowest** progression score in the dataset.
3. The **patient index** (row number) of each.

Print all four values at the end.

<details>
<summary>💡 Hint</summary>

```python
highest = df['progression'].iloc[0]
highest_idx = 0

for i in range(len(df)):
    val = df['progression'].iloc[i]
    if val > highest:
        highest = val
        highest_idx = i
```

Initialize `lowest` similarly and update it inside the same loop.

</details>

---

### Task 3.3 — Count High-Risk Patients (Loops + Conditions + Counter)

Define a patient as **high-risk** if their `progression` score is above the dataset's mean progression value. Using a `for` loop:

1. Count the number of high-risk patients.
2. Store their row indices in a list called `high_risk_indices`.
3. Print the count and the first 10 indices.

<details>
<summary>💡 Hint</summary>

```python
threshold = df['progression'].mean()
high_risk_indices = []

for i in range(len(df)):
    if df['progression'].iloc[i] > threshold:
        high_risk_indices.append(i)

print("High-risk patients:", len(high_risk_indices))
print("First 10 indices:", high_risk_indices[:10])
```

</details>

---

## Part 4 — Data Visualization

A picture is worth a thousand rows. Before modeling, always visualize.

### Task 4.1 — Distribution Histograms

Plot a **histogram** for each of the following four columns in a single figure with **2 rows × 2 columns** of subplots:
- `bmi`, `bp`, `age`, `progression`

Label every axis and give each subplot a title.

<details>
<summary>💡 Hint</summary>

```python
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
cols = ['bmi', 'bp', 'age', 'progression']

for ax, col in zip(axes.flatten(), cols):
    ax.hist(df[col], bins=20, color='steelblue', edgecolor='white')
    ax.set_title(col.upper())
    ax.set_xlabel(col)
    ax.set_ylabel("Count")

plt.tight_layout()
plt.show()
```

`axes.flatten()` converts the 2D array of axes into a flat list so you can iterate over it easily.

</details>

---

### Task 4.2 — Scatter Plots vs. Target

Create **four scatter plots** (2×2 layout) showing the relationship between each of the four features below and the `progression` target:
- `bmi`, `bp`, `s1`, `s5`

Color the dots blue, label the axes clearly, and add a title to each subplot.

<details>
<summary>💡 Hint</summary>

```python
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
features = ['bmi', 'bp', 's1', 's5']

for ax, feat in zip(axes.flatten(), features):
    ax.scatter(df[feat], df['progression'], color='blue', alpha=0.4, s=15)
    ax.set_xlabel(feat)
    ax.set_ylabel("Progression")
    ax.set_title(f"{feat.upper()} vs Progression")

plt.tight_layout()
plt.show()
```

</details>

---

### Task 4.3 — Visual Insight (Visualization + Loops + Conditions)

Loop through the feature columns list `['bmi', 'bp', 'age', 's1', 's5', 's6']` and **calculate the correlation** between each feature and `progression` using `df[feat].corr(df['progression'])`. Then:

1. Print each feature name and its correlation value (rounded to 4 decimal places).
2. Add a label next to each value:
   - `"Strong"` if `|correlation| >= 0.4`
   - `"Moderate"` if `|correlation| >= 0.2`
   - `"Weak"` otherwise

<details>
<summary>💡 Hint</summary>

```python
features = ['bmi', 'bp', 'age', 's1', 's5', 's6']

for feat in features:
    corr = round(df[feat].corr(df['progression']), 4)
    if abs(corr) >= 0.4:
        label = "Strong"
    elif abs(corr) >= 0.2:
        label = "Moderate"
    else:
        label = "Weak"
    print(f"{feat:<5}  corr = {corr:>7.4f}  →  {label}")
```

Correlation ranges from -1 to +1. Use `abs()` to evaluate strength regardless of direction.

</details>

---

## Part 5 — Train / Test Split

Before training a model, we separate our data so we can evaluate it fairly.

### Task 5.1 — Split the Dataset

Using `np.random.rand()`, create a boolean mask and split the dataset into:
- **80% training set** → `train`
- **20% test set** → `test`

Print the size of each split.

<details>
<summary>💡 Hint</summary>

```python
np.random.seed(42)                   # for reproducibility
msk   = np.random.rand(len(df)) < 0.8
train = df[msk]
test  = df[~msk]

print("Training set size:", len(train))
print("Test set size    :", len(test))
```

Setting a seed ensures you get the same split every time you run the notebook.

</details>

---

### Task 5.2 — Verify the Split (Conditions + Variables)

After splitting, check:
1. That neither split is empty — use an `if` statement and print `"Split looks good!"` or raise a warning.
2. Print the **percentage** of data in each split (rounded to 1 decimal place).
3. Print the **mean progression** of both the train and test sets. Are they close to each other? Add a comment explaining what this tells you about the split quality.

<details>
<summary>💡 Hint</summary>

```python
train_pct = round(len(train) / len(df) * 100, 1)
test_pct  = round(len(test)  / len(df) * 100, 1)

print(f"Train: {train_pct}%  |  Test: {test_pct}%")
print(f"Train mean progression: {train['progression'].mean():.2f}")
print(f"Test  mean progression: {test['progression'].mean():.2f}")
```

If train and test means are close, the split is representative — the data was not accidentally sorted.

</details>

---

## Part 6 — Build & Train the Model

Time to write your first ML model.

### Task 6.1 — Train a Linear Regression on BMI

Using `sklearn.linear_model.LinearRegression`:

1. Prepare `train_x` from the `bmi` column of the training set and `train_y` from `progression`.
2. Fit the model.
3. Print the **coefficient** (slope) and **intercept** of the fitted line.

<details>
<summary>💡 Hint</summary>

```python
from sklearn import linear_model

regr = linear_model.LinearRegression()

train_x = np.asarray(train[['bmi']])
train_y = np.asarray(train[['progression']])

regr.fit(train_x, train_y)

print("Coefficient (slope):", regr.coef_[0][0])
print("Intercept           :", regr.intercept_[0])
```

The model learns the equation: `progression = slope × bmi + intercept`

</details>

---

### Task 6.2 — Plot the Fit Line

Create a scatter plot of `train.bmi` vs `train.progression` and overlay the **regression line** in red.

- Label both axes.
- Add a title: `"BMI vs Disease Progression (Training Set)"`.

<details>
<summary>💡 Hint</summary>

```python
plt.scatter(train.bmi, train.progression, color='blue', alpha=0.5, label='Data')
plt.plot(train_x,
         regr.coef_[0][0] * train_x + regr.intercept_[0],
         '-r', linewidth=2, label='Fit line')
plt.xlabel("BMI (standardized)")
plt.ylabel("Disease Progression")
plt.title("BMI vs Disease Progression (Training Set)")
plt.legend()
plt.show()
```

</details>

---

### Task 6.3 — Make Predictions

Using the trained model, **predict** the progression values for the **test set**:

1. Prepare `test_x` from the `bmi` column of the test set.
2. Generate predictions using `.predict()`.
3. Print the **first 5 predicted values** side-by-side with the actual values.

<details>
<summary>💡 Hint</summary>

```python
test_x = np.asarray(test[['bmi']])
test_y = np.asarray(test[['progression']])

predictions = regr.predict(test_x)

for i in range(5):
    print(f"Predicted: {predictions[i][0]:.2f}   |   Actual: {test_y[i][0]:.2f}")
```

</details>

---

## Part 7 — Model Evaluation & Comparison

A model is only as good as its metrics.

### Task 7.1 — Compute Evaluation Metrics

Using the predictions from Task 6.3 and the actual `test_y` values, calculate and print:

1. **Mean Absolute Error (MAE)**
2. **Mean Squared Error (MSE)**
3. **R² Score**

<details>
<summary>💡 Hint</summary>

```python
from sklearn.metrics import r2_score

mae = np.mean(np.absolute(predictions - test_y))
mse = np.mean((predictions - test_y) ** 2)
r2  = r2_score(test_y, predictions)

print(f"MAE : {mae:.2f}")
print(f"MSE : {mse:.2f}")
print(f"R²  : {r2:.4f}")
```

- **MAE** is easy to interpret: average absolute prediction error in the same units as the target.
- **MSE** penalizes large errors more heavily.
- **R²** of 1.0 is a perfect fit; values close to 0 mean the model barely explains the variance.

</details>

---

### Task 7.2 — Try a Different Feature (Full Pipeline)

Repeat the entire pipeline from **Tasks 6.1 → 7.1** but this time use the **`s5`** feature (a blood serum measurement) instead of `bmi`.

1. Prepare `train_x` and `test_x` from the `s5` column.
2. Fit a new model.
3. Print coefficients and intercept.
4. Plot the fit line on the training data.
5. Predict and print MAE, MSE, and R².

<details>
<summary>💡 Hint</summary>

Follow the exact same steps as Part 6 and Task 7.1, just replace `'bmi'` with `'s5'` everywhere.

```python
train_x = np.asarray(train[['s5']])
test_x  = np.asarray(test[['s5']])

regr_s5 = linear_model.LinearRegression()
regr_s5.fit(train_x, train_y)
...
```

</details>

---

### Task 7.3 — Compare & Conclude (Metrics + Conditions + Print)

Create a summary comparison table by storing the metrics for **both models** in a dictionary, then print a formatted report:

```
========== Model Comparison ==========
Feature     MAE       MSE       R²
--------------------------------------
bmi         55.23     4871.10   0.3081
s5          52.41     4432.77   0.3664
--------------------------------------
Better model (by R²): s5
======================================
```

Use a condition to automatically detect and print which model has the **higher R²**.

<details>
<summary>💡 Hint</summary>

```python
results = {
    'bmi': {'MAE': mae_bmi, 'MSE': mse_bmi, 'R2': r2_bmi},
    's5' : {'MAE': mae_s5,  'MSE': mse_s5,  'R2': r2_s5},
}

print("=" * 38)
print(f"{'Feature':<10} {'MAE':>8} {'MSE':>10} {'R²':>8}")
print("-" * 38)
for feat, m in results.items():
    print(f"{feat:<10} {m['MAE']:>8.2f} {m['MSE']:>10.2f} {m['R2']:>8.4f}")
print("-" * 38)

best = max(results, key=lambda f: results[f]['R2'])
print(f"Better model (by R²): {best}")
print("=" * 38)
```

`max()` with a `key` function picks the dictionary key whose value satisfies the condition.

</details>

---

## Submission Checklist

Before submitting, make sure your notebook:

- [ ] Has **all 7 parts** completed with working code cells.
- [ ] Prints **clear, labelled output** for every task.
- [ ] Includes **comments** explaining your reasoning where appropriate.
- [ ] Has **no crash-causing errors** when run top to bottom with **Restart & Run All**.
- [ ] All plots have **axis labels and titles**.

---

