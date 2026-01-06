# 🧠 Multiple Logistic Regression from Scratch (HR Attrition Analysis)

A **production-style Machine Learning project** implementing **Multiple Logistic Regression** using pure Python scripts (no notebooks), real-world HR data, and **matplotlib-based visualizations** to understand *why* employees leave a company.

This project focuses on **clarity, interpretability, and engineering discipline**, not just model accuracy.

---

## 📌 Problem Statement

Employee attrition is a costly issue for organizations. The goal of this project is to:

> **Predict whether an employee will leave the company (1) or stay (0)**

using multiple features such as satisfaction level, evaluation score, working hours, department, salary, and promotion history.

This is a classic **binary classification problem** solved using **Multiple Logistic Regression**.

---

## 📂 Dataset

* **Source**: Public HR Analytics dataset
* **Target Variable**: `left` (0 = Stayed, 1 = Left)
* **Size**: ~15,000 records
* **Type**: Real-world, imbalanced dataset

### Key Features

* `satisfaction_level`
* `last_evaluation`
* `number_project`
* `average_montly_hours`
* `time_spend_company`
* `Work_accident`
* `promotion_last_5years`
* `sales` (department – categorical)
* `salary` (categorical)

---

## 🏗️ Project Structure

```
logistic_regression_hr/
│
├── data/
│   └── HR_comma_sep.csv
│
├── src/
│   ├── load_data.py        # Data loading
│   ├── preprocess.py       # Encoding + Standardization
│   ├── train.py            # Model training & evaluation
│   ├── visualize.py        # Coefficient & confusion matrix plots
│   └── viz_sigmoid.py      # Logistic curve visualization
│
├── main.py                 # Pipeline execution
├── requirements.txt
└── README.md
```

This structure mirrors **real ML codebases** and avoids notebook-driven workflows.

---

## 🔧 Preprocessing Pipeline

✔ One-Hot Encoding for categorical variables (`sales`, `salary`)
✔ Feature standardization using `StandardScaler`
✔ Labels left untouched (as required for classification)

Why this matters:

* Prevents string-to-float errors
* Ensures stable optimization
* Makes coefficients comparable

---

## 🧮 Model

* **Algorithm**: Multiple Logistic Regression
* **Library**: `scikit-learn`
* **Solver**: Default (lbfgs)
* **Regularization**: L2 (default)

Logistic regression was chosen for:

* Interpretability
* Probabilistic output
* Strong baseline performance

---

## 📈 Model Performance

* **Accuracy**: ~**0.78**

This is a realistic and expected result for this dataset.

⚠️ Note:
The dataset is **class-imbalanced** (~76% stay, ~24% leave). Therefore, accuracy alone is not sufficient and is interpreted alongside:

* Confusion Matrix
* Probability-based analysis

---

## 📊 Visualizations

All visualizations are done using **matplotlib only** (no seaborn, no notebooks).

### 1️⃣ Feature Coefficient Plot

Shows how each feature impacts the probability of an employee leaving:

* Positive coefficient → increases attrition probability
* Negative coefficient → reduces attrition probability

### 2️⃣ Confusion Matrix

Helps analyze:

* Correctly identified leavers
* Missed attrition cases
* Model usefulness beyond accuracy

### 3️⃣ Logistic (Sigmoid) Curves — *Key Insight*

For each feature:

* The feature is varied across a standardized range
* All other features are held at their mean
* Resulting sigmoid curve is plotted

This allows visualization of:

* Direction of influence
* Strength of influence
* Decision boundary behavior

This is the **correct way** to visualize logistic regression in a multivariate setting.

---

## 🧠 Key Learnings

* Difference between **linear** and **logistic** regression
* Handling categorical variables correctly
* Importance of feature scaling
* Interpreting logistic coefficients
* Why accuracy can be misleading
* How probabilistic classifiers behave internally

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
python main.py
```

All plots will render sequentially using matplotlib.

---

## 📌 Why This Project Stands Out

✔ Uses **real-world data**
✔ No notebooks — clean Python scripts
✔ Focus on **interpretability**, not just scores
✔ Proper ML reasoning (scaling, imbalance, visualization)
✔ Interview-ready explanations

---

## 🔮 Future Improvements

* ROC Curve & AUC visualization
* Precision, Recall, F1-score analysis
* Class-weighted logistic regression
* Threshold tuning
* Logistic regression from scratch (no sklearn)

---

## 👤 Author

**Maverick(Yashraj_Bhogade)**
Aspiring Machine Learning Engineer

---

> *This project was built to understand logistic regression deeply — not to chase artificial accuracy.*
