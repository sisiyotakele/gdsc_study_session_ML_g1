# Breast Cancer Detection using Logistic Regression

## Project Overview

This project applies **Logistic Regression** to predict whether a breast tumor is **benign** or **malignant** based on medical diagnostic features. The primary objective is to demonstrate the complete machine learning workflow in a clear, structured, and beginner-friendly manner, avoiding unnecessary complexity.

The project is implemented in **Google Colab** using Python and popular data science libraries.

---

## Dataset

* **File name:** `breast_cancer_bd.csv`
* **Target column:** `Class`

  * `2` → Benign
  * `4` → Malignant

### Dropped Column

* **Sample code number**
  This column is an identifier only and does not contribute to prediction accuracy, so it is removed during preprocessing.

---

## Tools & Libraries Used

* **pandas** – Data loading and manipulation
* **numpy** – Numerical operations
* **seaborn** – Data visualization
* **matplotlib** – Plotting graphs
* **scikit-learn** – Machine learning model training and evaluation

---

## Steps Followed

### 1️⃣ Load the Dataset

```python
import pandas as pd

df = pd.read_csv("breast_cancer_bd.csv")
df.head()
```

---

### 2️⃣ Data Cleaning

Remove the unnecessary identifier column:

```python
df = df.drop('Sample code number', axis=1)
```

---

### 3️⃣ Split Features and Target

```python
X = df.drop('Class', axis=1)
y = df['Class']
```

---

### 4️⃣ Visualize Data Distribution

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='Class', data=df)
plt.title("Distribution of Cancer Classes")
plt.show()
```

This visualization helps understand the class distribution between benign and malignant tumors.

---

### 5️⃣ Train-Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
```

---

### 6️⃣ Train Logistic Regression Model

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
```

---

### 7️⃣ Make Predictions

```python
y_pred = model.predict(X_test)
```

---

### 8️⃣ Model Evaluation

```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
```

---

## Evaluation Metrics Used

* **Accuracy** – Overall correctness of the model
* **Confusion Matrix** – Breakdown of correct and incorrect predictions
* **Classification Report** – Precision, Recall, and F1-score for each class

---

## Conclusion

This project demonstrates that **Logistic Regression** can effectively classify breast cancer tumors using medical features. The emphasis is on understanding each stage of the machine learning pipeline, making it suitable for beginners and educational purposes.

---

## How to Run This Project

1. Open **Google Colab**
2. Upload the `breast_cancer_bd.csv` dataset
3. Copy the code into a notebook
4. Run each cell step by step

---

## Author

Developed as a learning project to understand binary classification using Logistic Regression.
