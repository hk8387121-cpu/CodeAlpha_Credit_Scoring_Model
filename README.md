\# Credit Scoring Model



\## CodeAlpha Internship - Task 1



A machine learning-based credit scoring model that predicts whether a customer is likely to default on a credit card payment.



The project compares three classification algorithms:



\- Logistic Regression

\- Decision Tree

\- Random Forest



The models are evaluated using Accuracy, Precision, Recall, F1-Score, and ROC-AUC.



\---



\## Objective



The objective of this project is to build a classification model for predicting credit card payment default using customer financial and demographic information.



The project focuses on identifying customers who are more likely to default and comparing different machine learning classification algorithms.



\---



\## Dataset



The project uses the \*\*Default of Credit Card Clients Dataset\*\*.



Dataset characteristics:



\- Records: 30,000

\- Original columns: 25

\- Processed columns: 24

\- Target variable: `default\_payment`



\### Target Classes



| Value | Meaning |

|---|---|

| 0 | No Default |

| 1 | Default |



\### Target Distribution



\- No Default: 23,364

\- Default: 6,636

\- Default Rate: 22.12%



The dataset contains financial and demographic features including:



\- Credit limit

\- Age

\- Gender

\- Education

\- Marital status

\- Payment history

\- Bill amounts

\- Previous payment amounts



\---



\## Project Approach



```text

Dataset

&#x20;  ↓

Data Preprocessing

&#x20;  ↓

Exploratory Data Analysis

&#x20;  ↓

Train/Test Split

&#x20;  ↓

Feature Preprocessing

&#x20;  ↓

Model Training

&#x20;  ├── Logistic Regression

&#x20;  ├── Decision Tree

&#x20;  └── Random Forest

&#x20;  ↓

Model Evaluation

&#x20;  ├── Accuracy

&#x20;  ├── Precision

&#x20;  ├── Recall

&#x20;  ├── F1-Score

&#x20;  └── ROC-AUC

&#x20;  ↓

Model Comparison

&#x20;  ↓

Best Model Selection

&#x20;  ↓

Credit Default Prediction

