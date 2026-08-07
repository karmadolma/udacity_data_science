# Women's E-Commerce Clothing Reviews - Recommendation Prediction

## Overview

This project builds an end-to-end machine learning pipeline to predict whether a customer would recommend a clothing product based on structured product information and textual reviews.

The project demonstrates the complete machine learning workflow, including:

- Exploratory Data Analysis (EDA)
- Data preprocessing
- Feature engineering
- Natural Language Processing (NLP)
- Custom scikit-learn transformers
- Model training and evaluation
- Hyperparameter tuning using GridSearchCV

The goal is to showcase best practices for combining numerical, categorical, and text features into a single production-ready machine learning pipeline.

---

## Dataset

**Women's E-Commerce Clothing Reviews**

The dataset contains customer reviews from an online women's clothing retailer.

### Features

| Feature | Description |
|----------|-------------|
| Clothing ID | Product identifier |
| Age | Customer age |
| Title | Review title |
| Review Text | Customer review |
| Rating | Product rating (1–5) |
| Recommended IND | Target variable indicating whether the customer recommends the product |
| Positive Feedback Count | Number of positive feedback votes |
| Division Name | Product division |
| Department Name | Product department |
| Class Name | Product category |

---

## Project Workflow

### 1. Exploratory Data Analysis

- Summary statistics
- Missing value analysis
- Class imbalance inspection
- Distribution of numerical variables
- Review length analysis
- Correlation analysis

---

### 2. Data Preprocessing

The project uses a `ColumnTransformer` to process different feature types.

#### Numerical Features

- Missing value imputation
- Standard scaling

#### Categorical Features

- Missing value imputation
- One-hot encoding

#### Text Features

- spaCy preprocessing
- Lemmatization
- TF-IDF Vectorization

---

### 3. Natural Language Processing

The project uses **spaCy** (`en_core_web_sm`) to preprocess customer reviews.

Custom transformers were implemented using scikit-learn's `BaseEstimator` and `TransformerMixin`.

Examples include:

- `SpacyLemmatizer`
- `SpacyNER`

These transformers integrate seamlessly into the scikit-learn pipeline.

---

### 4. Feature Engineering

The final feature engineering pipeline combines:

- Numerical features
- Categorical features
- TF-IDF features extracted from:
  - Review Title
  - Review Text

using a `ColumnTransformer`.

---

### 5. Model

A Random Forest classifier was trained using the engineered features.


---

### 6. Hyperparameter Tuning

Model optimization was performed using **GridSearchCV**.

Example parameters tuned:

- Number of trees (`n_estimators`)
- Number of features considered at each split (`max_features`)

Cross-validation was used to identify the best-performing model.

---

## Technologies Used

- Python
- pandas
- NumPy
- scikit-learn
- spaCy
- matplotlib
- seaborn
- Jupyter Notebook

---





## Key Learning Outcomes

Through this project, I gained hands-on experience with:

- Building reusable scikit-learn pipelines
- Creating custom transformers compatible with scikit-learn
- Applying NLP techniques using spaCy
- Combining structured and unstructured data in a single model
- Hyperparameter tuning with cross-validation
- Developing reproducible machine learning workflows

---

## Author

**Karma Dolma Gurung**

SEO Analytics Lead | Data Analytics | Machine Learning | NLP | Python