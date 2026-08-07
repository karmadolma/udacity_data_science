# Women's E-Commerce Clothing Reviews Recommendation Prediction

This project develops an end-to-end machine learning pipeline to predict whether a customer would recommend a clothing product based on product information and customer reviews from the Women's E-Commerce Clothing Reviews dataset.

The pipeline combines structured data (numerical and categorical features) with unstructured text (review title and review text) using custom preprocessing and Natural Language Processing (NLP) techniques. The project demonstrates the complete machine learning workflow, including exploratory data analysis, feature engineering, preprocessing, model training, hyperparameter tuning, and evaluation.

---

## Getting Started

Clone this repository to your local machine and install the required dependencies before running the project.

### Dependencies

The project was developed using Python 3.10+.

Required packages include:

```text
pandas
numpy
scikit-learn
spacy
matplotlib
seaborn
jupyter
joblib
```

Download the spaCy English language model:

```bash
python -m spacy download en_core_web_sm
```

---

### Installation

1. Clone the repository

```bash
git clone https://github.com/karmadolma/udacity_data_science.git
```

2. Navigate to the project directory

```bash
cd udacity_data_science/nlp_pipeline_project
```

3. (Optional) Create and activate a virtual environment

```bash
python -m venv venv
```

macOS/Linux

```bash
source venv/bin/activate
```

Windows

```bash
venv\Scripts\activate
```

4. Install the required packages

```bash
pip install -r requirements.txt
```

5. Download the spaCy language model

```bash
python -m spacy download en_core_web_sm
```

---

## Testing

Train and evaluate the machine learning pipeline by running the notebook or Python script.

Example:

```bash
python model.py
```

or open

```text
recommendation_prediction.ipynb
```

using Jupyter Notebook.

---

### Break Down Tests

The project includes the following evaluation steps:

- Verify that the preprocessing pipeline successfully transforms numerical, categorical, and text features.
- Evaluate the trained classifier using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
- Perform cross-validation using GridSearchCV to identify the best hyperparameters.
- Compare baseline and tuned model performance.

Example output:

```text
Best Parameters:
{
    'randomforestclassifier__max_features': 100,
    'randomforestclassifier__n_estimators': 200
}

Accuracy: 0.89
```

---

## Project Instructions

The project consists of the following components:

1. Perform Exploratory Data Analysis (EDA) to understand the dataset.
2. Handle missing values and prepare numerical and categorical features.
3. Build custom spaCy transformers for text preprocessing.
4. Apply lemmatization to customer reviews.
5. Convert review text into TF-IDF features.
6. Combine numerical, categorical, and text features using a ColumnTransformer.
7. Train a Random Forest classifier within a scikit-learn Pipeline.
8. Tune model hyperparameters using GridSearchCV.
9. Evaluate the final model on the test dataset.

---

## Built With

- **Python** – Programming language
- **pandas** – Data manipulation and analysis
- **NumPy** – Numerical computing
- **scikit-learn** – Machine learning pipelines, preprocessing, model training, and evaluation
- **spaCy** – Natural Language Processing and lemmatization
- **matplotlib** – Data visualization
- **seaborn** – Statistical data visualization
- **Jupyter Notebook** – Interactive development environment

---

## License

This project is licensed under the MIT License. See the `LICENSE.txt` file for more information.