# Skybrisk Month 2 – Task 2: Simple Predictive Model

## Objective
Use Scikit-learn to predict a target column from other features. This project predicts Titanic passenger survival (`Survived`).

## Dataset
Download the CSV from https://www.kaggle.com/datasets/yasserh/titanic-dataset and save it in this folder as `Titanic-Dataset.csv`. It must include the `Survived` column.

## Files
- `titanic_prediction.py`: load, explore, preprocess, train, evaluate
- `requirements.txt`: required packages
- `Month_2_Notes.docx`: study notes
- `Month_2_Report_Template.docx`: report template
- `Titanic-Dataset.csv`: download separately from Kaggle

## Run in VS Code
1. Open this folder in VS Code.
2. In the terminal, run `python -m pip install -r requirements.txt`.
3. Download the dataset and put `Titanic-Dataset.csv` in this folder.
4. Run `python titanic_prediction.py`.

## Workflow
Explore the data; handle missing values; encode categorical features; split 80% train / 20% test; train Logistic Regression; evaluate accuracy, confusion matrix, precision, recall, and F1-score.

Important: use the actual output from your run in the report. Model results may vary with dataset versions and should not be guessed.
