# Month 5 — Flight Delay Prediction Using Machine Learning

## Project objective
Build a binary classification model to predict whether a flight will be delayed, using airline, route, date and other available flight features.

## Dataset
Use an Airline Delay Dataset from Kaggle. Download the CSV yourself and place it here:

`Month-5/data/airline_delay.csv`

The task brief does not specify one exact Kaggle dataset URL or fixed column schema. Before training, inspect the column names and update `TARGET_COLUMN` near the top of `flight_delay_prediction.py`. If your dataset contains delay minutes rather than a ready-made label, set `DELAY_MINUTES_COLUMN` to that column and set `TARGET_COLUMN` to a name that does not exist in the dataset, or adjust the target-building section.

A common convention is a delay of more than 15 minutes, but use the definition specified by your dataset/task.

## Setup (Windows / VS Code)
1. Extract this ZIP.
2. Open the `Month-5` folder in VS Code.
3. Create a folder named `data`.
4. Put the downloaded CSV inside `data` and rename it `airline_delay.csv`.
5. Open Terminal in VS Code and run:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python flight_delay_prediction.py
```

If activation is blocked, use your normal Python environment or activate the environment using the method supported by your terminal.

## What the script does
- Prints dataset shape, columns and sample rows.
- Drops columns with more than 60% missing values.
- Creates weekday and month features from a recognized flight-date column.
- Imputes missing values and one-hot encodes categorical features.
- Trains Logistic Regression and Random Forest models.
- Reports accuracy, precision, recall, ROC-AUC and classification report.
- Saves confusion matrices, ROC curves and a model comparison CSV in `outputs/`.
- Saves the best model by ROC-AUC to `models/flight_delay_model.pkl`.

## Important notes
- Dataset column names vary. Inspect them and configure the target column before running.
- Do not use actual arrival delay or other information only known after the flight as a prediction feature; that would cause target leakage.
- Accuracy alone may be misleading if delayed flights are a minority. Review precision, recall, ROC-AUC and the confusion matrix.
- Run the code on your own dataset and report the real measured metrics. The ZIP does not include a dataset or fabricated evaluation results.

## Deliverables
- Python script (can be copied into a notebook for the required `.ipynb` deliverable)
- Saved `.pkl` model
- Evaluation report based on your real run
- `outputs/model_comparison.csv`, confusion-matrix images and ROC curve

## Suggested GitHub location
`Skybrisk/Month-5/`
