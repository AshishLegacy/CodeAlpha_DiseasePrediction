# CodeAlpha - Disease Prediction from Medical Data (Task 4)

Classify breast-tumor measurements as benign or malignant using the Wisconsin Diagnostic Breast Cancer dataset. This is an educational classification experiment, not a clinical diagnostic system or a prediction of future cancer risk.

## Run (Python 3.12 recommended)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe predict.py example_patient.json
```

Alternatively open walkthrough.ipynb with a Python kernel from this directory.

## Data and evaluation

- 569 samples, 30 numeric features from digitized fine-needle aspirate images; 212 malignant and 357 benign.
- Source: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic (CC BY 4.0); loaded from the copy bundled with scikit-learn: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
- Attribution: Wolberg, W., Mangasarian, O., Street, N., and Street, W. Breast Cancer Wisconsin (Diagnostic), UCI Machine Learning Repository, DOI: 10.24432/C5DW2B.
- Stratified 80/20 training/test split, seed 42. The saved split indices make the experiment auditable.
- Compare logistic regression, RBF SVM and random forest using five-fold stratified cross-validation within training data only. Median imputation and scaling are inside each pipeline, so validation data cannot influence them.
- Select by mean cross-validation ROC-AUC; fit the selected model on all training data; evaluate once on the held-out test data. Positive class is malignant=1 (explicitly recoded from the original labels).
- Report accuracy, malignant precision/recall/F1, ROC-AUC, a majority baseline, confusion matrix and per-class report. No hyperparameters or threshold are selected using the test set.

## Files

`train.py`: full experiment. `predict.py`: named-feature JSON inference. `example_patient.json`: an actual held-out dataset example for a demonstration, not a real user's record. `results/`: trained pipeline, exact metrics, plots, predictions and split indices. Only load joblib files you trust.

## Limitations and next steps

The held-out set is small; one split cannot establish clinical reliability. Measurements require specialized imaging, not a general symptom questionnaire. Dataset performance does not establish performance across hospitals or demographic groups. The model probabilities are not calibrated clinical risks. Further work would need external validation, calibration, subgroup assessment and clinical review. This project intentionally makes no treatment recommendation.

## Recorded result

Test accuracy: **96.49% on 114 held-out breast cancer dataset examples**. Full metrics and evaluation artifacts are in [RESULTS.md](RESULTS.md) and `results/`. Both training and saved-model inference were run locally.
