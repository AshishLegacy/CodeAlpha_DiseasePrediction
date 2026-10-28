"""Compare classifiers using training-only cross-validation; evaluate one on holdout."""
from pathlib import Path
import json
import platform
import numpy as np
import sklearn
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (classification_report, confusion_matrix, ConfusionMatrixDisplay,
    RocCurveDisplay, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score)

BASE = Path(__file__).resolve().parent

def main():
    out = BASE / 'results'
    out.mkdir(exist_ok=True)
    data = load_breast_cancer()
    # UCI/sklearn uses malignant=0. Recode so positive=1 means malignant.
    X, y = data.data, (data.target == 0).astype(int)
    indices = np.arange(len(y))
    train, test = train_test_split(indices, test_size=0.2, stratify=y, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = {
        'Logistic regression': make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                                             LogisticRegression(max_iter=2000, random_state=42)),
        'SVM': make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                             SVC(kernel='rbf', probability=True, random_state=42)),
        'Random forest': make_pipeline(SimpleImputer(strategy='median'),
                                       RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=2)),
    }
    scores = {}
    for name, model in models.items():
        result = cross_validate(model, X[train], y[train], cv=cv,
                                scoring=['roc_auc', 'f1', 'recall', 'precision'], n_jobs=1)
        scores[name] = {metric: {'mean': float(result['test_' + metric].mean()),
                                'std': float(result['test_' + metric].std())}
                        for metric in ['roc_auc', 'f1', 'recall', 'precision']}
    winner = max(scores, key=lambda name: scores[name]['roc_auc']['mean'])
    model = models[winner].fit(X[train], y[train])
    pred = model.predict(X[test])
    prob = model.predict_proba(X[test])[:, 1]
    baseline = DummyClassifier(strategy='most_frequent').fit(X[train], y[train])
    metrics = dict(accuracy=float(accuracy_score(y[test], pred)),
                   precision=float(precision_score(y[test], pred, zero_division=0)),
                   recall=float(recall_score(y[test], pred)), f1=float(f1_score(y[test], pred)),
                   roc_auc=float(roc_auc_score(y[test], prob)))
    payload = {'selected_model': winner, 'selection_rule': 'highest mean training CV ROC-AUC',
               'seed': 42, 'positive_class': 'malignant', 'train_samples': len(train),
               'test_samples': len(test), 'cross_validation': scores, 'test_metrics': metrics,
               'baseline_test_accuracy': float(baseline.score(X[test], y[test])),
               'confusion_matrix_labels': ['benign', 'malignant'],
               'confusion_matrix': confusion_matrix(y[test], pred).tolist(),
               'classification_report': classification_report(y[test], pred,
                     target_names=['benign', 'malignant'], output_dict=True),
               'versions': {'python': platform.python_version(), 'sklearn': sklearn.__version__, 'numpy': np.__version__}}
    (out / 'metrics.json').write_text(json.dumps(payload, indent=2))
    np.savez_compressed(out / 'split_indices.npz', train=train, test=test)
    np.savetxt(out / 'test_predictions.csv', np.c_[test, y[test], pred, prob], delimiter=',',
               header='dataset_row,actual_malignant,predicted_malignant,probability_malignant', comments='')
    joblib.dump({'model': model, 'feature_names': list(data.feature_names),
                 'labels': {0: 'benign', 1: 'malignant'}}, out / 'model.joblib')
    (BASE / 'example_patient.json').write_text(json.dumps(dict(zip(data.feature_names,
                                                                 data.data[test[0]].tolist())), indent=2))
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y[test], pred, display_labels=['Benign', 'Malignant'], ax=ax, colorbar=False)
    ax.set_title('Breast cancer: held-out predictions')
    fig.tight_layout(); fig.savefig(out / 'confusion_matrix.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y[test], prob, ax=ax, name=winner)
    ax.plot([0, 1], [0, 1], '--', color='gray'); ax.set_title('Malignant class: held-out ROC')
    fig.tight_layout(); fig.savefig(out / 'roc_curve.png', dpi=160); plt.close(fig)
    print(json.dumps({'model': winner, **metrics}, indent=2))

if __name__ == '__main__':
    main()
