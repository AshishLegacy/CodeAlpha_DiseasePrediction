"""Predict from one JSON object with the 30 named dataset measurements."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('patient_json', type=Path)
    args = parser.parse_args()
    bundle = joblib.load(Path(__file__).resolve().parent / 'results/model.joblib')
    patient = json.loads(args.patient_json.read_text())
    expected = set(bundle['feature_names'])
    if set(patient) != expected:
        raise ValueError(f"Feature mismatch: missing={expected - set(patient)}, extra={set(patient) - expected}")
    row = np.array([[patient[name] for name in bundle['feature_names']]], dtype=float)
    if not np.isfinite(row).all():
        raise ValueError('All 30 values must be finite numeric measurements.')
    pred = int(bundle['model'].predict(row)[0])
    probability = float(bundle['model'].predict_proba(row)[0, 1])
    print(json.dumps({'class': bundle['labels'][pred], 'model_probability_malignant': probability,
                      'use': 'Educational dataset demonstration; not a clinical diagnosis.'}, indent=2))

if __name__ == '__main__':
    main()
