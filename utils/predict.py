import numpy as np
import joblib

def load_model_and_data():
    model = joblib.load("model/classifier.pkl")
    features = np.load("model/features.npz")

    X1 = features["X1"]
    X2 = features["X2"]
    X3 = features["X3"]

    X1_pooled = X1.mean(axis=1)
    X = np.concatenate([X1_pooled, X2, X3], axis=1)

    with open("model/sequence_ids.txt") as f:
        ids = [line.strip() for line in f]

    return model, X, ids

def predict_sequence(sequence, model, X, ids):
    sequence = sequence.upper().strip()
    if sequence not in ids:
        raise ValueError("Sequence not found in feature set.")
    idx = ids.index(sequence)
    x = X[idx].reshape(1, -1)
    pred = model.predict(x)[0]
    prob = model.predict_proba(x)[0].max()
    return {
        "sequence": sequence,
        "prediction": int(pred),
        "confidence": float(prob)
    }

def predict_batch(file_contents, model, X, ids):
    decoded = file_contents.decode()
    lines = decoded.strip().splitlines()
    results = []
    for line in lines:
        seq = line.strip()
        try:
            result = predict_sequence(seq, model, X, ids)
            results.append(result)
        except:
            results.append({
                "sequence": seq,
                "error": "Not found or invalid format"
            })
    return results
