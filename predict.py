import pandas as pd
import joblib
import io

def load_model_and_data():
    model = joblib.load("model/classifier.pkl")
    df = pd.read_csv("model/embeddings.csv", index_col=0)
    return model, df

def predict_sequence(sequence, model, df):
    sequence = sequence.upper().strip()
    if sequence not in df.index:
        raise ValueError("Sequence not found in precomputed embeddings.")
    x = df.loc[sequence].values.reshape(1, -1)
    pred = model.predict(x)[0]
    prob = model.predict_proba(x)[0].max()
    return {
        "sequence": sequence,
        "prediction": int(pred),
        "confidence": float(prob)
    }

def predict_batch(file_contents, model, df):
    decoded = file_contents.decode()
    lines = decoded.strip().splitlines()
    results = []
    for line in lines:
        seq = line.strip()
        try:
            result = predict_sequence(seq, model, df)
            results.append(result)
        except:
            results.append({
                "sequence": seq,
                "error": "Not found or invalid format"
            })
    return results
