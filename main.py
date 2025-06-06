from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from utils.predict import load_model_and_data, predict_sequence, predict_batch

app = FastAPI()
model, X_features, sequence_ids = load_model_and_data()

class SequenceRequest(BaseModel):
    sequence: str

@app.post("/predict")
def predict(seq: SequenceRequest):
    try:
        result = predict_sequence(seq.sequence, model, X_features, sequence_ids)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/batch_predict")
async def batch_predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        result = predict_batch(contents, model, X_features, sequence_ids)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
