from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import load_model, predict

app = FastAPI()

nlp = load_model("models/ner_model")

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict_endpoint(input: TextInput):
    entities = predict(nlp, input.text)  
    return {"entities": entities}