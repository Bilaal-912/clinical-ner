from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import load_model, predict
from src.db import get_entities

app = FastAPI()

nlp = load_model("models/ner_model")

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict_endpoint(input: TextInput):
    entities = predict(nlp, input.text)  
    return {"entities": entities}

@app.get("/entities")
def get_entities_endpoint():
    entities = get_entities()
    return {"entities": entities}

@app.get("/entities/{label}")
def get_entities_by_label(label: str):
    entities = get_entities(label)
    return {"entities": entities}
