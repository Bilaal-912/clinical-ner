import spacy

MODEL_PATH = "models/ner_model"

def load_model(path):
    nlp=spacy.load(path)
    print(f"Model loaded from {path}")
    return nlp

def predict(nlp, text):
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append({
            "label": ent.label_,
            "text": ent.text,
            "start": ent.start_char,
            "end": ent.end_char
        })
    return entities

def main():
    nlp = load_model(MODEL_PATH)
    text = "patient has chest pain and shortness of breath. she takes aspirin and zyrtec. diagnosis is allergic rhinitis and hypertension."
    print(f"\nInput: {text}")
    entities = predict(nlp, text)
    print("\nEntities found:")
    for ent in entities:
        print(f"{ent['label']}: {ent['text']}")

if __name__ == "__main__":
    main()