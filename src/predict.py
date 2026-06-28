import spacy

MODEL_PATH = "models/ner_model"

def load_model(path):
    nlp=spacy.load(path)
    print(f"Model loaded from {path}")
    return nlp

def predict(nlp, text):
    doc = nlp(text)
    for ent in doc.ents:
        print(f"{ent.label_}: {ent.text}")

def main():
    nlp = load_model(MODEL_PATH)
    text = text = "patient has chest pain and shortness of breath. she takes aspirin and zyrtec. diagnosis is allergic rhinitis and hypertension."
    print(f"\nInput: {text}")
    print("\nEntities found:")
    predict(nlp, text)

if __name__ == "__main__":
    main()