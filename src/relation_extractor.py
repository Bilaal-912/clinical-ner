import spacy

MODEL_PATH = "models/ner_model"

def load_model(path):
    nlp = spacy.load(path)
    print(f"Model loaded from {path}")
    return nlp

def extract_relations(doc,label_a,label_b,max_distance=50):
    relations=[]
    entities_a = [ent for ent in doc.ents if ent.label_ == label_a]
    entities_b = [ent for ent in doc.ents if ent.label_ == label_b]
    for ent_a in entities_a:
        for ent_b in entities_b:
            distance=ent_b.start_char-ent_a.end_char
            if 0 <= distance <= max_distance:
                relations.append((ent_a.text, label_a, ent_b.text, label_b))
    return relations

def main():
    nlp = load_model(MODEL_PATH)
    text ="she used claritin, and zyrtec. allegra also. allergic rhinitis."
    doc = nlp(text)
    print(f"\nInput: {text}\n")
    print("Entities found:")
    for ent in doc.ents:
        print(f"  {ent.label_}: {ent.text}")

    drug_dosage = extract_relations(doc, "DRUG", "DOSAGE")
    symptom_diagnosis = extract_relations(doc, "SYMPTOM", "DIAGNOSIS")

    print("\nDRUG -> DOSAGE relations:")
    for rel in drug_dosage:
        print(f"  {rel[0]} -> {rel[2]}")

    print("\nSYMPTOM -> DIAGNOSIS relations:")
    for rel in symptom_diagnosis:
        print(f"  {rel[0]} -> {rel[2]}")

if __name__ == "__main__":
    main()