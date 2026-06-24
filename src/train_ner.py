import os
import spacy
from spacy.tokens import DocBin
from spacy.training import Example

TRAIN_PATH="data/processed/train.spacy"
DEV_PATH="data/processed/dev.spacy"
MODEL_OUTPUT="models/ner_model"
N_ITER = 20

def load_data(path,nlp):
    doc_bin=DocBin().from_disk(path)
    docs=list(doc_bin.get_docs(nlp.vocab))
    print(f"Loaded {len(docs)} documents from {path}")
    return docs

def create_model(labels):
    nlp=spacy.blank("en")
    ner=nlp.add_pipe("ner")
    for label in labels:
        ner.add_label(label)
    return nlp

def train_model(nlp, train_docs, dev_docs):
    optimizer = nlp.begin_training()
    for iteration in range(N_ITER):
        losses = {}
        examples = []
        for doc in train_docs:
            example = Example.from_dict(doc, {"entities": [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]})
            examples.append(example)
        nlp.update(examples, sgd=optimizer, losses=losses)
        print(f"Iteration {iteration + 1}, Loss: {losses['ner']:.4f}")
        dev_examples = [Example.from_dict(doc, {"entities": [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]}) for doc in dev_docs]
        scores = nlp.evaluate(dev_examples)
        print(f"F1 Score: {scores['ents_f']:.4f}")
    return nlp

def main():
    labels=["DRUG", "DOSAGE", "SYMPTOM", "DIAGNOSIS", "BODY_PART"]
    nlp=create_model(labels)
    train_docs=load_data(TRAIN_PATH,nlp)
    dev_docs=load_data(DEV_PATH,nlp)
    trained_nlp = train_model(nlp, train_docs, dev_docs)
    os.makedirs(MODEL_OUTPUT, exist_ok=True)
    trained_nlp.to_disk(MODEL_OUTPUT)
    print(f"Model saved to {MODEL_OUTPUT}")

if __name__ == "__main__":
    main()