import json
import os
import random 
from spacy.tokens import DocBin
import spacy

ANNOTATIONS_PATH = "data/processed/annotations.json"    #where exports are rn
OUTPUT_DIR = "data/processed"         #where the output will be 
TRAIN_RATIO = 0.8                       # this means 80% of 30 documents go to training, 20% go to testing
RANDOM_SEED = 42                        #to add randomness to data

def load_annotations(filepath):
    with open(filepath,"r",encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} annotated documents.")
    return data

def convert_task(task):
    text=task["data"]["text"].strip()
    if not text:
        return None
    annotations=task["annotations"][0]
    result=annotations["result"]
    entities=[]
    for span in result:
        start=span["value"]["start"]
        end=span["value"]["end"]
        label=span["value"]["labels"][0]
        entities.append((start,end,label))
    return (text, {"entities": entities})

def has_overlap(entities):
    sorted_ents=sorted(entities,key=lambda e:e[0])
    for i in range(len(sorted_ents) - 1):
        _, end_a, _ = sorted_ents[i]
        start_b, _, _ = sorted_ents[i + 1]
        if start_b < end_a:
            return True
    return False

def remove_overlaps(entities):
    sorted_ents = sorted(entities, key=lambda e: e[0])
    cleaned=[]
    for ent in sorted_ents:
        start,end,label=ent
        conflict=False
        for kept_start,kept_end,_ in cleaned:
            if start<kept_end and end>kept_start:
                conflict=True
                break
            if not conflict:
                cleaned.append(ent)
    return cleaned

def build_training_data(raw_data):
    all_examples = []
    for task in raw_data:
        converted = convert_task(task)
        if converted is None:
            continue
        text, annotation_dict = converted
        entities = annotation_dict["entities"]
        if has_overlap(entities):
            entities = remove_overlaps(entities)
        all_examples.append((text, {"entities": entities}))
    random.seed(RANDOM_SEED)
    random.shuffle(all_examples)
    split_idx = int(len(all_examples) * TRAIN_RATIO)
    train_data = all_examples[:split_idx]
    dev_data = all_examples[split_idx:]
    return train_data, dev_data

def save_spacy_format(data, output_path, nlp):
    doc_bin = DocBin()
    for text, annotations in data:
        doc = nlp.make_doc(text)
        ents = []
        for start, end, label in annotations["entities"]:
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is None:
                continue
            ents.append(span)
            doc.ents = ents
        doc_bin.add(doc)
    doc_bin.to_disk(output_path)
    print(f"Saved {len(data)} docs to {output_path}")

def main():
    nlp = spacy.blank("en")
    raw_data = load_annotations(ANNOTATIONS_PATH)
    train_data, dev_data = build_training_data(raw_data)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_spacy_format(train_data, os.path.join(OUTPUT_DIR, "train.spacy"), nlp)
    save_spacy_format(dev_data, os.path.join(OUTPUT_DIR, "dev.spacy"), nlp)
    print("Done!")

if __name__ == "__main__":
    main()