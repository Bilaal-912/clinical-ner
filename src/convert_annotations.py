import json
import os

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


