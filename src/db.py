import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

def get_connection():
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )
    return conn

def save_entities(document_id, entities):
    conn = get_connection()
    cursor = conn.cursor()
    for entity in entities:
        cursor.execute(
            "INSERT INTO entities (document_id, entity_text, entity_label, start_char, end_char) VALUES (%s, %s, %s, %s, %s)",
            (document_id, entity["text"], entity["label"], entity["start"], entity["end"])
        )
    conn.commit()
    cursor.close()
    conn.close()
    print(f"Saved {len(entities)} entities for document {document_id}")

def get_entities(label=None):
    conn = get_connection()
    cursor = conn.cursor()
    if label:
        cursor.execute("SELECT * FROM entities WHERE entity_label = %s", (label,))
    else:
        cursor.execute("SELECT * FROM entities")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows

def main():
    test_entities = [
        {"text": "aspirin", "label": "DRUG", "start": 17, "end": 24},
        {"text": "75mg", "label": "DOSAGE", "start": 25, "end": 29},
        {"text": "chest pain", "label": "SYMPTOM", "start": 34, "end": 44}
    ]
    save_entities(1, test_entities)
    rows = get_entities()
    print("\nAll entities in database:")
    for row in rows:
        print(row)

if __name__ == "__main__":
    main()