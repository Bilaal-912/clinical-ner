import streamlit as st
from db import get_entities

st.set_page_config(page_title="Clinical NER Dashboard", layout="wide")
st.title("Clinical NER Dashboard")
st.write("Search and filter extracted clinical entities from medical transcriptions.")
st.sidebar.header("Filters")
label_filter = st.sidebar.selectbox(
    "Filter by entity type",
    ["All", "DRUG", "DOSAGE", "SYMPTOM", "DIAGNOSIS", "BODY_PART"]
)
if label_filter == "All":
    rows = get_entities()
else:
    rows = get_entities(label=label_filter)

st.subheader(f"Entities found: {len(rows)}")
if rows:
    st.table([{
        "ID": row[0],
        "Document ID": row[1],
        "Entity Text": row[2],
        "Label": row[3],
        "Start": row[4],
        "End": row[5]
    } for row in rows])
else:
    st.write("No entities found.")