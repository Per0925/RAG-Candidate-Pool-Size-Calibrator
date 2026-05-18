import pandas as pd
from datasets import load_dataset
from helper import chunk_text_by_sentences
from constants import CORPUS_PATH, SAMPLE_PATH, DATA_SIZE
import pickle

def get_dataset(split=f"train[:{DATA_SIZE}]"):
    # --- Ensure output directories exist ---
    # Create parent folders for corpus and sample files if they don't already exist
    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)

    # --- Skip regeneration if files already exist ---
    # Prevents overwriting existing corpus and sample CSV
    if CORPUS_PATH.exists() and SAMPLE_PATH.exists():
        print(f"Corpus already exists at {CORPUS_PATH}, CSV at {SAMPLE_PATH}. Skipping creation.")
        return
    
    # --- Load HotpotQA dataset ---
    dataset = load_dataset("hotpot_qa", "fullwiki", split=split)
    all_chunks = []       # Global list of unique text chunks
    chunk_to_id = {}      # Mapping from chunk text -> global ID
    data = []             # Rows for the final CSV

    # --- Step 1: Build global corpus of chunks ---
    print("Building Global Corpus...")
    for example in dataset:
        for title, sentences in zip(example["context"]["title"], example["context"]["sentences"]):
            full_text = f"{title}: {' '.join(sentences)}"
            # Split passage into overlapping sentence chunks
            passage_chunks = chunk_text_by_sentences(full_text, max_sentences=5, overlap=2)
            for chunk in passage_chunks:
                # Deduplicate: only add new chunks
                if chunk not in chunk_to_id:
                    chunk_to_id[chunk] = len(all_chunks)
                    all_chunks.append(chunk)

    # --- Step 2: Map queries to gold reference chunks ---
    print("Mapping Queries to IDs and Contexts...")
    for example in dataset:
        # Build lookup of supporting facts per title
        gold_map = {t: [] for t in example["supporting_facts"]["title"]}
        for title, sent_id in zip(example["supporting_facts"]["title"], example["supporting_facts"]["sent_id"]):
            gold_map[title].append(sent_id)

        ref_ids = []          # IDs of gold chunks
        ref_contexts = []     # Text of gold chunks
        all_query_contexts = []  # Full passages for this query

        for title, sentences in zip(example["context"]["title"], example["context"]["sentences"]):
            full_text = f"{title}: {' '.join(sentences)}"
            all_query_contexts.append(full_text)

            # Chunk passage and check if any gold sentence is inside            
            passage_chunks = chunk_text_by_sentences(full_text, max_sentences=5, overlap=2)
            gold_sentences = [sentences[sid] for sid in gold_map.get(title, []) if sid < len(sentences)]
            
            for chunk in passage_chunks:
                if any(gs in chunk for gs in gold_sentences):
                    actual_id = chunk_to_id[chunk]
                    ref_ids.append(actual_id)
                    ref_contexts.append(chunk)

        # Save one row per query
        data.append({
            "query": example["question"],
            "contexts": all_query_contexts,           
            "reference_contexts": list(set(ref_contexts)), 
            "reference_context_ids": list(set(ref_ids)), 
            "ground_truth": example["answer"]
        })

    # --- Step 3: Save corpus and query dataset ---
    with open(CORPUS_PATH, "wb") as f:
        pickle.dump(all_chunks, f)
    pd.DataFrame(data).to_csv(SAMPLE_PATH, index=False)
    print(f"Success. Saved {len(data)} samples and {len(all_chunks)} global chunks.")