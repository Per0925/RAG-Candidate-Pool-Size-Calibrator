import pandas as pd
import numpy as np
import ast
import os
import pickle
import faiss
import time
import traceback
from tqdm import tqdm
from helper import (
    build_faiss_index,
    evaluate_retrieval_step,
    get_utility,
    run_step_2,
    get_best_rank,
    get_binned_rerank_dataset,
    get_binned_rerank_result
)
from constants import (
    EMBEDDER,
    CROSS_ENCODER
)

async def get_rag_results(dataset, index, all_contexts, mode, emb_conf_threshold, top_n, binned_result_dataset=None, cap_value=None, max_rerank_top=50, k=3):
    """
    Run the retrieval pipeline for a single query and collect evaluation metrics.

    Workflow:
    1. Perform dense retrieval using FAISS (Step 1).
    2. Compute embedding confidence and evaluate precision/recall/utility.
    3. Decide reranking strategy (fixed vs. dynamic) based on mode and confidence.
    4. If reranking is enabled, rerank candidates with a cross-encoder (Step 2).
    5. Collect and return metrics for both steps.

    Args:
        dataset (dict): Query sample containing 'query', 'query_emb', and references.
        index (faiss.Index): FAISS index built over the corpus.
        all_contexts (list): List of all text chunks in the corpus.
        mode (str): Reranking strategy ('fixed' or 'dynamic').
        emb_conf_threshold (float): Confidence threshold for gating reranking.
        top_n (int): Candidate pool size (fixed) or threshold (dynamic).
        binned_result_dataset (dict, optional): Confidence bin mapping for dynamic mode.
        cap_value (int, optional): Maximum candidate pool size for dynamic mode.
        max_rerank_top (int): Max number of candidates to retrieve from FAISS.
        k (int): Number of top results to evaluate precision/recall on.
    """
    query = dataset["query"]
    step_metrics = {}

    # --- STEP 1: INITIAL RETRIEVAL (Dense FAISS search) ---
    t1_start = time.perf_counter()
    q_emb = np.array(dataset["query_emb"]).reshape(1, -1)
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    D, I = index.search(q_emb, min(max_rerank_top, len(all_contexts)))
    emb_confidence = float(np.clip(D[0][0], 0, 1))
    
    dense_indices = I[0].tolist()
    
    t1_end = time.perf_counter()
    t1 = t1_end - t1_start

    # Evaluate retrieval quality for Step 
    s1_p, s1_r = await evaluate_retrieval_step(query, dense_indices[:k], all_contexts, dataset)
    t_features_start = time.perf_counter()
    s1_utility = get_utility(s1_p, s1_r, t1)

    step_metrics["step1"] = {"latency": t1, "precision": s1_p, "recall": s1_r, "utility": s1_utility}
    step_metrics["s1_emb_conf"] = emb_confidence

    # Rank of ground truth in dense retrieval
    truth_id = dataset["reference_context_ids"] if dataset.get("reference_context_ids") else None
    rank_dense = get_best_rank(dense_indices, truth_id)
    step_metrics["rank_of_truth_dense"] = rank_dense

    t_features_end = time.perf_counter()
    t_feature = t_features_end - t_features_start

    # --- STEP 2: RERANKING (Cross-encoder) ---
    # Decide reranking strategy based on mode
    should_run_step_2 = emb_conf_threshold is not None and emb_confidence < emb_conf_threshold
    if mode == "fixed":
        # Fixed top-N candidates
        top_n = top_n
    elif mode == "dynamic":
        # Dynamic pool size
        top_n = max(k-1, min(cap_value, get_binned_rerank_result(binned_result_dataset, emb_confidence, cap_value)))

    # Perform reranking if gate is open
    final_indices, t2 = run_step_2(should_run_step_2, dense_indices, all_contexts, query, CROSS_ENCODER, top_n)

    # Evaluate retrieval quality after reranking
    s2_p, s2_r = await evaluate_retrieval_step(query, final_indices[:k], all_contexts, dataset)
    s2_utility = get_utility(s2_p, s2_r, t1 + t2)

    step_metrics["feature_latency"] = t_feature
    step_metrics["step2"] = {"latency": t1 + t2, "precision": s2_p, "recall": s2_r, "utility": s2_utility}

    return {
        "step_metrics": step_metrics
    }

async def process_sample(sample, index, all_contexts, mode, emb_conf_threshold, top_n, binned_result_dataset, cap_value):
    """
    Process a single query sample through the retrieval pipeline.

    Workflow:
    1. Encode the query into a dense embedding using the SentenceTransformer.
    2. Normalize the embedding for FAISS (cosine similarity).
    3. Run the retrieval pipeline (dense retrieval + optional rerank).
    4. Collect and return evaluation metrics for analysis.

    Args:
        sample (dict): One row from the dataset containing query, contexts, and ground truth.
        index (faiss.Index): FAISS index built over the global corpus.
        all_contexts (list): List of all text chunks in the corpus.
        mode (str): Evaluation mode ('fixed' or 'dynamic').
        emb_conf_threshold (float): Confidence threshold for gating reranking.
        top_n (int): Candidate pool size or threshold depending on mode.
        binned_result_dataset (dict): Confidence bin mapping for dynamic mode.
        cap_value (int): Maximum candidate pool size for dynamic mode.
    """
    # --- Step 1: Encode query ---
    # Generate the query embedding on-the-fly to ensure the latest model weights are used.
    query_text = sample["query"]
    q_emb = EMBEDDER.encode(f"Represent this sentence for searching relevant passages: {query_text}", convert_to_numpy=True)
    
    # --- Step 2: Normalize embedding ---
    # Normalization ensures FAISS IndexFlatIP behaves like cosine similarity.
    faiss.normalize_L2(q_emb.reshape(1, -1)) 
    sample["query_emb"] = q_emb

    # --- Step 3: Run retrieval pipeline ---
    # Pass the sample through the RAG pipeline (dense retrieval + optional rerank).
    res = await get_rag_results(sample, index, all_contexts, mode, emb_conf_threshold, top_n, binned_result_dataset, cap_value, max_rerank_top=50, k=3)
    metrics = res["step_metrics"]

    # --- Step 4: Collect metrics ---
    # Return a dictionary of key evaluation metrics for later aggregation.
    return {
        "query": query_text,                       # Original query text
        "expected": sample["ground_truth"],        # Ground truth answer
        "s1_lat": metrics["step1"]["latency"],     # Step 1 latency
        "s1_prec": metrics["step1"]["precision"],  # Step 1 precision
        "s1_rec": metrics["step1"]["recall"],      # Step 1 recall
        "s1_util": metrics["step1"]["utility"],    # Step 1 utility score
        "s2_lat": metrics["step2"]["latency"],     # Step 2 latency
        "s2_prec": metrics["step2"]["precision"],  # Step 2 precision
        "s2_rec": metrics["step2"]["recall"],      # Step 2 recall
        "s2_util": metrics["step2"]["utility"],    # Step 2 utility score
        "feature_latency": metrics["feature_latency"],  # Time spent computing features
        "s1_emb_conf": metrics["s1_emb_conf"],     # Embedding confidence score
        "rank_of_truth_dense": metrics["rank_of_truth_dense"],  # Rank of gold chunk in dense retrieval
    }

async def main_pipeline(dataset_file="src/dataset.csv", corpus_file="src/global_corpus.pkl", mode="dynamic", emb_conf_threshold=1, top_n=20):
    """
    Evaluate the retrieval pipeline on a dataset using a given corpus.

    Workflow:
    1. Load the static corpus (global chunks).
    2. Load dataset of queries and parse reference contexts.
    3. Build FAISS index over corpus embeddings.
    4. Resume from existing results if available.
    5. If dynamic mode, load confidence bin statistics and cap value.
    6. Iterate through samples, run pipeline, and save results incrementally.

    Args:
        dataset_file (str): Path to the dataset CSV containing queries and references.
        corpus_file (str): Path to the global corpus pickle file.
        mode (str): Evaluation mode ('fixed' or 'dynamic').
        emb_conf_threshold (float): Confidence threshold for gating reranking.
        top_n (int): Candidate pool size or threshold depending on mode.
    """
    # --- Step 1: Load the static corpus ---
    if not os.path.exists(corpus_file):
        raise FileNotFoundError(f"Run prepare_and_save_corpus first to create {corpus_file}")
    with open(corpus_file, "rb") as f:
        all_chunks = pickle.load(f)
        print(all_chunks[:5])  # Print first 5 chunks for verification
    print(f"Loaded {len(all_chunks)} chunks from static corpus.")

    # --- Step 2: Load dataset of queries ---
    df = pd.read_csv(dataset_file)  

    # Convert stringified lists back to Python lists
    for col in ["contexts", "reference_contexts", "reference_context_ids"]:
        df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x)

    # --- Step 3: Build FAISS index over corpus ---
    print("Encoding embeddings...")
    all_context_embs = EMBEDDER.encode(all_chunks, show_progress_bar=True, convert_to_numpy=True)
    index, _ = build_faiss_index(all_context_embs)

    # --- Step 4: Resume logic ---
    result_file = f"src/results_{mode}_{emb_conf_threshold}_{top_n}.csv"
    start_idx = 0
    if os.path.exists(result_file):
        try:
            processed_df = pd.read_csv(result_file, on_bad_lines='skip')
            start_idx = len(processed_df)
            print(f"Resuming from index {start_idx}...")
        except Exception:
            pass

    eval_samples = df.iloc[start_idx:].to_dict('records')

    if mode == "dynamic":
        binned_result_dataset, cap_value = get_binned_rerank_dataset(filename=f"src/results_fixed_{emb_conf_threshold}_{top_n}.csv", threshold=top_n)
    else:
        binned_result_dataset, cap_value = None, None

    # --- Step 5: Evaluation loop ---
    print(f"Starting Evaluation for {len(eval_samples)} samples...")
    print(f"Mode: {mode}, Threshold: {emb_conf_threshold}, Top-N: {top_n}")
    for sample in tqdm(eval_samples, desc="Processing"):
        try:
            # Run retrieval + reranking pipeline for one sample
            res = await process_sample(sample, index, all_chunks, mode, emb_conf_threshold, top_n, binned_result_dataset, cap_value)
            # Save results immediately to avoid losing progress
            res_df = pd.DataFrame([res])
            res_df.to_csv(result_file, mode='a', index=False, header=not os.path.exists(result_file))
            
        except Exception as e:
            # Log errors without stopping the loop
            print(f"Error on query: {sample.get('query', 'Unknown')} | Error: {e}")
            print(f"FAILED QUERY: {sample.get('query')}")
            print(f"QUERY LEN: {len(sample.get('query', ''))}")
            traceback.print_exc()
            continue

    print(f"Complete. Results in {result_file}")


