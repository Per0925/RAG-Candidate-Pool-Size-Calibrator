from nltk.tokenize import sent_tokenize
import numpy as np
import faiss
import math
import time
import ast
import pandas as pd
from constants import TRAIN_RATIO
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import NonLLMContextRecall, IDBasedContextPrecision

def chunk_text_by_sentences(text, max_sentences=5, overlap=2):
    """
    Split text into overlapping chunks of sentences.

    Args:
        text (str): Input passage to be chunked.
        max_sentences (int): Maximum number of sentences per chunk.
        overlap (int): Number of sentences to overlap between consecutive chunks.

    Returns:
        list[str]: List of text chunks.

    Notes:
        - Overlap ensures context continuity across chunks.
        - Example: with max_sentences=5 and overlap=2,
          chunk1 = sentences[0:5], chunk2 = sentences[3:8], etc.
    """
    sentences = sent_tokenize(text)
    chunks = []
    start = 0
    while start < len(sentences):
        end = min(start + max_sentences, len(sentences))
        chunk = " ".join(sentences[start:end])
        chunks.append(chunk)
        start += max_sentences - overlap
    return chunks

def build_faiss_index(all_context_embs):
    """
    Build a FAISS index for dense retrieval.

    Args:
        all_context_embs (np.ndarray): Matrix of embeddings for all corpus chunks.

    Returns:
        (faiss.Index, np.ndarray): FAISS index object and normalized embeddings.

    Notes:
        - Embeddings are L2-normalized so IndexFlatIP behaves like cosine similarity.
        - IndexFlatIP is a simple inner-product index suitable for small/medium corpora.
    """
    all_context_embs = all_context_embs / np.linalg.norm(all_context_embs, axis=1, keepdims=True)
    dim = all_context_embs.shape[1]
    index = faiss.IndexFlatIP(dim) 
    index.add(all_context_embs)
    return index, all_context_embs

async def evaluate_retrieval_step(query, indices, all_contexts, dataset):
    """
    Evaluate retrieval quality for a given set of candidate indices.

    Args:
        query (str): User query text.
        indices (list[int]): Candidate indices retrieved from FAISS/BM25.
        all_contexts (list[str]): Corpus chunks corresponding to indices.
        dataset (dict): Ground truth sample with reference contexts/IDs.

    Returns:
        (float, float): Precision score, Recall score.

    Notes:
        - Uses RAGAS metrics:
          * IDBasedContextPrecision: measures how many retrieved IDs match gold IDs.
          * NonLLMContextRecall: measures how much gold context is covered.
        - Returns 0.0 if no reference contexts/IDs are available.
    """
    texts = [all_contexts[idx] for idx in indices]
    ids = [str(idx) for idx in indices]
    
    sample = SingleTurnSample(
        user_input=query,
        retrieved_contexts=texts,
        retrieved_context_ids=ids,
        reference_contexts=dataset.get("reference_contexts", []),
        reference_context_ids=[str(i) for i in dataset.get("reference_context_ids", [])]
    )
    
    p_score = await IDBasedContextPrecision().single_turn_ascore(sample) if sample.reference_context_ids else 0.0
    r_score = await NonLLMContextRecall().single_turn_ascore(sample) if sample.reference_contexts else 0.0
    return p_score, r_score

def get_utility(p, r, latency, threshold=1.0, k=5):
    """
    Compute a latency-aware utility score for a retrieval step.

    Args:
        p (float): Precision score.
        r (float): Recall score.
        latency (float): Time taken for the step (seconds).
        threshold (float): Latency threshold where penalty starts.
        k (int): Steepness of sigmoid penalty curve.

    Returns:
        float: Utility score (F1 penalized by latency).

    Notes:
        - F1 captures balance between precision and recall.
        - Sigmoid penalty reduces utility as latency exceeds threshold.
        - This encourages methods that are both accurate and efficient.
    """
    # 1. Standard F1 for retrieval quality
    f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
    
    # 2. Latency penalty using sigmoid decay
    # Near threshold → penalty ~1.0 (no effect)
    # Far above threshold → penalty drops sharply
    resource_penalty = 1 / (1 + math.exp(k * (latency - threshold)))
    
    return f1 * resource_penalty

def run_step_2(should_run_step_2, candidate_indices, all_contexts, query, cross_encoder, top_n):
    """
    Refined Step 2: Selective Reranking

    Args:
        should_run_step_2 (bool): Whether reranking should be performed (gate open/closed).
        candidate_indices (list[int]): List of candidate chunk indices from previous step.
        all_contexts (list[str]): Corpus chunks corresponding to indices.
        query (str): User query text.
        cross_encoder: Cross-encoder model used for reranking.
        top_n (int): Number of top candidates to rerank.

    Returns:
        (list[int], float): Reranked list of indices + latency of reranking step.
    """

    # --- Gate check ---
    # If reranking is disabled (e.g., confidence gate closed), return original candidates unchanged.
    if not should_run_step_2:
        return candidate_indices, 0.0

    # --- Step 1: Select subset to rerank ---
    # Only rerank the top_n candidates; keep the rest in original order.
    t2_start = time.perf_counter()
    to_rerank = candidate_indices[:top_n]
    the_rest = candidate_indices[top_n:]

    # --- Step 2: Score candidates with cross-encoder ---
    scored_results = []
    for idx in to_rerank:
        # Validate index and fetch corresponding text
        if isinstance(idx, int) and 0 <= idx < len(all_contexts):
            context_text = all_contexts[idx]
        else:
            # Fallback: empty string if index is invalid
            context_text = ""
        
        # Predict relevance score for (query, context) pair
        # Ensure score is cast to float (some models return arrays)
        score = float(cross_encoder.predict([(query, context_text)]))
        scored_results.append((score, idx))

    # --- Step 3: Sort candidates by score ---
    # Higher scores = more relevant
    scored_results.sort(key=lambda x: x[0], reverse=True)

    # --- Step 4: Reconstruct full candidate list ---
    # Place reranked top_n first, followed by untouched remainder
    reranked_indices = [idx for score, idx in scored_results]
    t2_end = time.perf_counter()
    t2 = t2_end - t2_start

    return reranked_indices + the_rest, t2


def get_best_rank(retrieved_indices, truth_ids):
    """
    Find the best (lowest) rank position of any ground-truth ID in the retrieved list.

    Args:
        retrieved_indices (list[int]): Candidate indices returned by retrieval.
        truth_ids (list[int] or str): Ground-truth indices (may be list or stringified list).

    Returns:
        int or float: Best rank (1-based). If none found, returns infinity.
    """

    # --- Normalize truth_ids input ---
    # Convert string "[1, 2]" to list if necessary
    if isinstance(truth_ids, str):
        try:
            truth_ids = ast.literal_eval(truth_ids)
        except:
            truth_ids = [int(truth_ids)]
    
    # Ensure truth_ids is always a list
    if not isinstance(truth_ids, list):
        truth_ids = [truth_ids]

    # --- Find ranks of all truth IDs ---
    ranks = []
    for t_id in truth_ids:
        try:
            # index() returns 0-based position → add 1 for human-readable rank
            ranks.append(retrieved_indices.index(t_id) + 1)
        except ValueError:
            # Skip if truth ID not found in retrieved list
            continue

    # --- Return best rank ---
    # If at least one truth ID found, return lowest rank; else infinity
    return min(ranks) if ranks else float('inf')

def get_tail_mean(series, q=0.95):
    """
    Compute the mean of the 'tail' values in a series.

    Args:
        series (pd.Series): Numeric series (e.g., ranks or scores).
        q (float): Quantile threshold (default = 0.95).

    Returns:
        float: Mean of values greater than or equal to the quantile threshold.

    Notes:
        - Useful for analyzing the behavior of the worst-performing samples.
        - Example: q=0.95 → take the top 5% of values and compute their mean.
    """
    # --- Step 1: Find quantile cutoff ---
    threshold = series.quantile(q)

    # --- Step 2: Select tail values ---
    # Use >= to include all values at or above the threshold
    tail = series[series >= threshold]

    # --- Step 3: Return mean of tail ---
    return tail.mean()

def calculate_sigmoid_k(conf, k_max, midpoint=0.75, steepness=50):
    """
    Calculate dynamic rerank depth (candidate pool size) based on embedding confidence.

    Args:
        conf (float): Embedding confidence score (0–1).
        k_max (int): Maximum pool size (upper bound).
        midpoint (float): Confidence value where pool size is ~k_max/2.
        steepness (int): Controls how sharply pool size decreases as confidence increases.

    Returns:
        int: Recommended candidate pool size (rounded, non-negative).

    Notes:
        - Uses a sigmoid curve: low confidence → larger pool, high confidence → smaller pool.
        - Ensures pool size never drops below 0.
        - Example: conf=0.6 → large pool (~22); conf=0.9 → small pool (~0–1).
    """
    # --- Step 1: Apply sigmoid formula ---
    # As confidence increases, denominator grows → k decreases
    k = k_max / (1 + np.exp(steepness * (conf - midpoint)))

    # --- Step 2: Round and clip ---
    # Round to nearest integer and ensure non-negative
    return np.maximum(0, np.round(k)).astype(int)

def get_binned_rerank_dataset(filename, threshold):
    """
    Recommend a rerank pool size based on embedding confidence.

    Workflow:
    1. Load evaluation results and restrict to training portion.
    2. Clean rank values (replace infinities with a penalty).
    3. Define "buried truth" (truth not in top-N).
    4. Bin queries by embedding confidence.
    5. Compute rank statistics per bin (burial probability, quantiles, etc.).
    6. Map each bin to an optimal rerank depth using a sigmoid model.
    7. Return the rerank depth for the given confidence.

    Args:
        conf (float): Embedding confidence score for the current query.
        filename (str): Path to CSV file containing evaluation results.
        threshold (int): Rank cutoff for defining "buried" truth (default = 10).
        print_stats (bool): If True, print rank statistics for inspection.

    Returns:
        dict: Mapping from confidence intervals to optimal rerank pool sizes.
        float: Cap value for low-confidence bin (used to limit max rerank depth).
    """

    # --- Step 1: Load dataset and restrict to training portion ---
    df = pd.read_csv(filename)
    df = df[:int(len(df) * TRAIN_RATIO)]

    # --- Step 2: Clean rank values ---
    # Replace infinite ranks (truth not found) with 50 as a penalty
    df['rank_clean'] = df['rank_of_truth_dense'].replace([np.inf, -np.inf], 50)

    # --- Step 3: Define "buried truth" ---
    # A query is considered "buried" if the truth is ranked below the threshold
    df['is_buried'] = (df['rank_clean'] > threshold).astype(int)

    # --- Step 4: Bin embedding confidence ---
    # Focus on confidence range [0.6, 0.95] where most queries lie
    bins = np.linspace(0.6, 0.95, 8)
    df['conf_bin'] = pd.cut(df['s1_emb_conf'], bins=bins)

    # --- Step 5: Compute rank statistics per bin ---
    # Includes burial probability, quantiles, tail mean, median, and sample size
    rank_stats = df.groupby('conf_bin', observed=True)['rank_clean'].agg([
        ('probability_of_burial', 'mean'),
        ('90th_quantile', lambda x: x.quantile(0.9)),
        ('tail_mean_90', get_tail_mean),
        ('median', 'median'),
        ('sample_size', 'count')
    ])

    # For the [0.6, 0.65] confidence interval, also record the 90th percentile
    # of rank_clean values. This acts as a cap for rerank depth in that bin.
    cap_value = int(rank_stats.loc[pd.Interval(left=0.6, right=0.65, closed='right'), '90th_quantile'])+5

    # Find the overall median confidence across the dataset (not per bin).
    median_conf = df['s1_emb_conf'].median()

    # --- Step 6: Map bins to rerank depths ---
    # For each confidence bin, compute the "optimal rerank pool size"
    # using a sigmoid function applied to the bin center.
    bin_centers = rank_stats.index.categories.mid
    rank_stats['optimal_rerank_top'] = [calculate_sigmoid_k(c, k_max=cap_value, midpoint=median_conf) for c in bin_centers]

    # --- Step 7: Build rerank map and extract cap value ---
    # Convert the optimal rerank sizes into a dictionary keyed by confidence interval.
    rerank_map = rank_stats['optimal_rerank_top'].to_dict()

    # --- Step 8: Return results ---
    # Return both the rerank map (all bins → pool sizes) and the cap value
    # for the low-confidence interval.
    return rerank_map, cap_value

def get_binned_rerank_result(rerank_map, conf, cap_value):
    """
    Given a confidence score, return the recommended rerank pool size.

    Args:
        rerank_map (dict): Mapping from confidence intervals → optimal rerank sizes.
        conf (float): Embedding confidence score for the current query.
        cap_value (int or float): Maximum allowed rerank depth (cap for low-confidence bin).

    Returns:
        int: Final rerank pool size for this confidence.
    """

    # --- Step 1: Match confidence to its bin ---
    # Iterate through the confidence intervals in rerank_map.
    # If the current confidence falls inside an interval, return the mapped value.
    # Apply a cap so the rerank size never exceeds cap_value.
    for interval, value in rerank_map.items():
        if conf in interval:
            return min(value, cap_value)

    # --- Step 2: Handle out-of-range confidence values ---
    # If confidence is below 0.6, default to the cap_value (large pool).
    # If confidence is above 0.95, default to 0 (skip reranking).
    return cap_value if conf <= 0.6 else 0
