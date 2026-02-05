import re
import numpy as np
from collections import Counter
from typing import List, Dict, Callable


def chunk_transcript_semantic(
    transcript: List[Dict],
    # --- lexical chunking ---
    window_size: int = 24,
    overlap_threshold: float = 0.2,
    silence_gap_threshold: float = 6.0,

    # --- chunk size control ---
    min_chunk_duration: float = 60.0,
    max_chunk_duration: float = 300.0,

    # --- embedding refinement ---
    refine_with_embeddings: bool = True,
    embed_fn: Callable[[str], np.ndarray] = None,
    embedding_window_size: int = 12,
    embedding_similarity_threshold: float = 0.75,

    # --- boundary confidence & overlap policy ---
    hard_boundary_confidence: float = 0.7,
    overlap_boundary_confidence: float = 0.4,
    overlap_duration: float = 30.0,

    stopwords: set = None,
    debug: bool = False,
):
    """
    Automatic semantic chunking with:
    - lexical cohesion
    - optional embedding refinement (cached)
    - boundary confidence scoring
    - adaptive overlapping chunk generation
    """

    if stopwords is None:
        stopwords = set()

    if refine_with_embeddings and embed_fn is None:
        raise ValueError("embed_fn must be provided if refine_with_embeddings=True")

    if debug:
        print(f"[DEBUG] Starting chunking on {len(transcript)} transcript segments")

    # ------------------------------------------------------------------
    # Step 1: Tokenization (robust to ASR noise & no punctuation)
    # ------------------------------------------------------------------

    def tokenize(text):
        # Simple word-level tokens; ASR noise is fine here
        tokens = re.findall(r"\b\w+\b", text.lower())
        return [t for t in tokens if t not in stopwords]

    tokenized = [tokenize(seg["text"]) for seg in transcript]

    # ------------------------------------------------------------------
    # Step 2: Lexical overlap (TextTiling-style)
    # Detects shifts in vocabulary usage → proxy for meaning shifts
    # ------------------------------------------------------------------

    def lexical_overlap(left_tokens, right_tokens):
        if not left_tokens or not right_tokens:
            return 0.0
        left = Counter(left_tokens)
        right = Counter(right_tokens)
        intersection = sum((left & right).values())
        return intersection / max(sum(left.values()), sum(right.values()))

    candidate_boundaries = {}

    for i in range(window_size, len(transcript) - window_size):
        left_tokens = []
        right_tokens = []

        for j in range(i - window_size, i):
            left_tokens.extend(tokenized[j])

        for j in range(i, i + window_size):
            right_tokens.extend(tokenized[j])

        overlap = lexical_overlap(left_tokens, right_tokens)

        if overlap < overlap_threshold:
            prev_end = transcript[i - 1]["start"] + transcript[i - 1]["duration"]
            gap = transcript[i]["start"] - prev_end
            candidate_boundaries[i] = {
                "lexical_overlap": overlap,
                "silence_gap": max(0.0, gap),
                "embedding_similarity": None,
            }
            if debug:
                print(
                    f"[DEBUG] Lexical boundary candidate at index {i} "
                    f"(overlap={overlap:.3f})"
                )

    # ------------------------------------------------------------------
    # Step 3: Timing-based hard signals (long pauses)
    # Pauses often indicate topic resets or Q&A transitions
    # ------------------------------------------------------------------

    for i in range(1, len(transcript)):
        prev_end = transcript[i - 1]["start"] + transcript[i - 1]["duration"]
        gap = transcript[i]["start"] - prev_end

        if gap >= silence_gap_threshold:
            left_tokens = []
            right_tokens = []

            for j in range(max(0, i - window_size), i):
                left_tokens.extend(tokenized[j])

            for j in range(i, min(i + window_size, len(transcript))):
                right_tokens.extend(tokenized[j])

            overlap = lexical_overlap(left_tokens, right_tokens)

            candidate_boundaries.setdefault(i, {
                "lexical_overlap": 1.0,
                "silence_gap": 0.0,
                "embedding_similarity": None,
            })
            candidate_boundaries[i]["silence_gap"] = gap
            candidate_boundaries[i]["lexical_overlap"] = overlap

            if debug:
                print(
                    f"[DEBUG] Silence boundary candidate at index {i} "
                    f"(gap={gap:.2f}s)"
                )

    if debug:
        print(
            f"[DEBUG] Found {len(candidate_boundaries)} candidate chunks"
        )

    # ------------------------------------------------------------------
    # Step 4: Embedding cache
    # Prevents recomputation of identical context windows
    # ------------------------------------------------------------------

    embedding_cache = {}

    def cached_embed(text):
        key = hash(text)
        if key not in embedding_cache:
            if debug:
                print(
                    f"[DEBUG] Computing embedding for text window "
                    f"(~{len(text.split())} tokens)"
                )
            embedding_cache[key] = embed_fn(text)
        else:
            if debug:
                print("[DEBUG] Reusing cached embedding")
        return embedding_cache[key]

    def cosine(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    # ------------------------------------------------------------------
    # Step 5: Embedding refinement at boundaries only
    # Confirms whether lexical boundary reflects semantic separation
    # ------------------------------------------------------------------

    if refine_with_embeddings:
        if debug:
            print(f"[DEBUG] Refining {len(candidate_boundaries)} boundaries with embeddings")

        for i in list(candidate_boundaries.keys()):
            left_start = max(0, i - embedding_window_size)
            right_end = min(len(transcript), i + embedding_window_size)

            left_text = " ".join(seg["text"] for seg in transcript[left_start:i])
            right_text = " ".join(seg["text"] for seg in transcript[i:right_end])

            if left_text.strip() and right_text.strip():
                left_emb = cached_embed(left_text)
                right_emb = cached_embed(right_text)
                sim = cosine(left_emb, right_emb)
            else:
                sim = 0.0  # degenerate case → treat as strong boundary

            candidate_boundaries[i]["embedding_similarity"] = sim

            if debug:
                print(
                    f"[DEBUG] Boundary {i}: embedding similarity = {sim:.3f}"
                )

    # ------------------------------------------------------------------
    # Step 6: Boundary confidence score
    # Turns multiple weak signals into one interpretable value
    # ------------------------------------------------------------------

    def compute_confidence(info):
        """
        Higher confidence → stronger evidence of a real semantic boundary
        """
        lexical_component = 1.0 - info["lexical_overlap"]

        embedding_component = 0.0
        if info["embedding_similarity"] is not None:
            embedding_component = 1.0 - info["embedding_similarity"]

        silence_component = min(info["silence_gap"] / silence_gap_threshold, 1.0)

        # Weighted sum; lexical & embedding dominate
        confidence = (
            0.4 * lexical_component +
            0.4 * embedding_component +
            0.2 * silence_component
        )

        return max(0.0, min(1.0, confidence))

    for i, info in candidate_boundaries.items():
        info["confidence"] = compute_confidence(info)

        if debug:
            print(
                f"[DEBUG] Boundary {i}: confidence={info['confidence']:.3f} "
                f"(lexical={info['lexical_overlap']:.3f}, "
                f"embedding={info['embedding_similarity']}, "
                f"silence={info['silence_gap']:.2f})"
            )

    # ------------------------------------------------------------------
    # Step 7: Build chunks with adaptive overlap policy
    # Strong boundary → hard split
    # Medium boundary → split + overlap
    # Weak boundary → ignore
    # ------------------------------------------------------------------

    chunks = []
    current_chunk = [transcript[0]]
    current_start = transcript[0]["start"]

    for i in range(1, len(transcript)):
        seg = transcript[i]
        current_end = seg["start"] + seg["duration"]
        chunk_duration = current_end - current_start

        boundary_info = candidate_boundaries.get(i)
        confidence = boundary_info["confidence"] if boundary_info else 0.0

        hard_split = (
            confidence >= hard_boundary_confidence and
            chunk_duration >= min_chunk_duration
        )

        overlap_split = (
            overlap_boundary_confidence <= confidence < hard_boundary_confidence and
            chunk_duration >= min_chunk_duration
        )

        force_split = chunk_duration >= max_chunk_duration

        if hard_split or force_split:
            if debug:
                reason = "FORCED" if force_split else "HARD"
                print(
                    f"[DEBUG] {reason} split at index {i} "
                    f"(confidence={confidence:.3f}, duration={chunk_duration:.1f}s)"
                )
            chunks.append(current_chunk)
            current_chunk = [seg]
            current_start = seg["start"]

        elif overlap_split:
            if debug:
                print(
                    f"[DEBUG] OVERLAP split at index {i} "
                    f"(confidence={confidence:.3f}, overlap={overlap_duration}s, duration={chunk_duration:.1f}s)"
                )

            overlap_segments = []
            overlap_start_time = current_end - overlap_duration

            for s in reversed(current_chunk):
                if s["start"] >= overlap_start_time:
                    overlap_segments.insert(0, s)

            chunks.append(current_chunk)
            current_chunk = overlap_segments + [seg]
            current_start = current_chunk[0]["start"]

        else:
            if debug and i in candidate_boundaries:
                print(
                    f"[DEBUG] Ignoring weak boundary at index {i} "
                    f"(confidence={confidence:.3f})"
                )
            current_chunk.append(seg)

    if current_chunk:
        chunks.append(current_chunk)

    if debug:
        print(f"[DEBUG] Finished chunking → {len(chunks)} chunks produced")

    return chunks