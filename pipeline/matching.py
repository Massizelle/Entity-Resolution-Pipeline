try:
	from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional runtime dependency may fail to load
	SentenceTransformer = None
import contextlib
import io
import logging
try:
	from sklearn.feature_extraction.text import TfidfVectorizer
	from sklearn.metrics.pairwise import cosine_similarity
except Exception:  # pragma: no cover - optional runtime dependency may fail to load
	TfidfVectorizer = None
	cosine_similarity = None
import pandas as pd
import numpy as np
import os
import sys
import math
import json
import pickle
import time
from collections import Counter
from pipeline.data_ingestion import DATASET_REGISTRY
from pipeline.clustering import materialize_incremental_clusters
from pipeline.adaptive_rescue import apply_adaptive_rescue

PARTIAL_SCORE_COLUMNS = [
	"id_A",
	"id_B",
	"certificate_score",
	"evidence_count",
	"witness_path",
	"region_size",
	"jaccard_score",
	"tfidf_score",
	"sbert_score",
	"value_score",
	"is_match",
]
PARTIAL_SCORE_SCHEMA_VERSION = 2


def _ansi(text, code):
	if sys.stdout.isatty():
		return f"\033[{code}m{text}\033[0m"
	return text


def _render_progress_bar(label, current, total, width=32, color=None):
	if total <= 0:
		total = 1
	ratio = max(0.0, min(1.0, current / total))
	filled = int(round(width * ratio))
	filled_text = "#" * filled
	if color and filled_text:
		filled_text = _ansi(filled_text, color)
	bar = filled_text + "-" * (width - filled)
	percent = int(round(ratio * 100))
	return f"{label} [{bar}] {percent:>3}% ({current}/{total})"


class _ChunkProgressBar:
	def __init__(self, label, total, *, total_rows=None):
		self.label = label
		self.total = max(1, int(total))
		self.current = 0
		self.total_rows = max(0, int(total_rows or 0))
		self.processed_rows = 0
		self.enabled = sys.stdout.isatty()
		self._last_line = ""
		self.start_time = time.time()

	def _eta_suffix(self):
		if self.current <= 0:
			return " eta --:--"
		elapsed = max(0.0, time.time() - self.start_time)
		rate = elapsed / self.current
		remaining = max(0.0, (self.total - self.current) * rate)
		minutes, seconds = divmod(int(round(remaining)), 60)
		hours, minutes = divmod(minutes, 60)
		if hours > 0:
			return f" eta {hours:d}:{minutes:02d}:{seconds:02d}"
		return f" eta {minutes:02d}:{seconds:02d}"

	def _row_suffix(self):
		if self.total_rows <= 0:
			return ""
		return f"  rows {self.processed_rows:,}/{self.total_rows:,}"

	def update(self, current, *, processed_rows=None):
		self.current = max(0, min(int(current), self.total))
		if processed_rows is not None:
			self.processed_rows = max(0, min(int(processed_rows), self.total_rows or int(processed_rows)))
		line = (
			_render_progress_bar(self.label, self.current, self.total, color="36")
			+ self._row_suffix()
			+ self._eta_suffix()
		)
		self._last_line = line
		if self.enabled:
			print(f"\r{line}", end="", flush=True)
		else:
			print(line, flush=True)

	def close(self):
		if self.enabled and self._last_line:
			print("", flush=True)
##### ÉTAPE 1 — Préparation  

def _resolve_ground_truth_path(dataset, mock, prefix):
	mock_truth_path = f"{prefix}/{dataset}/ground_truth.csv"
	cleaned_truth_path = f"data/cleaned/{dataset}/ground_truth.csv"
	if mock and os.path.exists(mock_truth_path):
		return mock_truth_path
	return cleaned_truth_path


def _parse_progressive_stages(progressive_stages, total_rows):
	if not progressive_stages:
		return []
	values = []
	for raw in str(progressive_stages).split(","):
		raw = raw.strip()
		if not raw:
			continue
		value = int(raw)
		if value > 0:
			values.append(value)
	return sorted({stage for stage in values if stage <= total_rows})


def _load_candidate_pairs(pairs_path, candidate_strategy="cw_semantic_predictive", limit=None):
	pairs_df = pd.read_csv(pairs_path)
	if candidate_strategy == "cw_semantic_predictive" and "certificate_score" in pairs_df.columns:
		sort_cols = ["certificate_score", "id_A", "id_B"]
		ascending = [False, True, True]
		pairs_df = pairs_df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
	elif candidate_strategy == "cw_semantic_predictive":
		print(f"[MATCH] candidate_strategy={candidate_strategy} requested but no certificate_score found; using file order.")

	if limit is not None:
		pairs_df = pairs_df.head(limit).copy()
	return pairs_df


def load_and_prepare(pairs_path, source1_path, source2_path): 
	pairs_df = pd.read_csv(pairs_path)
	return prepare_pairs_dataframe(pairs_df, source1_path, source2_path)


def prepare_pairs_dataframe(pairs_df, source1_path, source2_path):
	pairs_df = pairs_df.copy()
	pairs_df['id_A'] = pairs_df['id_A'].astype(str)
	pairs_df['id_B'] = pairs_df['id_B'].astype(str)

	source_1_lookup = _load_text_lookup(source1_path)
	source_2_lookup = _load_text_lookup(source2_path)

	pairs_df['text_A'] = pairs_df['id_A'].map(source_1_lookup).fillna("")
	pairs_df['text_B'] = pairs_df['id_B'].map(source_2_lookup).fillna("")
	return pairs_df


##### ÉTAPE 2 — Value Similarity 

def jaccard_sim(text_a, text_b):
	tokens_A = set(text_a.split())
	tokens_B = set(text_b.split())
	intersection = tokens_A.intersection(tokens_B)
	union = tokens_A.union(tokens_B)
	return 0 if len(union) == 0 else len(intersection) / len(union)


def tfidf_sim(texts_a, texts_b, batch_size=1000): 
	if TfidfVectorizer is None or cosine_similarity is None:
		return _fallback_tfidf_sim(texts_a, texts_b)
	vectorizer = TfidfVectorizer()
	results = []
	for i in range(0, len(texts_a), batch_size):
		batch_a = texts_a[i:i+batch_size]
		batch_b = texts_b[i:i+batch_size]
		tfidf_matrix = vectorizer.fit_transform(batch_a + batch_b)
		scores = cosine_similarity(tfidf_matrix[:len(batch_a)], tfidf_matrix[len(batch_a):])
		results.append(np.diag(scores))
	return np.concatenate(results)


def _fallback_tfidf_sim(texts_a, texts_b):
	def tokenize(text):
		return [tok for tok in str(text).lower().split() if tok]

	doc_freq = Counter()
	docs = [tokenize(text) for text in list(texts_a) + list(texts_b)]
	for tokens in docs:
		doc_freq.update(set(tokens))
	n_docs = max(1, len(docs))

	def build_vec(text):
		tokens = tokenize(text)
		tf = Counter(tokens)
		vec = {}
		for tok, count in tf.items():
			idf = math.log((1.0 + n_docs) / (1.0 + doc_freq[tok])) + 1.0
			vec[tok] = count * idf
		norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
		return {tok: val / norm for tok, val in vec.items()}

	def cosine(left, right):
		if len(left) > len(right):
			left, right = right, left
		return sum(value * right.get(tok, 0.0) for tok, value in left.items())

	vectors_a = [build_vec(text) for text in texts_a]
	vectors_b = [build_vec(text) for text in texts_b]
	return np.array([cosine(a, b) for a, b in zip(vectors_a, vectors_b)])
 

_model = None
_TEXT_LOOKUP_CACHE = {}


def _concat_object_columns(df):
	"""Materialize one text field per entity row using all non-metadata text columns."""
	obj_cols = [
		col for col in df.select_dtypes(include=['object', 'string']).columns
		if col not in ['id', 'source']
	]
	if not obj_cols:
		return pd.Series([""] * len(df), index=df.index, dtype=str)
	return df[obj_cols].fillna("").astype(str).agg(" ".join, axis=1).str.strip()


def _load_text_lookup(source_path):
	"""
	Load and cache `id -> text` for a cleaned source file.
	"""
	cache_key = (
		source_path,
		os.path.getmtime(source_path) if os.path.exists(source_path) else None,
		os.path.getsize(source_path) if os.path.exists(source_path) else None,
	)
	cached = _TEXT_LOOKUP_CACHE.get(cache_key)
	if cached is not None:
		return cached

	df = pd.read_csv(source_path)
	df["id"] = df["id"].astype(str)
	df["text"] = _concat_object_columns(df)
	lookup = df.set_index("id")["text"].astype(str)
	_TEXT_LOOKUP_CACHE[cache_key] = lookup
	return lookup

def _get_model():
	global _model
	if SentenceTransformer is None:
		return None
	if _model is None:
		previous_levels = {}
		for logger_name in ("sentence_transformers", "transformers", "torch"):
			logger = logging.getLogger(logger_name)
			previous_levels[logger_name] = logger.level
			logger.setLevel(logging.ERROR)
		try:
			with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
				_model = SentenceTransformer('all-MiniLM-L6-v2', local_files_only=True)
		except Exception:
			with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
				_model = SentenceTransformer('all-MiniLM-L6-v2')
		finally:
			for logger_name, level in previous_levels.items():
				logging.getLogger(logger_name).setLevel(level)
	return _model


def _warm_model_if_needed():
	if SentenceTransformer is None or _model is not None:
		return
	print("[MATCH] Initialisation du modèle SBERT...")
	_get_model()

def sbert_sim(texts_a, texts_b):
	model = _get_model()
	if model is None:
		return np.clip(tfidf_sim(texts_a, texts_b), 0.0, 1.0)
	embeddings_a = model.encode(texts_a, batch_size=64, show_progress_bar=False)
	embeddings_b = model.encode(texts_b, batch_size=64, show_progress_bar=False)
	cosine_sim = cosine_similarity(embeddings_a, embeddings_b)
	return np.diag(cosine_sim)
  
##### ÉTAPE 3 — Matches provisoires 

def compute_value_sim(df, sbert_scores=None):
	result = df.copy()
	jaccard_sim_score = df.apply(lambda row: jaccard_sim(row['text_A'], row['text_B']), axis=1)
	tfidf_sim_score = tfidf_sim(df['text_A'].tolist(), df['text_B'].tolist())
	sbert_sim_score = (
		sbert_scores
		if sbert_scores is not None
		else sbert_sim(df['text_A'].tolist(), df['text_B'].tolist())
	)

	result['jaccard_score'] = jaccard_sim_score
	result['tfidf_score'] = tfidf_sim_score
	result['sbert_score'] = sbert_sim_score

	value_score = (jaccard_sim_score + tfidf_sim_score + sbert_sim_score) / 3

	result['value_score'] = value_score
	result['is_match'] = result['value_score'].apply(lambda x: 1 if x > 0.5 else 0)

	return result

##### ÉTAPE 4 — Neighbor Similarity (collective ER) 

def build_neighbor_index(blocks_df):
	neighbors = {}
	for block_id, group in blocks_df.groupby('block_id'):
		for _, row in group.iterrows():
			entity_id = row['entity_id']
			for _, row2 in group.iterrows():
				entity_id2 = row2['entity_id']
				if row['entity_id'] != row2['entity_id']:
					if row['source'] != row2['source']:
						if entity_id not in neighbors:
							neighbors[entity_id] = set()
						neighbors[entity_id].add(entity_id2)
	return neighbors
  
# Cette fonction doit retourner un DataFrame avec id_A, id_B, neighbor_score
# neighbor_score(A, B) = nombre de voisins de A matchés avec voisins de B / nombre total de voisins de A et B

def compute_neighbor_sim(blocks_df, provisional_matches_df):
	matched_pairs = set(zip(provisional_matches_df['id_A'], provisional_matches_df['id_B']))   
	neighbors = build_neighbor_index(blocks_df)
	rows = []
	for _, matches in provisional_matches_df.iterrows():
		A = matches['id_A']
		B = matches['id_B']
		neighbors_A = neighbors.get(A, set())
		neighbors_B = neighbors.get(B, set())
		matched_count = sum(1 for v_A in neighbors_A for v_B in neighbors_B if (v_A, v_B) in matched_pairs or (v_B, v_A) in matched_pairs)
		total = len(neighbors_A) + len(neighbors_B)
		score = matched_count / total if total > 0 else 0.0
		rows.append({'id_A': A, 'id_B': B, 'neighbor_score': score})
	return pd.DataFrame(rows)


def _normalize_series(series):
	numeric = pd.to_numeric(series, errors='coerce').fillna(0.0)
	if numeric.empty:
		return numeric
	min_val = float(numeric.min())
	max_val = float(numeric.max())
	if max_val <= min_val:
		return pd.Series(np.zeros(len(numeric)), index=series.index, dtype=float)
	return (numeric - min_val) / (max_val - min_val)


def _contextual_match_scores(value_df):
	"""
	Build a monotone contextual score.

	The previous collective stage mixed value and neighbor scores in a way that
	could only decrease good pair scores when neighbor evidence was sparse.
	This version only adds bounded support signals:
	- `certificate_score` from Member 2 when available
	- mutual-best bonus across candidate rows
	- local margin bonus when the best option is clearly separated
	"""
	df = value_df.copy()
	if df.empty:
		df["certificate_score_norm"] = pd.Series(dtype=float)
		df["mutual_best_bonus"] = pd.Series(dtype=float)
		df["local_margin_bonus"] = pd.Series(dtype=float)
		df["neighbor_score"] = pd.Series(dtype=float)
		df["final_score"] = pd.Series(dtype=float)
		df["is_match"] = pd.Series(dtype=int)
		return df

	df["certificate_score_norm"] = (
		_normalize_series(df["certificate_score"])
		if "certificate_score" in df.columns
		else 0.0
	)

	df["rank_A"] = df.groupby("id_A")["value_score"].rank(method="first", ascending=False)
	df["rank_B"] = df.groupby("id_B")["value_score"].rank(method="first", ascending=False)
	df["mutual_best_bonus"] = (
		(df["rank_A"] == 1.0) & (df["rank_B"] == 1.0)
	).astype(float)

	best_a = df.groupby("id_A")["value_score"].transform("max")
	second_a = (
		df[df["rank_A"] > 1.0]
		.groupby("id_A")["value_score"]
		.transform("max")
	)
	df["margin_A"] = np.where(
		df["rank_A"] == 1.0,
		best_a - second_a.reindex(df.index).fillna(0.0),
		0.0,
	)

	best_b = df.groupby("id_B")["value_score"].transform("max")
	second_b = (
		df[df["rank_B"] > 1.0]
		.groupby("id_B")["value_score"]
		.transform("max")
	)
	df["margin_B"] = np.where(
		df["rank_B"] == 1.0,
		best_b - second_b.reindex(df.index).fillna(0.0),
		0.0,
	)

	df["local_margin_bonus"] = np.clip(
		(df["margin_A"].clip(lower=0.0) + df["margin_B"].clip(lower=0.0)) / 2.0,
		0.0,
		1.0,
	)

	df["neighbor_score"] = np.clip(
		0.65 * df["mutual_best_bonus"] + 0.35 * df["local_margin_bonus"],
		0.0,
		1.0,
	)
	df["final_score"] = np.clip(
		df["value_score"]
		+ 0.18 * df["certificate_score_norm"]
		+ 0.12 * df["mutual_best_bonus"]
		+ 0.08 * df["local_margin_bonus"],
		0.0,
		1.0,
	)
	df["is_match"] = (df["final_score"] > 0.5).astype(int)
	return df.drop(columns=["rank_A", "rank_B", "margin_A", "margin_B"])
  
  
##### ÉTAPE 5 — Score final 
def combine_scores(value_df, blocks_df): 
	del blocks_df  # kept in the signature for pipeline compatibility
	return apply_adaptive_rescue(_contextual_match_scores(value_df))


##### ÉTAPE 6 — Évaluation
def evaluate(matches_df, ground_truth_path):
	if (not os.path.exists(ground_truth_path)):
		print(f"Évaluation impossible : {ground_truth_path} introuvable.")
		return None
	ground_truth_df = pd.read_csv(ground_truth_path)
	if 'is_match' in matches_df.columns:
		matches_df = matches_df[matches_df['is_match'].astype(int) == 1]
	preds = set(zip(matches_df['id_A'].astype(str), matches_df['id_B'].astype(str)))
	reels = set(
		zip(
			ground_truth_df.iloc[:, 0].astype(str),
			ground_truth_df.iloc[:, 1].astype(str),
		)
	)
	
	len_reels = len(reels)
	len_preds = len(preds)
	true_positifs = len(preds.intersection(reels))
	
	precision = true_positifs / len_preds if len_preds > 0 else 0.0
	recall = true_positifs / len_reels if len_reels > 0 else 0.0
	f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
	
	return precision, recall, f1_score
	

def _print_progressive_evaluation(final_df, truth_path, progressive_stages):
	stages = _parse_progressive_stages(progressive_stages, len(final_df))
	if not stages:
		return
	print("[MATCH] Progressive evaluation checkpoints:")
	for stage in stages:
		checkpoint_df = final_df.head(stage)
		result = evaluate(checkpoint_df, truth_path)
		if result is None:
			return
		precision, recall, f1_score = result
		n_matches = int(checkpoint_df["is_match"].astype(int).sum())
		print(
			f"  top_{stage:>6}: matches={n_matches:>6}  "
			f"precision={precision:.3f}  recall={recall:.3f}  f1={f1_score:.3f}"
		)


def _manual_cosine_diagonal(embeddings_a, embeddings_b):
	left = np.asarray(embeddings_a, dtype=float)
	right = np.asarray(embeddings_b, dtype=float)
	left_norm = np.linalg.norm(left, axis=1, keepdims=True)
	right_norm = np.linalg.norm(right, axis=1, keepdims=True)
	left_norm[left_norm == 0.0] = 1.0
	right_norm[right_norm == 0.0] = 1.0
	left = left / left_norm
	right = right / right_norm
	return np.einsum("ij,ij->i", left, right)


def _matching_cache_dir(prefix, dataset, candidate_strategy, limit=None):
	run_key = "full" if limit is None else f"limit_{limit}"
	return os.path.join(prefix, dataset, "_matching_cache", candidate_strategy, run_key)


def _cache_manifest(cache_dir):
	return os.path.join(cache_dir, "progress.json")


def _partial_scores_path(cache_dir):
	return os.path.join(cache_dir, "partial_scores.csv")


def _embedding_cache_path(cache_dir, side):
	return os.path.join(cache_dir, f"embeddings_{side}.pkl")


def _status_path(prefix, dataset):
	return os.path.join(prefix, dataset, "matching_status.json")


def _checkpoint_cluster_outputs(
	final_df,
	source1_path,
	source2_path,
	output_path,
	clustering_algorithm,
	*,
	completed,
	processed_rows,
	total_rows,
):
	return materialize_incremental_clusters(
		matches_df=final_df,
		source1_path=source1_path,
		source2_path=source2_path,
		output_dir=output_path,
		algorithm=clustering_algorithm,
		completed=completed,
		processed_rows=processed_rows,
		total_rows=total_rows,
		promote_final=completed,
	)


def _build_matching_metadata(pairs_path, source1_path, source2_path, candidate_strategy, limit):
	return {
		"pairs_path": pairs_path,
		"pairs_mtime": os.path.getmtime(pairs_path) if os.path.exists(pairs_path) else None,
		"pairs_size": os.path.getsize(pairs_path) if os.path.exists(pairs_path) else None,
		"source1_path": source1_path,
		"source1_mtime": os.path.getmtime(source1_path) if os.path.exists(source1_path) else None,
		"source2_path": source2_path,
		"source2_mtime": os.path.getmtime(source2_path) if os.path.exists(source2_path) else None,
		"candidate_strategy": candidate_strategy,
		"limit": limit,
		"partial_score_schema_version": PARTIAL_SCORE_SCHEMA_VERSION,
	}


def _metadata_matches(saved, current):
	return all(saved.get(key) == value for key, value in current.items())


def _load_progress(cache_dir, metadata, resume):
	manifest_path = _cache_manifest(cache_dir)
	partial_path = _partial_scores_path(cache_dir)
	if not resume or not os.path.exists(manifest_path) or not os.path.exists(partial_path):
		return {"next_index": 0, "processed_rows": 0, "completed": False, "metadata": metadata}

	with open(manifest_path, "r", encoding="utf-8") as handle:
		payload = json.load(handle)
	if not _metadata_matches(payload.get("metadata", {}), metadata):
		return {"next_index": 0, "processed_rows": 0, "completed": False, "metadata": metadata}
	return payload


def _save_progress(cache_dir, payload):
	os.makedirs(cache_dir, exist_ok=True)
	with open(_cache_manifest(cache_dir), "w", encoding="utf-8") as handle:
		json.dump(payload, handle, indent=2)


def _load_embedding_cache(path):
	if not os.path.exists(path):
		return {}
	with open(path, "rb") as handle:
		return pickle.load(handle)


def _save_embedding_cache(path, cache):
	with open(path, "wb") as handle:
		pickle.dump(cache, handle)


def _reset_partial_cache(cache_dir, reason):
	partial_path = _partial_scores_path(cache_dir)
	manifest_path = _cache_manifest(cache_dir)
	if os.path.exists(partial_path):
		broken_path = partial_path + ".broken"
		if os.path.exists(broken_path):
			os.remove(broken_path)
		os.replace(partial_path, broken_path)
	if os.path.exists(manifest_path):
		os.remove(manifest_path)
	print(f"[MATCH] Cache reset: {reason}")


def _append_partial_scores(cache_dir, chunk_df):
	partial_path = _partial_scores_path(cache_dir)
	write_header = not os.path.exists(partial_path)
	for column in PARTIAL_SCORE_COLUMNS:
		if column not in chunk_df.columns:
			chunk_df[column] = np.nan
	chunk_df.loc[:, PARTIAL_SCORE_COLUMNS].to_csv(
		partial_path,
		mode="a",
		header=write_header,
		index=False,
	)


def _load_partial_scores(cache_dir):
	partial_path = _partial_scores_path(cache_dir)
	if not os.path.exists(partial_path):
		return pd.DataFrame()
	try:
		df = pd.read_csv(partial_path)
	except pd.errors.ParserError:
		_reset_partial_cache(cache_dir, "partial_scores schema mismatch")
		return pd.DataFrame()
	if list(df.columns) != PARTIAL_SCORE_COLUMNS:
		_reset_partial_cache(cache_dir, "partial_scores header mismatch")
		return pd.DataFrame()
	return df


def _chunk_pairs(pairs_df, start_index, chunk_size):
	for start in range(start_index, len(pairs_df), chunk_size):
		yield start, pairs_df.iloc[start:start + chunk_size].copy()


def _encode_missing_embeddings(cache, missing_ids, text_lookup):
	if not missing_ids:
		return cache
	model = _get_model()
	if model is None:
		return cache
	texts = [text_lookup[entity_id] for entity_id in missing_ids]
	embeddings = model.encode(texts, batch_size=64, show_progress_bar=False)
	for entity_id, embedding in zip(missing_ids, embeddings):
		cache[str(entity_id)] = np.asarray(embedding, dtype=float)
	return cache


def _cached_sbert_scores(df, cache_dir):
	left_path = _embedding_cache_path(cache_dir, "left")
	right_path = _embedding_cache_path(cache_dir, "right")
	left_cache = _load_embedding_cache(left_path)
	right_cache = _load_embedding_cache(right_path)

	left_lookup = (
		df[["id_A", "text_A"]]
		.drop_duplicates(subset=["id_A"])
		.assign(id_A=lambda frame: frame["id_A"].astype(str))
		.set_index("id_A")["text_A"]
		.astype(str)
		.to_dict()
	)
	right_lookup = (
		df[["id_B", "text_B"]]
		.drop_duplicates(subset=["id_B"])
		.assign(id_B=lambda frame: frame["id_B"].astype(str))
		.set_index("id_B")["text_B"]
		.astype(str)
		.to_dict()
	)

	missing_left = [entity_id for entity_id in left_lookup if entity_id not in left_cache]
	missing_right = [entity_id for entity_id in right_lookup if entity_id not in right_cache]
	left_cache = _encode_missing_embeddings(left_cache, missing_left, left_lookup)
	right_cache = _encode_missing_embeddings(right_cache, missing_right, right_lookup)
	if missing_left:
		_save_embedding_cache(left_path, left_cache)
	if missing_right:
		_save_embedding_cache(right_path, right_cache)

	if _get_model() is None:
		return np.clip(tfidf_sim(df["text_A"].tolist(), df["text_B"].tolist()), 0.0, 1.0)

	left_embeddings = np.vstack([left_cache[str(entity_id)] for entity_id in df["id_A"].astype(str)])
	right_embeddings = np.vstack([right_cache[str(entity_id)] for entity_id in df["id_B"].astype(str)])
	if cosine_similarity is not None:
		return np.diag(cosine_similarity(left_embeddings, right_embeddings))
	return _manual_cosine_diagonal(left_embeddings, right_embeddings)


def _finalize_matching_outputs(
	partial_df,
	blocks_df,
	output_path,
	truth_path,
	progressive_stages=None,
	completed=False,
	processed_rows=0,
	total_rows=0,
):
	partial_df = partial_df.copy()
	for name in ["id_A", "id_B"]:
		if name in partial_df.columns:
			partial_df[name] = partial_df[name].astype(str)

	if partial_df.empty:
		jaccard_df = pd.DataFrame(columns=["id_A", "id_B", "similarity_score", "is_match"])
		tfidf_df = jaccard_df.copy()
		sbert_df = jaccard_df.copy()
		v_df = jaccard_df.copy()
		f_df = pd.DataFrame(columns=["id_A", "id_B", "final_score", "is_match"])
		final_df = f_df.copy()
	else:
		final_df = combine_scores(partial_df, blocks_df)
		jaccard_df = partial_df[['id_A', 'id_B', 'jaccard_score', 'is_match']].rename(columns={'jaccard_score': 'similarity_score'})
		tfidf_df = partial_df[['id_A', 'id_B', 'tfidf_score', 'is_match']].rename(columns={'tfidf_score': 'similarity_score'})
		sbert_df = partial_df[['id_A', 'id_B', 'sbert_score', 'is_match']].rename(columns={'sbert_score': 'similarity_score'})
		v_df = partial_df[['id_A', 'id_B', 'value_score' ,'is_match']].rename(columns={'value_score': 'similarity_score'})
		f_df = final_df[['id_A', 'id_B', 'final_score', 'is_match']]

	jaccard_df.to_csv(os.path.join(output_path, "match_results_jaccard.csv"), index=False)
	tfidf_df.to_csv(os.path.join(output_path, "match_results_tfidf.csv"), index=False)
	sbert_df.to_csv(os.path.join(output_path, "match_results_sbert.csv"), index=False)
	v_df.to_csv(os.path.join(output_path, "match_results_combined.csv"), index=False)
	f_df.to_csv(os.path.join(output_path, "match_results_collective.csv"), index=False)

	if total_rows > 0:
		print(
			f"[MATCH] Progress: processed_rows={processed_rows:,}/{total_rows:,}"
			f"  completed={completed}"
		)
	if progressive_stages and not final_df.empty:
		_print_progressive_evaluation(final_df, truth_path, progressive_stages)
	result = evaluate(final_df, truth_path)
	if result:
		precision, recall, f1_score = result
		print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1_score:.3f}")
	return final_df
	
##### ÉTAPE 7 — Orchestration
def run_matching(
	dataset,
	mock=False,
	limit=None,
	candidate_strategy="cw_semantic_predictive",
	progressive_stages=None,
	time_limit_seconds=None,
	resume=True,
	chunk_size=250,
	online_clustering=False,
	clustering_algorithm="connected_components",
	online_cluster_every_n_chunks=1,
):
	prefix = "output/mock" if mock else "output"
	cfg = DATASET_REGISTRY.get(dataset, {})

	pairs_path   = f"{prefix}/{dataset}/candidate_pairs.csv"
	blocks_path  = f"{prefix}/{dataset}/blocks.csv"
	# FIX: in mock mode, source files live alongside the other mock fixtures,
	# not in data/cleaned/ (which requires a full Member 1 run).
	if mock:
		source1_path = f"{prefix}/{dataset}/cleaned_source1.csv"
		source2_path = f"{prefix}/{dataset}/cleaned_source2.csv"
	else:
		source1_path = f"data/cleaned/{dataset}/cleaned_source1.csv"
		source2_path = f"data/cleaned/{dataset}/cleaned_source2.csv"
	truth_path   = _resolve_ground_truth_path(dataset, mock, prefix)

	if not os.path.exists(blocks_path):
		print(f"Matching impossible : {blocks_path} introuvable.")
		return None
		
	print(f"[{dataset}] Chargement des données...")
	pairs_df = _load_candidate_pairs(
		pairs_path,
		candidate_strategy=candidate_strategy,
		limit=limit,
	)
	print(
		f"[{dataset}] Paires candidates retenues pour matching: {len(pairs_df):,}"
		f"  [strategy={candidate_strategy}]"
	)
	blocks_df = pd.read_csv(blocks_path)
	output_path = f"{prefix}/{dataset}/"
	cache_dir = _matching_cache_dir(prefix, dataset, candidate_strategy, limit=limit)
	os.makedirs(cache_dir, exist_ok=True)
	os.makedirs(output_path, exist_ok=True)
	metadata = _build_matching_metadata(
		pairs_path,
		source1_path,
		source2_path,
		candidate_strategy,
		limit,
	)
	progress = _load_progress(cache_dir, metadata, resume=resume)
	if progress.get("next_index", 0) > 0:
		partial_preview = _load_partial_scores(cache_dir)
		if partial_preview.empty:
			progress = {"next_index": 0, "processed_rows": 0, "completed": False, "metadata": metadata}
			_save_progress(cache_dir, progress)
	if progress.get("next_index", 0) == 0 and os.path.exists(_partial_scores_path(cache_dir)):
		_reset_partial_cache(cache_dir, "starting fresh from row 0")
		progress = {"next_index": 0, "processed_rows": 0, "completed": False, "metadata": metadata}
		_save_progress(cache_dir, progress)
	if progress.get("completed") and progress.get("next_index", 0) >= len(pairs_df):
		print(f"[{dataset}] Reprise: matching déjà terminé, rechargement des résultats partiels.")
	else:
		if progress.get("next_index", 0) > 0:
			print(f"[{dataset}] Reprise depuis la paire {progress['next_index']:,}.")
		print(f"[{dataset}] Calcul value similarity...")
		_warm_model_if_needed()
		start_time = time.time()
		next_index = int(progress.get("next_index", 0))
		total_chunks = max(1, (len(pairs_df) + chunk_size - 1) // chunk_size)
		completed_chunks = 0 if next_index <= 0 else min(total_chunks, (next_index + chunk_size - 1) // chunk_size)
		progress_bar = _ChunkProgressBar(f"[{dataset}] Matching", total_chunks, total_rows=len(pairs_df))
		progress_bar.update(completed_chunks, processed_rows=next_index)
		chunk_counter = 0
		for start, chunk_pairs in _chunk_pairs(pairs_df, next_index, chunk_size):
			chunk_counter += 1
			if time_limit_seconds is not None and time.time() - start_time >= time_limit_seconds:
				break
			merged_df = prepare_pairs_dataframe(chunk_pairs, source1_path, source2_path)
			merged_df = merged_df.dropna(subset=['text_A', 'text_B'])
			if merged_df.empty:
				next_index = start + len(chunk_pairs)
				progress = {
					"next_index": next_index,
					"processed_rows": next_index,
					"completed": next_index >= len(pairs_df),
					"metadata": metadata,
				}
				_save_progress(cache_dir, progress)
				progress_bar.update(completed_chunks + chunk_counter, processed_rows=next_index)
				continue
			sbert_scores = _cached_sbert_scores(merged_df, cache_dir)
			value_df = compute_value_sim(merged_df, sbert_scores=sbert_scores)
			_append_partial_scores(cache_dir, value_df)
			next_index = start + len(chunk_pairs)
			progress = {
				"next_index": next_index,
				"processed_rows": next_index,
				"completed": next_index >= len(pairs_df),
				"metadata": metadata,
			}
			_save_progress(cache_dir, progress)
			progress_bar.update(completed_chunks + chunk_counter, processed_rows=next_index)
			should_refresh_clusters = (
				online_clustering
				and (
					progress["completed"]
					or online_cluster_every_n_chunks <= 1
					or (chunk_counter % online_cluster_every_n_chunks == 0)
				)
			)
			if should_refresh_clusters:
				partial_df = _load_partial_scores(cache_dir)
				checkpoint_final_df = _finalize_matching_outputs(
					partial_df=partial_df,
					blocks_df=blocks_df,
					output_path=output_path,
					truth_path=truth_path,
					progressive_stages=None,
					completed=progress["completed"],
					processed_rows=int(progress.get("processed_rows", 0)),
					total_rows=len(pairs_df),
				)
				_checkpoint_cluster_outputs(
					final_df=checkpoint_final_df,
					source1_path=source1_path,
					source2_path=source2_path,
					output_path=output_path,
					clustering_algorithm=clustering_algorithm,
					completed=progress["completed"],
					processed_rows=int(progress.get("processed_rows", 0)),
					total_rows=len(pairs_df),
				)
			if time_limit_seconds is not None and time.time() - start_time >= time_limit_seconds:
				break
		progress_bar.close()

	partial_df = _load_partial_scores(cache_dir)
	completed = int(progress.get("next_index", 0)) >= len(pairs_df)
	progress["completed"] = completed
	_save_progress(cache_dir, progress)

	print(f"[{dataset}] Calcul collective similarity...")
	final_df = _finalize_matching_outputs(
		partial_df=partial_df,
		blocks_df=blocks_df,
		output_path=output_path,
		truth_path=truth_path,
		progressive_stages=progressive_stages if cfg.get("truth_col_s1") and cfg.get("truth_col_s2") else None,
		completed=completed,
		processed_rows=int(progress.get("processed_rows", 0)),
		total_rows=len(pairs_df),
	)
	cluster_payload = None
	if online_clustering:
		print(f"[{dataset}] Mise à jour des clusters incrémentaux...")
		cluster_payload = _checkpoint_cluster_outputs(
			final_df=final_df,
			source1_path=source1_path,
			source2_path=source2_path,
			output_path=output_path,
			clustering_algorithm=clustering_algorithm,
			completed=completed,
			processed_rows=int(progress.get("processed_rows", 0)),
			total_rows=len(pairs_df),
		)
	if not (cfg.get("truth_col_s1") and cfg.get("truth_col_s2")):
		print(f"[{dataset}] Évaluation ignorée : aucune ground truth CSV configurée.")

	status_payload = {
		"dataset": dataset,
		"candidate_strategy": candidate_strategy,
		"limit": limit,
		"time_limit_seconds": time_limit_seconds,
		"resume": resume,
		"chunk_size": chunk_size,
		"online_clustering": online_clustering,
		"clustering_algorithm": clustering_algorithm if online_clustering else None,
		"online_cluster_every_n_chunks": online_cluster_every_n_chunks if online_clustering else None,
		"processed_rows": int(progress.get("processed_rows", 0)),
		"total_rows": len(pairs_df),
		"completed": completed,
		"cache_dir": cache_dir,
	}
	with open(_status_path(prefix, dataset), "w", encoding="utf-8") as handle:
		json.dump(status_payload, handle, indent=2)

	return {
		"completed": completed,
		"processed_rows": int(progress.get("processed_rows", 0)),
		"total_rows": len(pairs_df),
		"cache_dir": cache_dir,
		"status_path": _status_path(prefix, dataset),
		"final_df": final_df,
		"cluster_payload": cluster_payload,
	}


  
























    
   
    
   

  
