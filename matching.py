from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import os
##### ÉTAPE 1 — Préparation  

def load_and_prepare(pairs_path, source1_path, source2_path): 
	source_1_df = pd.read_csv(source1_path)
	source_2_df = pd.read_csv(source2_path)
	pairs_df = pd.read_csv(pairs_path)
	
	source_1_df['id'] = source_1_df['id'].astype(str)
	source_2_df['id'] = source_2_df['id'].astype(str)
	pairs_df['id_A'] = pairs_df['id_A'].astype(str)
	pairs_df['id_B'] = pairs_df['id_B'].astype(str)

	def concat_obj_cols(df):
		# 1. Colonnes object sauf id et source
		obj_cols = [col for col in df.select_dtypes(include='object').columns
			if col not in ['id', 'source']]

		# 2. Concaténer en colonne text en ignorant les NaN
		df['text'] = df[obj_cols].apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)

		return df

	source_1_df = concat_obj_cols(source_1_df)
	source_2_df = concat_obj_cols(source_2_df)

	merged_df = pairs_df.merge(source_1_df[['id', 'text']], left_on='id_A', right_on='id', how='left') \
		.drop(columns='id') \
		.rename(columns={'text': 'text_A'})

	merged_df = merged_df.merge(source_2_df[['id', 'text']], left_on='id_B', right_on='id', how='left') \
		.drop(columns='id') \
		.rename(columns={'text': 'text_B'})
	return merged_df


##### ÉTAPE 2 — Value Similarity 

def jaccard_sim(text_a, text_b):
	tokens_A = set(text_a.split())
	tokens_B = set(text_b.split())
	intersection = tokens_A.intersection(tokens_B)
	union = tokens_A.union(tokens_B)
	return 0 if len(union) == 0 else len(intersection) / len(union)


def tfidf_sim(texts_a, texts_b, batch_size=1000): 
	vectorizer = TfidfVectorizer()
	results = []
	for i in range(0, len(texts_a), batch_size):
		batch_a = texts_a[i:i+batch_size]
		batch_b = texts_b[i:i+batch_size]
		tfidf_matrix = vectorizer.fit_transform(batch_a + batch_b)
		scores = cosine_similarity(tfidf_matrix[:len(batch_a)], tfidf_matrix[len(batch_a):])
		results.append(np.diag(scores))
	return np.concatenate(results)
 

_model = None

def _get_model():
	global _model
	if _model is None:
		_model = SentenceTransformer('all-MiniLM-L6-v2')
	return _model

def sbert_sim(texts_a, texts_b):
	model = _get_model()
	embeddings_a = model.encode(texts_a, batch_size=64, show_progress_bar=True)
	embeddings_b = model.encode(texts_b, batch_size=64, show_progress_bar=True)
	cosine_sim = cosine_similarity(embeddings_a, embeddings_b)
	return np.diag(cosine_sim)
  
##### ÉTAPE 3 — Matches provisoires 

def compute_value_sim(df):
	result = pd.DataFrame(df[['id_A', 'id_B']]);
	jaccard_sim_score = df.apply(lambda row: jaccard_sim(row['text_A'], row['text_B']), axis=1)
	tfidf_sim_score = tfidf_sim(df['text_A'].tolist(), df['text_B'].tolist())
	sbert_sim_score = sbert_sim(df['text_A'].tolist(), df['text_B'].tolist())

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
  
  
##### ÉTAPE 5 — Score final 
def combine_scores(value_df, blocks_df): 
	provisional = value_df[value_df['is_match'] == 1]
	
	if provisional.empty:
		value_df['final_score'] = value_df['value_score']
		value_df['neighbor_score'] = 0.0
		value_df['is_match'] = value_df['is_match']
		return value_df                        
	neighbor_df = compute_neighbor_sim(blocks_df, provisional)
	
	finale_df = value_df.merge(neighbor_df, on=['id_A', 'id_B'], how='left')
	finale_df['final_score'] = 0.6 * finale_df['value_score'] + 0.4 * finale_df['neighbor_score'].fillna(0)
	finale_df['is_match'] = finale_df['final_score'].apply(lambda x: 1 if x > 0.5 else 0) 
	return finale_df


##### ÉTAPE 6 — Évaluation
def evaluate(matches_df, ground_truth_path):
	if (not os.path.exists(ground_truth_path)):
		print(f"Évaluation impossible : {ground_truth_path} introuvable.")
		return None
	ground_truth_df = pd.read_csv(ground_truth_path)
	preds = set(zip(matches_df['id_A'], matches_df['id_B']))
	reels = set(zip(ground_truth_df.iloc[:, 0], ground_truth_df.iloc[:,1]))
	
	len_reels = len(reels)
	len_preds = len(preds)
	true_positifs = len(preds.intersection(reels))
	
	precision = true_positifs / len_preds if len_preds > 0 else 0.0
	recall = true_positifs / len_reels if len_reels > 0 else 0.0
	f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
	
	return precision, recall, f1_score
	

##### ÉTAPE 7 — Orchestration
def run_matching(dataset, mock=False, limit=None):
	prefix = "output/mock" if mock else "output"
	
	pairs_path   = f"{prefix}/{dataset}/candidate_pairs.csv"
	blocks_path  = f"{prefix}/{dataset}/blocks.csv"
	source1_path = f"data/cleaned/{dataset}/cleaned_source1.csv"
	source2_path = f"data/cleaned/{dataset}/cleaned_source2.csv"
	truth_path   = f"data/cleaned/{dataset}/ground_truth.csv"

	if not os.path.exists(blocks_path):
		print(f"Matching impossible : {blocks_path} introuvable.")
		return None
		
	print(f"[{dataset}] Chargement des données...")
	merged_df = load_and_prepare(pairs_path, source1_path, source2_path)
	merged_df = merged_df.dropna(subset=['text_A', 'text_B']) 
	merged_df = merged_df.head(limit)
	
	print(f"[{dataset}] Calcul value similarity...")
	value_df = compute_value_sim(merged_df)

	blocks_df = pd.read_csv(blocks_path)
	print(f"[{dataset}] Calcul collective similarity...")
	final_df = combine_scores(value_df, blocks_df)
	  
	print(f"[{dataset}] Sauvegarde des CSV...")                                                                                
	output_path = f"output/{dataset}/"
	
	jaccard_df = value_df[['id_A', 'id_B', 'jaccard_score', 'is_match']].rename(columns={'jaccard_score': 'similarity_score'})
	jaccard_df.to_csv(output_path + "match_results_jaccard.csv", index=False)
	
	tfidf_df = value_df[['id_A', 'id_B', 'tfidf_score', 'is_match']].rename(columns={'tfidf_score': 'similarity_score'})
	tfidf_df.to_csv(output_path + "match_results_tfidf.csv", index=False)
	
	sbert_df = value_df[['id_A', 'id_B', 'sbert_score', 'is_match']].rename(columns={'sbert_score': 'similarity_score'})
	sbert_df.to_csv(output_path + "match_results_sbert.csv", index=False)
	
	v_df = value_df[['id_A', 'id_B', 'value_score' ,'is_match']].rename(columns={'value_score': 'similarity_score'})
	v_df.to_csv(output_path + "match_results_combined.csv", index=False)
	
	f_df = final_df[['id_A', 'id_B', 'final_score', 'is_match']]
	f_df.to_csv(output_path + "match_results_collective.csv", index=False)
	
	result = evaluate(final_df, truth_path)
	if result:
		precision, recall, f1_score = result
		print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1_score:.3f}")

	return final_df


  
























    
   
    
   

  
