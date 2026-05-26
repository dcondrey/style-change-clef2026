import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
import re
from collections import Counter

from sklearn.cluster import KMeans

class StyleChangeAbsoluteMax:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.embed_model = SentenceTransformer(model_name)

    def detect_changes(self, embeddings, lines):
        """
        Global Author Clustering:
        Assumes there are N authors. Clusters all sentences into 2 or more groups.
        Author shifts occur when adjacent sentences belong to different clusters.
        """
        if len(lines) < 3:
            return [0] * (len(lines) - 1)
            
        # 1. Cluster embeddings (Assume 2 authors for baseline, but could be dynamic)
        n_clusters = 2
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(embeddings)
        
        # 2. Identify shifts
        changes = []
        for i in range(len(clusters) - 1):
            is_change = 1 if clusters[i] != clusters[i+1] else 0
            changes.append(is_change)
            
        return changes

def evaluate_on_hard(data_dir, limit=100):
    sc = StyleChangeAbsoluteMax()
    val_dir = os.path.join(data_dir, 'hard', 'validation')
    problems = sorted([f for f in os.listdir(val_dir) if f.startswith('problem-') and f.endswith('.txt')])[:limit]
    
    y_true, y_pred = [], []
    for prob_file in problems:
        with open(os.path.join(val_dir, prob_file), 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        
        truth_file = 'truth-' + prob_file.replace('.txt', '.json')
        with open(os.path.join(val_dir, truth_file), 'r') as f:
            true_changes = json.load(f)['changes']
        
        if len(lines) < 2: continue
        
        embeddings = sc.embed_model.encode(lines)
        pred_changes = sc.detect_changes(embeddings, lines)
        
        min_len = min(len(pred_changes), len(true_changes))
        y_pred.extend(pred_changes[:min_len])
        y_true.extend(true_changes[:min_len])
        
    print("\nAdversarial Author Verification (HARD subset):")
    print(classification_report(y_true, y_pred))
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")

if __name__ == "__main__":
    data_dir = "style-change/data/extracted/mawsa26-pan-zenodo"
    evaluate_on_hard(data_dir, limit=50)
