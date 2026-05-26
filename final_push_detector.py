import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import re

class StyleChangeFinalPush:
    """
    Absolute Max Style Change Detection:
    Uses Hybrid Clustering + Trajectory Flow.
    """
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.embed_model = SentenceTransformer(model_name)

    def detect_changes(self, embeddings, lines):
        if len(lines) < 3:
            return [0] * (len(lines) - 1)
            
        # 1. Trajectory Flow: Similarity of current sentence to the 'Moving Average' of the current author
        changes = []
        author_window = [embeddings[0]]
        
        for i in range(1, len(lines)):
            author_centroid = np.mean(author_window, axis=0)
            sim = cosine_similarity([author_centroid], [embeddings[i]])[0][0]
            
            # 2. Local jump vs Global Shift
            local_sim = cosine_similarity([embeddings[i-1]], [embeddings[i]])[0][0]
            
            # Heuristic: If similarity to the CUMULATIVE author profile drops significantly
            is_change = 0
            if (sim < 0.45 and local_sim < 0.6) or (local_sim < 0.25):
                is_change = 1
                # Start new author window
                author_window = [embeddings[i]]
            else:
                author_window.append(embeddings[i])
                if len(author_window) > 10: author_window.pop(0) # Keep rolling
                
            changes.append(is_change)
            
        return changes

def evaluate_final(data_dir, limit=100):
    sc = StyleChangeFinalPush()
    y_true, y_pred = [], []
    for subset in ['easy', 'medium', 'hard']:
        val_dir = os.path.join(data_dir, subset, 'validation')
        problems = sorted([f for f in os.listdir(val_dir) if f.startswith('problem-') and f.endswith('.txt')])[:limit]
        
        for prob_file in problems:
            with open(os.path.join(val_dir, prob_file), 'r') as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
            with open(os.path.join(val_dir, 'truth-' + prob_file.replace('.txt', '.json')), 'r') as f:
                true_changes = json.load(f)['changes']
            
            if len(lines) < 2: continue
            embeddings = sc.embed_model.encode(lines)
            pred = sc.detect_changes(embeddings, lines)
            min_len = min(len(pred), len(true_changes))
            y_pred.extend(pred[:min_len])
            y_true.extend(true_changes[:min_len])
            
    print("\nFINAL PUSH STYLE CHANGE EVALUATION (Aggregated):")
    print(classification_report(y_true, y_pred))

if __name__ == "__main__":
    data_dir = "style-change/data/extracted/mawsa26-pan-zenodo"
    evaluate_final(data_dir, limit=100)
