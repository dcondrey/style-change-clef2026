import os
import json
import numpy as np
import re
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

class TrajectoryManifoldDetector:
    """
    Final Absolute Max Solution: Trajectory Manifold Exit Detection.
    Models the current author as a local Gaussian Manifold.
    Detects authorship shifts as a sustained 'Manifold Exit'.
    """
    def __init__(self):
        # We use character n-grams to build the style-space
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4))

    def extract_features(self, lines):
        # Transform lines into a dense style-space
        X = self.vectorizer.fit_transform(lines).toarray()
        # Reduce dimensionality to focus on the core 'Style Manifold'
        pca = PCA(n_components=min(10, X.shape[0], X.shape[1]))
        return pca.fit_transform(X)

    def detect_changes(self, lines, sensitivity=2.0):
        n = len(lines)
        if n < 5: return [0] * (n - 1)
        
        # 1. Map to Style Manifold
        coords = self.extract_features(lines)
        
        changes = []
        # We start with the first 3 sentences to define the author's subspace
        author_samples = [coords[0], coords[1], coords[2]]
        
        for i in range(1, n):
            # Calculate Mahalanobis-like distance to current author's centroid
            centroid = np.mean(author_samples, axis=0)
            dist = np.linalg.norm(coords[i] - centroid)
            
            # Local variance of the author
            author_std = np.std([np.linalg.norm(s - centroid) for s in author_samples]) + 1e-6
            
            # Z-score of the new sentence in the author's style-space
            z_score = dist / author_std
            
            # LOGIC: A shift is a high Z-score that doesn't 're-merge'
            # If z_score > sensitivity, it's a potential shift
            is_change = 1 if z_score > sensitivity else 0
            
            if is_change:
                # Reset manifold for the new suspected author
                author_samples = [coords[i]]
            else:
                # Accumulate the author's manifold, staying local to catch subtle drifts
                author_samples.append(coords[i])
                if len(author_samples) > 8: author_samples.pop(0)
                
            changes.append(is_change)
            
        return changes

def run_manifold_test(data_dir, subset='hard', limit=100):
    detector = TrajectoryManifoldDetector()
    val_dir = os.path.join(data_dir, subset, 'validation')
    problems = sorted([f for f in os.listdir(val_dir) if f.startswith('problem-') and f.endswith('.txt')])[:limit]
    
    y_true, y_pred = [], []
    for prob_file in problems:
        with open(os.path.join(val_dir, prob_file), 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        with open(os.path.join(val_dir, 'truth-' + prob_file.replace('.txt', '.json')), 'r') as f:
            true_changes = json.load(f)['changes']
        
        if len(lines) < 2: continue
        # Sensitivity 3.0 = High precision, 1.5 = High recall
        pred = detector.detect_changes(lines, sensitivity=2.8)
        
        min_len = min(len(pred), len(true_changes))
        y_pred.extend(pred[:min_len])
        y_true.extend(true_changes[:min_len])
            
    print(f"\nTRAJECTORY MANIFOLD EXIT ANALYSIS (Subset: {subset.upper()}):")
    print(classification_report(y_true, y_pred))

if __name__ == "__main__":
    data_dir = "style-change/data/extracted/mawsa26-pan-zenodo"
    run_manifold_test(data_dir, 'hard', 200)
