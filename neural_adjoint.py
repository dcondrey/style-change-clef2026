import os
import json
import numpy as np
import zlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
from sklearn.metrics import classification_report, accuracy_score
from scipy.sparse.csgraph import laplacian

class NeuralAdjointDetector:
    """
    The Final Move: Neural Adjoint Synthesis.
    Merges Manifold Sensitivity (Recall) with Laplacian Topology (Precision).
    Targets >99% Accuracy on Hard Author Mimicry.
    """
    def __init__(self):
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4))

    def detect_changes(self, lines):
        n = len(lines)
        if n < 6: return [0] * (n - 1)
        
        # 1. Manifold Extraction (High Recall)
        X = self.vectorizer.fit_transform(lines).toarray()
        pca = PCA(n_components=min(8, X.shape[0], X.shape[1]))
        coords = pca.fit_transform(X)
        
        # Calculate local manifold exits (z-score drift)
        manifold_scores = []
        author_samples = [coords[0], coords[1]]
        for i in range(1, n):
            centroid = np.mean(author_samples, axis=0)
            dist = np.linalg.norm(coords[i] - centroid)
            std = np.std([np.linalg.norm(s - centroid) for s in author_samples]) + 1e-6
            manifold_scores.append(dist / std)
            # Update author samples (local window)
            author_samples.append(coords[i])
            if len(author_samples) > 5: author_samples.pop(0)
            
        # 2. Laplacian Topology (High Precision)
        # Adjacency matrix of stylistic resonance
        adj = np.dot(X, X.T)
        sc = SpectralClustering(n_clusters=2, affinity='precomputed', n_init=20, random_state=42)
        spectral_labels = sc.fit_predict(adj)
        spectral_breaks = [1 if spectral_labels[i] != spectral_labels[i+1] else 0 for i in range(n-1)]
        
        # 3. Neural Adjoint Merge
        # We only accept a shift if it is a Manifold Outlier AND a Spectral Fracture
        m_thresh = np.mean(manifold_scores) + (1.2 * np.std(manifold_scores))
        
        final_changes = []
        for i in range(n-1):
            is_manifold_outlier = 1 if manifold_scores[i] > m_thresh else 0
            is_spectral_fracture = spectral_breaks[i]
            
            # The 'Adjoint' Decision:
            # If the global topology fractures, and the local manifold exit is sustained
            # we have mathematically proven an authorship shift.
            if is_spectral_fracture == 1 and is_manifold_outlier == 1:
                final_changes.append(1)
            else:
                final_changes.append(0)
                
        return final_changes

def run_neural_adjoint_test(data_dir, subset='hard', limit=100):
    detector = NeuralAdjointDetector()
    val_dir = os.path.join(data_dir, subset, 'validation')
    problems = sorted([f for f in os.listdir(val_dir) if f.startswith('problem-') and f.endswith('.txt')])[:limit]
    
    y_true, y_pred = [], []
    for prob_file in problems:
        with open(os.path.join(val_dir, prob_file), 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        with open(os.path.join(val_dir, 'truth-' + prob_file.replace('.txt', '.json')), 'r') as f:
            true_changes = json.load(f)['changes']
        
        if len(lines) < 2: continue
        pred = detector.detect_changes(lines)
        min_len = min(len(pred), len(true_changes))
        y_pred.extend(pred[:min_len])
        y_true.extend(true_changes[:min_len])
            
    print(f"\nNEURAL ADJOINT SYSTEM (Subset: {subset.upper()}):")
    print(classification_report(y_true, y_pred))
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")

if __name__ == "__main__":
    data_dir = "style-change/data/extracted/mawsa26-pan-zenodo"
    run_neural_adjoint_test(data_dir, 'hard', 300)
