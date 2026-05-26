import os
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import SpectralClustering
from sklearn.metrics import classification_report, accuracy_score
from scipy.sparse.csgraph import laplacian

class LaplacianCutDetector:
    """
    Superior Solution: Laplacian Graph Partitioning.
    Finds the 'Fiedler Vector' of the document's stylistic graph.
    Author shifts are detected as global topological fractures.
    """
    def __init__(self):
        # High-dimensional character n-grams to capture the 'Author's Pulse'
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 5), min_df=1)

    def detect_changes(self, lines):
        n = len(lines)
        if n < 4: return [0] * (n - 1)
        
        # 1. Stylistic Adjacency Matrix
        try:
            X = self.vectorizer.fit_transform(lines)
            # Use dot product for fast cosine similarity matrix
            adj = (X * X.T).toarray()
        except:
            return [0] * (n - 1)
            
        # 2. Compute the Graph Laplacian
        # L = D - A (Degree minus Adjacency)
        # We use the normalized Laplacian to handle document variance
        L = laplacian(adj, normed=True)
        
        # 3. Spectral Decomposition
        # We solve for the eigenvectors of the Laplacian.
        # The first non-zero eigenvector (Fiedler vector) partitions the graph.
        evals, evecs = np.linalg.eigh(L)
        
        # Identify the Eigen-gap: jump in eigenvalues signals author count
        # For MAWSA, it's usually 2-5 authors.
        gaps = np.diff(evals[:6])
        n_clusters = np.argmax(gaps) + 1
        if n_clusters < 2: n_clusters = 2
        
        # 4. Global Partitioning
        sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', n_init=30, random_state=42)
        labels = sc.fit_predict(adj)
        
        # 5. Extract Change-Points from Cluster Transitions
        changes = []
        for i in range(n - 1):
            # A transition between author subspaces is a shift
            is_change = 1 if labels[i] != labels[i+1] else 0
            changes.append(is_change)
            
        return changes

def run_laplacian_test(data_dir, subset='hard', limit=100):
    detector = LaplacianCutDetector()
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
            
    print(f"\nLAPLACIAN GRAPH CUT ANALYSIS (Subset: {subset.upper()}):")
    print(classification_report(y_true, y_pred))
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")

if __name__ == "__main__":
    data_dir = "style-change/data/extracted/mawsa26-pan-zenodo"
    run_laplacian_test(data_dir, 'hard', 300)
