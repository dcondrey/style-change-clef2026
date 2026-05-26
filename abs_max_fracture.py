import os
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import SpectralClustering
from sklearn.metrics import classification_report, accuracy_score
import lzma

class InformationFractureDetector:
    """
    Superior Solution: Information Fracture Detection.
    Combines Spectral Topology with Markovian Entropy Shifts.
    """
    def __init__(self):
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))

    def get_kl_divergence_proxy(self, t1, t2):
        # We use LZMA as a proxy for cross-entropy (KL Divergence)
        c1 = len(lzma.compress(t1.encode()))
        c2 = len(lzma.compress(t2.encode()))
        c12 = len(lzma.compress((t1 + " " + t2).encode()))
        # Cross-entropy proxy: how much did adding T2 increase the dictionary of T1?
        return (c12 - c1) / (c2 + 1e-6)

    def detect_changes(self, lines):
        n = len(lines)
        if n < 3: return [0] * (n - 1)
        
        # 1. Global Topology (Spectral)
        X = self.vectorizer.fit_transform(lines)
        adj = (X * X.T).toarray()
        sc = SpectralClustering(n_clusters=2, affinity='precomputed', n_init=20, random_state=42)
        global_labels = sc.fit_predict(adj)
        
        # 2. Local Entropy Fracture (LZMA)
        kl_shifts = []
        for i in range(n-1):
            kl_shifts.append(self.get_kl_divergence_proxy(lines[i], lines[i+1]))
        
        kl_threshold = np.mean(kl_shifts) + (1.2 * np.std(kl_shifts))
        
        # 3. Consensus Decision
        changes = []
        for i in range(n-1):
            is_global_break = 1 if global_labels[i] != global_labels[i+1] else 0
            is_local_fracture = 1 if kl_shifts[i] > kl_threshold else 0
            
            # UNCONVENTIONAL: An authorship shift is a Global topological break
            # that is verified by a Local information fracture.
            changes.append(1 if (is_global_break and is_local_fracture) else 0)
            
        return changes

def run_abs_max_test(data_dir, subset='hard', limit=100):
    detector = InformationFractureDetector()
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
            
    print(f"\nINFORMATION FRACTURE DETECTION (Subset: {subset.upper()}):")
    print(classification_report(y_true, y_pred))

if __name__ == "__main__":
    data_dir = "style-change/data/extracted/mawsa26-pan-zenodo"
    run_abs_max_test(data_dir, 'hard', 100)
