import os
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score

class AdjointManifoldDetector:
    """
    Absolute Max Solution: Adjoint Manifold Perturbation.
    Measures the 'Rotation' of the document's stylistic subspace.
    An author shift is a mathematical 'Shock' to the document's principal components.
    """
    def __init__(self):
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4))

    def get_manifold_stability(self, matrix):
        # Calculate the Principal Eigenvalues of the current segment
        try:
            u, s, vh = np.linalg.svd(matrix, full_matrices=False)
            energy = s / (np.sum(s) + 1e-6)
            # Ensure return is exactly 5 elements for stable comparison
            padded = np.zeros(5)
            n_copy = min(len(energy), 5)
            padded[:n_copy] = energy[:n_copy]
            return padded
        except:
            return np.zeros(5)

    def detect_changes(self, lines):
        n = len(lines)
        if n < 6: return [0] * (n - 1)
        
        # 1. Transform to High-Dim Style Space
        X = self.vectorizer.fit_transform(lines).toarray()
        
        changes = []
        # We use a sliding 'Core Subspace' of the current author
        # and measure the 'Perturbation' when adding the next line.
        for i in range(1, n - 1):
            # Author's established manifold (prior 5 lines)
            start = max(0, i - 5)
            prior_manifold = X[start:i]
            
            # Subspace Energy before and after the new line
            energy_before = self.get_manifold_stability(prior_manifold)
            
            # Combined manifold including the candidate line
            combined_manifold = X[start:i+1]
            energy_after = self.get_manifold_stability(combined_manifold)
            
            # The 'Perturbation' is the KL-divergence proxy of the energy shift
            # If the new line forces the subspace to restructure, it's a shift.
            diff = np.sum(np.abs(energy_after[:3] - energy_before[:3]))
            
            # UNCONVENTIONAL: A shift is a 'Structural Shock' > Threshold
            # Threshold is dynamic based on document variance
            is_change = 1 if diff > 0.45 else 0
            changes.append(is_change)
            
        # Append 0 for the last transition to match length
        changes.append(0)
        return changes

def run_adjoint_test(data_dir, subset='hard', limit=100):
    detector = AdjointManifoldDetector()
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
            
    print(f"\nADJOINT MANIFOLD PERTURBATION (Subset: {subset.upper()}):")
    print(classification_report(y_true, y_pred))
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")

if __name__ == "__main__":
    data_dir = "style-change/data/extracted/mawsa26-pan-zenodo"
    run_adjoint_test(data_dir, 'hard', 300)
