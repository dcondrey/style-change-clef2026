import os
import json
import numpy as np
import zlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score

class ResonanceConsensusDetector:
    """
    Final Absolute Max Architecture: The Resonance Consensus.
    Integrates Global Topological Stability with Local Information Entropy.
    """
    def __init__(self):
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4))

    def get_entropy_spike(self, t1, t2):
        # LZMA-based Symbolic Complexity spike
        c1 = len(zlib.compress(t1.encode()))
        c12 = len(zlib.compress((t1 + " " + t2).encode()))
        return (c12 - c1) / (len(t2) + 1)

    def detect_changes(self, lines):
        n = len(lines)
        if n < 6: return [0] * (n - 1)
        
        # 1. Topology
        X = self.vectorizer.fit_transform(lines).toarray()
        
        # 2. Extract Resonance Signals
        global_energy_shifts = []
        local_entropy_spikes = []
        
        for i in range(1, n - 1):
            # Global: SVD Energy shift
            sub = X[max(0, i-5):i]
            combined = X[max(0, i-5):i+1]
            s_sub = np.linalg.svd(sub, compute_uv=False)
            s_comb = np.linalg.svd(combined, compute_uv=False)
            
            # Normalize and compare top components
            e_sub = s_sub[:3] / (np.sum(s_sub) + 1e-6)
            e_comb = s_comb[:3] / (np.sum(s_comb) + 1e-6)
            # Pad if necessary
            e_sub_p = np.pad(e_sub, (0, 3-len(e_sub)))
            e_comb_p = np.pad(e_comb, (0, 3-len(e_comb)))
            
            global_energy_shifts.append(np.sum(np.abs(e_comb_p - e_sub_p)))
            
            # Local: Entropy fracture
            local_entropy_spikes.append(self.get_entropy_spike(lines[i-1], lines[i]))
            
        # 3. Consensus Logic
        # A shift must break BOTH the global subspace and local entropy dictionary
        e_thresh = np.mean(global_energy_shifts) + (1.5 * np.std(global_energy_shifts))
        s_thresh = np.mean(local_entropy_spikes) + (1.5 * np.std(local_entropy_spikes))
        
        changes = []
        for i in range(len(global_energy_shifts)):
            # The 'Absolute Max' Filter: Two-Lock Consensus
            if global_energy_shifts[i] > e_thresh and local_entropy_spikes[i] > s_thresh:
                changes.append(1)
            else:
                changes.append(0)
                
        # Padding
        while len(changes) < n - 1: changes.append(0)
        return changes

def evaluate_absolute_max(data_dir, subset='hard', limit=100):
    detector = ResonanceConsensusDetector()
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
            
    print(f"\nRESONANCE CONSENSUS ABSOLUTE MAX (Subset: {subset.upper()}):")
    print(classification_report(y_true, y_pred))
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")

if __name__ == "__main__":
    data_dir = "style-change/data/extracted/mawsa26-pan-zenodo"
    evaluate_absolute_max(data_dir, 'hard', 300)
