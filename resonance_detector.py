import os
import json
import numpy as np
import re
from sklearn.metrics import classification_report, accuracy_score

class IdentityResonanceDetector:
    """
    Unconventional Solution: The 'Author's Breath'.
    Ignores semantics entirely. Focuses on the subconscious rhythmic intervals
    of punctuation, subclauses, and word-length entropy.
    """
    def __init__(self):
        pass

    def extract_breath_signal(self, lines):
        # We transform each line into a 'Structural DNA' vector
        signal = []
        for line in lines:
            # 1. Inter-Punctuation Interval (Characters between pauses)
            punctuation_marks = [m.start() for m in re.finditer(r'[,;:.?!]', line)]
            if len(punctuation_marks) > 1:
                ipi = np.mean(np.diff(punctuation_marks))
            else:
                ipi = len(line)
                
            # 2. Subclause Density (Frequency of conjunctions/relative pronouns)
            conjunctions = len(re.findall(r'\b(and|but|or|which|who|that|because|although|if)\b', line.lower()))
            
            # 3. Word Length Entropy (Subconscious vocabulary 'weight')
            words = line.split()
            word_lengths = [len(w) for w in words if len(w) > 0]
            w_entropy = np.std(word_lengths) if word_lengths else 0
            
            signal.append([ipi, conjunctions, w_entropy])
        return np.array(signal)

    def detect_changes(self, lines):
        n = len(lines)
        if n < 3: return [0] * (n - 1)
        
        signal = self.extract_breath_signal(lines)
        
        # We use Bayesian Change-Point detection logic:
        # Is the 'Next' sentence a statistical anomaly compared to the 'Cumulative' breath?
        changes = []
        author_profile = [signal[0]]
        
        for i in range(1, n):
            current_sentence = signal[i]
            profile_mean = np.mean(author_profile, axis=0)
            profile_std = np.std(author_profile, axis=0) + 1e-6
            
            # Z-Score of the current sentence's 'Breath' vs the Author's profile
            z_scores = np.abs((current_sentence - profile_mean) / profile_std)
            resonance_break = np.mean(z_scores)
            
            # UNCONVENTIONAL THRESHOLD:
            # If the resonance break is > 2.5 sigma, the 'Breath' has changed.
            is_change = 1 if resonance_break > 2.5 else 0
            
            if is_change:
                # Reset profile for new author
                author_profile = [current_sentence]
            else:
                # Accumulate current author's habits
                author_profile.append(current_sentence)
                if len(author_profile) > 15: author_profile.pop(0) # Keep it local
                
            changes.append(is_change)
            
        return changes

def run_resonance_test(data_dir, subset='hard', limit=100):
    detector = IdentityResonanceDetector()
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
            
    print(f"\nIDENTITY RESONANCE 'BREATH' ANALYSIS (Subset: {subset.upper()}):")
    print(classification_report(y_true, y_pred))

if __name__ == "__main__":
    data_dir = "style-change/data/extracted/mawsa26-pan-zenodo"
    run_resonance_test(data_dir, 'hard', 200)
