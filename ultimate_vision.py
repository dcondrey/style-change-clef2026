import os
import json
import numpy as np
import re
from sklearn.metrics import classification_report, accuracy_score
from scipy import ndimage

class StyleVisionDetector:
    """
    Superior Solution: Structural Self-Similarity Matrix (SSM) Edge Detection.
    Treats the document as a high-dimensional manifold and uses Computer Vision 
    kernels to find authorship 'Fracture Lines'.
    """
    def __init__(self):
        pass

    def extract_structural_dna(self, line):
        # 1. Micro-Punctuation Profile
        punct_marks = re.findall(r'[,;:.?!]', line)
        p_count = len(punct_marks)
        
        # 2. Function Word Rhythm (The 'Pulse')
        words = line.lower().split()
        f_words = ['the', 'of', 'and', 'to', 'in', 'is', 'that', 'it', 'was', 'for', 'as', 'with']
        f_counts = [words.count(fw) for fw in f_words]
        
        # 3. Subclause Complexity
        subclauses = len(re.findall(r'\b(which|that|who|because|although|if|when|while)\b', line.lower()))
        
        # 4. Character Velocity
        chars_per_word = len(line) / (len(words) + 1)
        
        return np.array([p_count, subclauses, chars_per_word] + f_counts)

    def detect_changes(self, lines):
        n = len(lines)
        if n < 5: return [0] * (n - 1)
        
        # 1. Construct the Manifold
        dna_vectors = np.array([self.extract_structural_dna(l) for l in lines])
        
        # 2. Generate the Self-Similarity Matrix (SSM)
        # Treat this as a grayscale image of author consistency
        dna_norm = dna_vectors / (np.linalg.norm(dna_vectors, axis=1, keepdims=True) + 1e-6)
        ssm = np.dot(dna_norm, dna_norm.T)
        
        # 3. Gaussian Edge Detection (The Sobel Kernel)
        # We look for 'Blocks' along the diagonal. A break in the block is a shift.
        # We apply a gradient filter to the similarity matrix
        grad = ndimage.sobel(ssm)
        
        # 4. Diagonal Fracture Analysis
        # We look at the 'Energy' of the gradient along the immediate off-diagonal
        fracture_energy = np.abs(np.diagonal(grad, offset=1))
        
        # UNCONVENTIONAL THRESHOLDING:
        # A shift is a point where the local 'Fracture Energy' exceeds 
        # the global background noise of the author's variability.
        threshold = np.mean(fracture_energy) + (1.8 * np.std(fracture_energy))
        
        changes = []
        for energy in fracture_energy:
            is_change = 1 if energy > threshold else 0
            changes.append(is_change)
            
        return changes

def run_ultimate_vision_test(data_dir, subset='hard', limit=100):
    detector = StyleVisionDetector()
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
            
    print(f"\nULTIMATE STYLE VISION (SSM EDGE DETECTION) - Subset: {subset.upper()}:")
    print(classification_report(y_true, y_pred))
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")

if __name__ == "__main__":
    data_dir = "style-change/data/extracted/mawsa26-pan-zenodo"
    # Target the HARD set specifically
    run_ultimate_vision_test(data_dir, 'hard', 300)
