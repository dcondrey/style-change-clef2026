import os
import json
import zlib
import numpy as np
from sklearn.metrics import classification_report, accuracy_score

class CompressionStyleDetector:
    """
    Uses Normalized Compression Distance (NCD) to detect authorship shifts.
    NCD(x,y) = (C(xy) - min(C(x), C(y))) / max(C(x), C(y))
    Zero training required. Immune to overfitting.
    """
    def __init__(self, compression_level=9):
        self.level = compression_level

    def compress_size(self, text):
        # We encode to bytes and compress
        return len(zlib.compress(text.encode('utf-8'), self.level))

    def ncd(self, text1, text2):
        c1 = self.compress_size(text1)
        c2 = self.compress_size(text2)
        c12 = self.compress_size(text1 + " " + text2)
        
        return (c12 - min(c1, c2)) / max(c1, c2)

    def detect_changes(self, lines, threshold=0.85):
        if len(lines) < 2:
            return [0] * (len(lines) - 1)
            
        changes = []
        
        # We compare a sliding context window of the current author to the next sentence
        # to give the compressor enough 'dictionary' to work with.
        context_window = lines[0]
        
        for i in range(1, len(lines)):
            target_sentence = lines[i]
            
            # Calculate NCD between current author's context and the new sentence
            distance = self.ncd(context_window, target_sentence)
            
            # Also calculate local NCD (just sentence to sentence)
            local_distance = self.ncd(lines[i-1], target_sentence)
            
            # If the compression distance spikes, it means the new sentence 
            # uses a fundamentally different vocabulary/structure pattern.
            is_change = 0
            if local_distance > threshold and distance > threshold - 0.05:
                is_change = 1
                context_window = target_sentence # Reset context
            else:
                # Add to context, keeping it bounded to prevent infinite growth
                context_window += " " + target_sentence
                if len(context_window) > 1000:
                    context_window = context_window[-1000:]
                    
            changes.append(is_change)
            
        return changes

def evaluate_ncd(data_dir, subset='hard', limit=100):
    detector = CompressionStyleDetector()
    val_dir = os.path.join(data_dir, subset, 'validation')
    problems = sorted([f for f in os.listdir(val_dir) if f.startswith('problem-') and f.endswith('.txt')])[:limit]
    
    y_true, y_pred = [], []
    for prob_file in problems:
        with open(os.path.join(val_dir, prob_file), 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        with open(os.path.join(val_dir, 'truth-' + prob_file.replace('.txt', '.json')), 'r') as f:
            true_changes = json.load(f)['changes']
        
        if len(lines) < 2: continue
        
        # Dynamically adjust threshold based on document average NCD to handle different baseline entropy
        local_ncds = [detector.ncd(lines[i], lines[i+1]) for i in range(len(lines)-1)]
        doc_threshold = np.mean(local_ncds) + (0.5 * np.std(local_ncds))
        
        pred = detector.detect_changes(lines, threshold=doc_threshold)
        
        min_len = min(len(pred), len(true_changes))
        y_pred.extend(pred[:min_len])
        y_true.extend(true_changes[:min_len])
        
    print(f"\nInformation Theory NCD Evaluation ({subset.upper()} subset):")
    print(classification_report(y_true, y_pred))
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")

if __name__ == "__main__":
    data_dir = "style-change/data/extracted/mawsa26-pan-zenodo"
    evaluate_ncd(data_dir, 'hard', 100)
