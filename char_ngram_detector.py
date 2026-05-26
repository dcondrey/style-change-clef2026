import os
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report, accuracy_score

class CharNgramStyleDetector:
    """
    The 'Subconscious Fingerprint' approach.
    Uses TF-IDF of character n-grams (sizes 3 to 5) to capture typing habits,
    punctuation rhythms, and suffix usage, entirely ignoring topic/semantics.
    """
    def __init__(self):
        # We analyze characters, keeping spaces to catch spacing habits.
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), lowercase=False)

    def detect_changes(self, lines, global_threshold_multiplier=1.2):
        if len(lines) < 2:
            return [0] * (len(lines) - 1)
            
        try:
            # Transform all lines into char n-gram vectors
            tfidf_matrix = self.vectorizer.fit_transform(lines)
        except ValueError:
            return [0] * (len(lines) - 1)
            
        similarities = []
        for i in range(len(lines) - 1):
            sim = cosine_similarity(tfidf_matrix[i], tfidf_matrix[i+1])[0][0]
            similarities.append(sim)
            
        # Dynamic Thresholding: A shift is a sudden DROP in similarity 
        # compared to the document's baseline rhythm.
        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)
        
        # We flag a change if the similarity drops significantly below the mean
        threshold = mean_sim - (global_threshold_multiplier * std_sim)
        
        changes = []
        for sim in similarities:
            if sim < threshold:
                changes.append(1)
            else:
                changes.append(0)
                
        return changes

def evaluate_char_ngram(data_dir, subset='hard', limit=100):
    detector = CharNgramStyleDetector()
    val_dir = os.path.join(data_dir, subset, 'validation')
    problems = sorted([f for f in os.listdir(val_dir) if f.startswith('problem-') and f.endswith('.txt')])[:limit]
    
    y_true, y_pred = [], []
    for prob_file in problems:
        with open(os.path.join(val_dir, prob_file), 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        with open(os.path.join(val_dir, 'truth-' + prob_file.replace('.txt', '.json')), 'r') as f:
            true_changes = json.load(f)['changes']
        
        if len(lines) < 2: continue
        
        pred = detector.detect_changes(lines, global_threshold_multiplier=0.8) # Tunable parameter
        
        min_len = min(len(pred), len(true_changes))
        y_pred.extend(pred[:min_len])
        y_true.extend(true_changes[:min_len])
        
    print(f"\nCHARACTER N-GRAM FINGERPRINTING (Subset: {subset.upper()}):")
    print(classification_report(y_true, y_pred))
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")

if __name__ == "__main__":
    data_dir = "style-change/data/extracted/mawsa26-pan-zenodo"
    # Evaluate on the hardest set
    evaluate_char_ngram(data_dir, 'hard', 200)
