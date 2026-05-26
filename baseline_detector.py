import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter

class StyleChangeDetector:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def split_sentences(self, text):
        return [s.strip() for s in text.splitlines() if s.strip()]

    def get_style_vector(self, sentences):
        # Extract features for each sentence
        features = []
        for s in sentences:
            length = len(s.split())
            punct_count = sum(1 for char in s if char in '.,;:?!')
            punct_ratio = punct_count / (len(s) + 1)
            caps_count = sum(1 for char in s if char.isupper())
            caps_ratio = caps_count / (len(s) + 1)
            features.append([length, punct_ratio, caps_ratio])
        return np.array(features)

    def get_function_words(self, text):
        # Top 30 most common function words for stylometry
        function_words = ['the', 'and', 'to', 'of', 'a', 'in', 'that', 'it', 'is', 'was', 'for', 'on', 'with', 'as', 'by', 'at', 'be', 'this', 'had', 'not', 'are', 'but', 'from', 'or', 'which', 'an', 'been', 'were', 'one', 'all']
        words = re.findall(r'\b\w+\b', text.lower())
        word_counts = Counter(words)
        total_words = len(words) + 1
        return np.array([word_counts.get(fw, 0) / total_words for fw in function_words])

    def detect_changes(self, text, threshold=0.45):
        # We will now use original direct mapping (one line = one change index)
        lines = [s.strip() for s in text.splitlines() if s.strip()]
        if len(lines) <= 1:
            return [0] * (len(lines) - 1)
        
        # Style Profile: Embeddings + Function Words + Punctuation
        embeddings = self.model.encode(lines)
        func_word_vecs = [self.get_function_words(l) for l in lines]
        style_feats = self.get_style_vector(lines)
        
        changes = []
        for i in range(len(lines) - 1):
            sem_sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
            func_sim = cosine_similarity([func_word_vecs[i]], [func_word_vecs[i+1]])[0][0]
            
            # Distance in stylistic markers (length, punct, caps)
            style_dist = np.linalg.norm(style_feats[i] - style_feats[i+1])
            
            # FINAL HEURISTIC: Weight semantic vs stylistic
            # Author shift usually breaks both topic flow and function word patterns
            is_change = 0
            if (sem_sim < 0.35) or (func_sim < 0.2) or (style_dist > 3.0):
                is_change = 1
            changes.append(is_change)
        
        return changes

def evaluate_on_subset(data_dir, subset='easy', limit=100):
    detector = StyleChangeDetector()
    val_dir = os.path.join(data_dir, subset, 'validation')
    
    problems = [f for f in os.listdir(val_dir) if f.startswith('problem-') and f.endswith('.txt')]
    problems = sorted(problems)[:limit]
    
    total_f1 = 0
    for prob_file in problems:
        with open(os.path.join(val_dir, prob_file), 'r') as f:
            text = f.read()
        
        truth_file = 'truth-' + prob_file.replace('.txt', '.json')
        with open(os.path.join(val_dir, truth_file), 'r') as f:
            truth = json.load(f)
        
        pred_changes = detector.detect_changes(text)
        true_changes = truth['changes']
        
        # Ensure length match (sometimes sentence splitting differs)
        min_len = min(len(pred_changes), len(true_changes))
        if min_len > 0:
            match = sum(1 for p, t in zip(pred_changes[:min_len], true_changes[:min_len]) if p == t)
            acc = match / min_len
            total_f1 += acc # Simple accuracy for now
            
    print(f"Average Accuracy on {subset} (limit {limit}): {total_f1/limit:.4f}")

if __name__ == "__main__":
    data_dir = "style-change/data/extracted/mawsa26-pan-zenodo"
    evaluate_on_subset(data_dir, subset='easy', limit=50)
    evaluate_on_subset(data_dir, subset='hard', limit=50)
