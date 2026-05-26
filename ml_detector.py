import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import re
from collections import Counter

class StyleChangeML:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.embed_model = SentenceTransformer(model_name)
        self.clf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)

    def get_style_features(self, s):
        length = len(s.split())
        punct_count = sum(1 for char in s if char in '.,;:?!')
        punct_ratio = punct_count / (len(s) + 1)
        caps_count = sum(1 for char in s if char.isupper())
        caps_ratio = caps_count / (len(s) + 1)
        
        # Function words
        fwords = ['the', 'and', 'to', 'of', 'a', 'in', 'that', 'it', 'is', 'was', 'for', 'on', 'with', 'as', 'by', 'at', 'be', 'this', 'had', 'not', 'are', 'but', 'from', 'or', 'which', 'an', 'been', 'were', 'one', 'all']
        words = re.findall(r'\b\w+\b', s.lower())
        word_counts = Counter(words)
        total_words = len(words) + 1
        fw_feats = [word_counts.get(fw, 0) / total_words for fw in fwords]
        
        return [length, punct_ratio, caps_ratio] + fw_feats

    def extract_pair_features(self, s1, s2, e1, e2):
        style1 = self.get_style_features(s1)
        style2 = self.get_style_features(s2)
        sem_sim = cosine_similarity([e1], [e2])[0][0]
        diffs = [abs(a - b) for a, b in zip(style1, style2)]
        return [sem_sim] + diffs

    def extract_contextual_features(self, embeddings, lines, i):
        f_pair = self.extract_pair_features(lines[i], lines[i+1], embeddings[i], embeddings[i+1])
        prev_sim, next_sim = 1.0, 1.0
        if i > 0:
            prev_sim = cosine_similarity([embeddings[i-1]], [embeddings[i]])[0][0]
        if i < len(embeddings) - 2:
            next_sim = cosine_similarity([embeddings[i+1]], [embeddings[i+2]])[0][0]
        return f_pair + [prev_sim, next_sim]

    def prepare_dataset(self, data_dir, subset='easy', limit=200):
        X, y = [], []
        val_dir = os.path.join(data_dir, subset, 'validation')
        problems = sorted([f for f in os.listdir(val_dir) if f.startswith('problem-') and f.endswith('.txt')])[:limit]
        
        for prob_file in problems:
            with open(os.path.join(val_dir, prob_file), 'r') as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
            
            truth_file = 'truth-' + prob_file.replace('.txt', '.json')
            with open(os.path.join(val_dir, truth_file), 'r') as f:
                true_changes = json.load(f)['changes']
            
            if len(lines) < 2: continue
            
            embeddings = self.embed_model.encode(lines)
            for i in range(min(len(lines) - 1, len(true_changes))):
                X.append(self.extract_contextual_features(embeddings, lines, i))
                y.append(true_changes[i])
                
        return np.array(X), np.array(y)

if __name__ == "__main__":
    data_dir = "style-change/data/extracted/mawsa26-pan-zenodo"
    sc = StyleChangeML()
    
    print("Preparing Training Data (Easy)...")
    X_train, y_train = sc.prepare_dataset(data_dir, subset='easy', limit=100)
    
    print("Preparing Test Data (Hard)...")
    X_test, y_test = sc.prepare_dataset(data_dir, subset='hard', limit=50)
    
    print(f"Training on {len(X_train)} transitions...")
    sc.clf.fit(X_train, y_train)
    
    y_pred = sc.clf.predict(X_test)
    print("\nEvaluation on HARD subset:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
