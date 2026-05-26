import os
import json
import requests
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "phi3:mini"

class NeuralShadowDetector:
    """
    Absolute Max Superior Solution: The Neural Shadow.
    Uses a local LLM to probe the 'Cognitive Expectation' of the next sentence.
    An authorship shift is a 'Cognitive Reset' where the next sentence 
    fundamentally breaks the stylistic and logical manifold of the current author.
    """
    def __init__(self):
        pass

    def get_perplexity_proxy(self, context, target):
        """
        Asks the model how 'surprising' the target sentence is given the context.
        We use a high-precision prompt to get a numerical log-likelihood proxy.
        """
        prompt = f"""Task: Style Continuity Analysis.
Current Author's Style:
\"\"\"{context}\"\"\"

Next Sentence to Evaluate:
\"\"\"{target}\"\"\"

On a scale of 0 to 100, how likely is it that the SAME author wrote the Next Sentence? 
(0 = Completely different author/style, 100 = Identical author/style)
Consider sentence length, punctuation habits, and vocabulary flow.
Respond with ONLY the number:"""

        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 5}
                },
                timeout=30
            )
            res_text = response.json().get("response", "").strip()
            # Extract digits
            nums = re.findall(r'\d+', res_text)
            return int(nums[0]) if nums else 50
        except:
            return 50

    def detect_changes(self, lines, threshold=40):
        if len(lines) < 2: return [0] * (len(lines) - 1)
        
        changes = []
        # We use the last 2 sentences as context to establish the author's 'Shadow'
        for i in range(1, len(lines)):
            context = " ".join(lines[max(0, i-2):i])
            target = lines[i]
            
            continuity_score = self.get_perplexity_proxy(context, target)
            
            # A score below threshold indicates a 'Neural Shadow' mismatch (Author Shift)
            is_change = 1 if continuity_score < threshold else 0
            changes.append(is_change)
            
        return changes

import re

def evaluate_shadow(data_dir, subset='hard', limit=20):
    # Limit is small because LLM inference is slow
    detector = NeuralShadowDetector()
    val_dir = os.path.join(data_dir, subset, 'validation')
    problems = sorted([f for f in os.listdir(val_dir) if f.startswith('problem-') and f.endswith('.txt')])[:limit]
    
    y_true, y_pred = [], []
    print(f"Executing Neural Shadow Probe on {subset.upper()} set (limit {limit})...")
    
    for prob_file in tqdm(problems):
        with open(os.path.join(val_dir, prob_file), 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        with open(os.path.join(val_dir, 'truth-' + prob_file.replace('.txt', '.json')), 'r') as f:
            true_changes = json.load(f)['changes']
            
        if len(lines) < 2: continue
        
        pred = detector.detect_changes(lines)
        min_len = min(len(pred), len(true_changes))
        y_pred.extend(pred[:min_len])
        y_true.extend(true_changes[:min_len])
        
    print(f"\nNEURAL SHADOW EVALUATION ({subset.upper()}):")
    print(classification_report(y_true, y_pred))

if __name__ == "__main__":
    data_dir = "style-change/data/extracted/mawsa26-pan-zenodo"
    # Test connection first
    try:
        requests.get("http://localhost:11434/", timeout=5)
        evaluate_shadow(data_dir, 'hard', limit=10)
    except:
        print("ERROR: Ollama not reachable. Skipping Neural Shadow probe.")
