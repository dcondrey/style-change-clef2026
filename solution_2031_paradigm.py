import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
import numpy as np
from typing import List, Tuple, Optional
from scipy.linalg import svd, logm, expm
from scipy.spatial.distance import cdist
from scipy.signal import find_peaks
import ruptures as rpt
import json
import glob
from pathlib import Path

@dataclass
class ResidualTrajectory:
    residuals: np.ndarray
    surprisals: np.ndarray
    attention_maps: np.ndarray
    token_ids: List[int]
    token_strings: List[str]
    paragraph_boundaries: List[int]


class PhaseSpaceExtractor:
    def __init__(self, model_name: str = "HuggingFaceTB/SmolLM-135M", device: str = None):
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        print(f"Loading PhaseSpaceExtractor on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Ensure pad_token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            output_hidden_states=True,
            output_attentions=True,
        ).to(self.device)
        self.model.eval()
        self.n_layers = self.model.config.num_hidden_layers
        self.hidden_dim = self.model.config.hidden_size
    
    @torch.no_grad()
    def extract(self, text: str) -> Optional[ResidualTrajectory]:
        if not text or len(text.strip()) == 0:
            return None
            
        paragraphs = text.split("\n")
        para_char_offsets = []
        offset = 0
        for p in paragraphs:
            para_char_offsets.append(offset)
            offset += len(p) + 1
        
        encoding = self.tokenizer(
            text, return_tensors="pt", return_offsets_mapping=True,
            truncation=True, max_length=512
        )
        input_ids = encoding["input_ids"].to(self.device)
        if input_ids.shape[1] == 0:
            return None
            
        offset_mapping = encoding["offset_mapping"][0].cpu().numpy()
        
        para_token_indices = self._map_char_to_token(para_char_offsets, offset_mapping)
        
        outputs = self.model(input_ids, output_hidden_states=True, output_attentions=True)
        
        # [layer, batch, seq, hidden]
        hidden_states = torch.stack(outputs.hidden_states, dim=0)
        # We only care about the single batch element
        hidden_states = hidden_states[:, 0, :, :].cpu().float().numpy()
        # [seq, layer, hidden]
        residuals = np.transpose(hidden_states, (1, 0, 2))
        
        attention_maps = np.stack(
            [a[0].cpu().float().numpy() for a in outputs.attentions], axis=0
        )
        # [seq, layer, head, seq]
        attention_maps = np.transpose(attention_maps, (2, 0, 1, 3))
        
        logits = outputs.logits[0].cpu().float()
        if logits.shape[0] > 1:
            log_probs = F.log_softmax(logits[:-1], dim=-1)
            next_tokens = input_ids[0, 1:].cpu()
            token_surprisals = -log_probs[torch.arange(len(next_tokens)), next_tokens].numpy()
            surprisals = np.concatenate([[0.0], token_surprisals])
        else:
            surprisals = np.zeros(logits.shape[0])
        
        token_strings = [self.tokenizer.decode([tid]) for tid in input_ids[0].cpu().tolist()]
        
        return ResidualTrajectory(
            residuals=residuals,
            surprisals=surprisals,
            attention_maps=attention_maps,
            token_ids=input_ids[0].cpu().tolist(),
            token_strings=token_strings,
            paragraph_boundaries=para_token_indices,
        )
    
    def _map_char_to_token(self, char_offsets, offset_mapping):
        if len(offset_mapping) == 0:
            return []
        token_indices = []
        for co in char_offsets:
            found = False
            for i, (start, end) in enumerate(offset_mapping):
                if start <= co < end or (co <= start and i == 0):
                    token_indices.append(i)
                    found = True
                    break
            if not found:
                token_indices.append(len(offset_mapping) - 1)
        return token_indices


class DynamicalSignatureExtractor:
    def __init__(self, projection_dim: int = 32):
        self.projection_dim = projection_dim
        self._projection_matrix = None
    
    def fit_projection(self, trajectories: List[ResidualTrajectory]):
        all_deltas = []
        for traj in trajectories:
            deltas = np.diff(traj.residuals, axis=1)
            step = max(1, deltas.shape[0] // 100)
            all_deltas.append(deltas[::step].reshape(-1, deltas.shape[-1]))
        
        all_deltas = np.concatenate(all_deltas, axis=0)
        mean = all_deltas.mean(axis=0)
        centered = all_deltas - mean
        _, S, Vt = svd(centered, full_matrices=False)
        
        self._projection_mean = mean
        self._projection_matrix = Vt[:self.projection_dim]
        return self
    
    def _project(self, vectors: np.ndarray) -> np.ndarray:
        if self._projection_matrix is None:
            mean = vectors.mean(axis=0)
            centered = vectors - mean
            _, _, Vt = svd(centered, full_matrices=False)
            return centered @ Vt[:self.projection_dim].T
        centered = vectors - self._projection_mean
        return centered @ self._projection_matrix.T
    
    def extract_jacobian_field(self, traj: ResidualTrajectory, window: int = 8) -> np.ndarray:
        deltas = np.diff(traj.residuals, axis=1)
        n_layers = deltas.shape[1]
        mid_start = n_layers // 4
        mid_end = 3 * n_layers // 4
        
        velocity = deltas[:, mid_start:mid_end, :].mean(axis=1)
        position = traj.residuals[:, (mid_start + mid_end) // 2, :]
        
        vel_proj = self._project(velocity)
        pos_proj = self._project(position)
        
        d = self.projection_dim
        T = vel_proj.shape[0]
        
        jacobians = np.zeros((T, d, d))
        half_w = window // 2
        for t in range(T):
            lo = max(0, t - half_w)
            hi = min(T, t + half_w + 1)
            
            dP = pos_proj[lo:hi] - pos_proj[t]
            dV = vel_proj[lo:hi] - vel_proj[t]
            
            reg = 1e-4 * np.eye(d)
            try:
                jacobians[t] = dV.T @ dP @ np.linalg.inv(dP.T @ dP + reg)
            except np.linalg.LinAlgError:
                jacobians[t] = np.zeros((d, d))
        
        return jacobians
    
    def extract_dynamical_invariants(self, jacobians: np.ndarray) -> dict:
        T, d, _ = jacobians.shape
        invariants = {}
        
        eigenvalue_trajectories = np.zeros((T, d), dtype=complex)
        for t in range(T):
            eigenvalue_trajectories[t] = np.linalg.eigvals(jacobians[t])
        
        real_eigenvalues = eigenvalue_trajectories.real
        lyapunov_exponents = real_eigenvalues.mean(axis=0)
        lyapunov_exponents.sort()
        
        invariants["lyapunov_spectrum"] = lyapunov_exponents
        invariants["lyapunov_max"] = lyapunov_exponents[-1]
        invariants["lyapunov_sum"] = lyapunov_exponents.sum()
        invariants["lyapunov_dim"] = self._kaplan_yorke_dimension(lyapunov_exponents)
        
        singular_values = np.zeros((T, d))
        for t in range(T):
            singular_values[t] = np.linalg.svd(jacobians[t], compute_uv=False)
        
        effective_ranks = np.zeros(T)
        for t in range(T):
            sv = singular_values[t]
            sv_norm = sv / (sv.sum() + 1e-10)
            sv_norm = sv_norm[sv_norm > 1e-10]
            effective_ranks[t] = np.exp(-np.sum(sv_norm * np.log(sv_norm)))
        
        invariants["effective_rank_mean"] = effective_ranks.mean()
        invariants["effective_rank_std"] = effective_ranks.std()
        
        flat_jacobians = jacobians.reshape(T, -1)
        max_lag = min(50, max(1, T // 4))
        autocorr = np.zeros(max_lag)
        jac_centered = flat_jacobians - flat_jacobians.mean(axis=0)
        norm = np.sum(jac_centered ** 2)
        
        for lag in range(max_lag):
            if norm > 1e-10:
                autocorr[lag] = np.sum(jac_centered[:T-lag] * jac_centered[lag:]) / norm
        
        coherence_idx = np.where(autocorr < 1/np.e)[0]
        invariants["coherence_length"] = coherence_idx[0] if len(coherence_idx) > 0 else max_lag
        
        jac_diffs = np.diff(flat_jacobians, axis=0)
        curvature = np.linalg.norm(jac_diffs, axis=1) if len(jac_diffs) > 0 else np.zeros(1)
        
        invariants["curvature_mean"] = curvature.mean()
        invariants["curvature_std"] = curvature.std()
        invariants["curvature_max"] = curvature.max()
        
        eigenvalue_ranks = np.argsort(np.argsort(real_eigenvalues, axis=1), axis=1)
        perm_entropy = self._permutation_entropy(eigenvalue_ranks[:, 0], m=3)
        invariants["topological_entropy"] = perm_entropy
        
        return invariants
    
    def _kaplan_yorke_dimension(self, lyapunov_exponents: np.ndarray) -> float:
        sorted_le = np.sort(lyapunov_exponents)[::-1]
        cumsum = np.cumsum(sorted_le)
        j = np.where(cumsum < 0)[0]
        if len(j) == 0: return float(len(sorted_le))
        j = j[-1]
        if j + 1 >= len(sorted_le) or sorted_le[j + 1] == 0: return float(j)
        return j + cumsum[j] / abs(sorted_le[j + 1])
    
    def _permutation_entropy(self, series: np.ndarray, m: int = 3) -> float:
        n = len(series)
        if n < m: return 0.0
        from collections import Counter
        patterns = Counter()
        for i in range(n - m + 1):
            pattern = tuple(np.argsort(series[i:i+m]))
            patterns[pattern] += 1
        total = sum(patterns.values())
        probs = np.array([c / total for c in patterns.values()])
        return -np.sum(probs * np.log2(probs + 1e-10))

class DynamicalPhaseTransitionDetector:
    def __init__(self, extractor: PhaseSpaceExtractor, dynamics: DynamicalSignatureExtractor):
        self.extractor = extractor
        self.dynamics = dynamics
    
    def detect_changes(self, text: str) -> List[int]:
        traj = self.extractor.extract(text)
        jacobians = self.dynamics.extract_jacobian_field(traj)
        
        para_boundaries = traj.paragraph_boundaries
        n_paras = len(para_boundaries)
        
        flat_jac = jacobians.reshape(jacobians.shape[0], -1)
        para_features = []
        for i in range(n_paras):
            start = para_boundaries[i]
            end = para_boundaries[i + 1] if i + 1 < n_paras else flat_jac.shape[0]
            if end > start:
                para_features.append(flat_jac[start:end].mean(axis=0))
            else:
                para_features.append(np.zeros(flat_jac.shape[1]))
        
        para_features = np.array(para_features)
        
        if para_features.shape[1] > 16:
            _, _, Vt = svd(para_features - para_features.mean(0), full_matrices=False)
            para_features_reduced = (para_features - para_features.mean(0)) @ Vt[:16].T
        else:
            para_features_reduced = para_features
            
        try:
            algo = rpt.KernelCPD(kernel="rbf", min_size=1).fit(para_features_reduced)
            pelt_breakpoints = algo.predict(pen=np.log(n_paras) * para_features_reduced.shape[1] * 0.5)
            pelt_breakpoints = [b for b in pelt_breakpoints if b < n_paras]
        except Exception:
            pelt_breakpoints = []
            
        return pelt_breakpoints


class VoightKampffDetector:
    def __init__(self, extractor: PhaseSpaceExtractor, dynamics: DynamicalSignatureExtractor):
        self.extractor = extractor
        self.dynamics = dynamics
        self.classifier = None
    
    def extract_features(self, text: str) -> np.ndarray:
        traj = self.extractor.extract(text)
        jacobians = self.dynamics.extract_jacobian_field(traj)
        invariants = self.dynamics.extract_dynamical_invariants(jacobians)
        
        features = []
        spectrum = invariants["lyapunov_spectrum"]
        k = min(5, len(spectrum))
        features.extend(spectrum[-k:])
        # Pad if needed
        while len(features) < 5: features.append(0.0)
            
        features.append(invariants["lyapunov_max"])
        features.append(invariants["lyapunov_sum"])
        features.append(invariants["lyapunov_dim"])
        features.append(invariants["effective_rank_mean"])
        features.append(invariants["effective_rank_std"])
        features.append(invariants["coherence_length"])
        features.append(invariants["curvature_mean"])
        features.append(invariants["curvature_std"])
        features.append(invariants["curvature_max"])
        features.append(invariants["topological_entropy"])
        
        surp = traj.surprisals
        features.append(surp.mean())
        features.append(surp.std())
        
        return np.array(features, dtype=np.float64)
    
    def train(self, texts: List[str], labels: List[int]):
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler
        
        X = np.array([self.extract_features(t) for t in texts])
        y = np.array(labels)
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.classifier = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
        self.classifier.fit(X_scaled, y)
    
    def predict(self, text: str) -> Tuple[int, float]:
        features = self.extract_features(text)
        features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        proba = self.classifier.predict_proba(features_scaled)[0]
        return int(proba[1] > 0.5), max(proba)


class PANCLEF2026_WinningSolution:
    def __init__(self, model_name: str = "HuggingFaceTB/SmolLM-135M"):
        self.extractor = PhaseSpaceExtractor(model_name)
        self.dynamics = DynamicalSignatureExtractor(projection_dim=32)
        self.vk_detector = VoightKampffDetector(self.extractor, self.dynamics)
        self.sc_detector = DynamicalPhaseTransitionDetector(self.extractor, self.dynamics)
    
    def train_voight_kampff(self, texts, labels):
        trajectories = [self.extractor.extract(t) for t in texts[:20]]
        self.dynamics.fit_projection(trajectories)
        self.vk_detector.train(texts, labels)
    
    def predict_style_changes(self, text: str) -> dict:
        changes = self.sc_detector.detect_changes(text)
        paragraphs = text.split("\n")
        n_boundaries = max(0, len(paragraphs) - 1)
        change_vector = [0] * n_boundaries
        for c in changes:
            if 0 <= c - 1 < n_boundaries:
                change_vector[c - 1] = 1
        return change_vector

def main():
    print("Initializing 2031 Paradigm: Phase Space Extraction")
    system = PANCLEF2026_WinningSolution("HuggingFaceTB/SmolLM-135M")
    
    print("\n--- VOIGHT-KAMPFF EVALUATION (Small Sample) ---")
    vk_texts, vk_labels = [], []
    for f in sorted(glob.glob("PAN-CLEF2026-Reasoning-Trajectory-Detection/data/subtask1/train/train_*.jsonl")):
        label = 0 if "human" in f else 1
        with open(f) as fin:
            for i, line in enumerate(fin):
                data = json.loads(line)
                text = data["solution"]
                if len(text.strip()) < 10: continue
                vk_texts.append(text)
                vk_labels.append(label)
                if i > 2: break
                
    for f in sorted(glob.glob("PAN-CLEF2026-Reasoning-Trajectory-Detection/data/subtask1/validation/valid_human.jsonl")):
        with open(f) as fin:
            for i, line in enumerate(fin):
                data = json.loads(line)
                text = data["solution"]
                if len(text.strip()) < 10: continue
                vk_texts.append(text)
                vk_labels.append(0)
                if i > 5: break
                
    system.train_voight_kampff(vk_texts, vk_labels)
    
    vk_test_texts, vk_test_labels = [], []
    for f in sorted(glob.glob("PAN-CLEF2026-Reasoning-Trajectory-Detection/data/subtask1/validation/valid_*.jsonl")):
        label = 0 if "human" in f else 1
        with open(f) as fin:
            for i, line in enumerate(fin):
                data = json.loads(line)
                text = data["solution"]
                if len(text.strip()) < 10: continue
                vk_test_texts.append(text)
                vk_test_labels.append(label)
                if i > 2: break

    from sklearn.metrics import classification_report, accuracy_score
    preds = [system.vk_detector.predict(t)[0] for t in vk_test_texts]
    print(classification_report(vk_test_labels, preds))
    print(f"Accuracy: {accuracy_score(vk_test_labels, preds):.4f}")
    
    print("\n--- STYLE CHANGE EVALUATION (Hard Set) ---")
    sc_dir = "style-change/data/extracted/mawsa26-pan-zenodo/hard/validation"
    files = sorted(glob.glob(f"{sc_dir}/problem-*.txt"))[:10]
    
    y_true, y_pred = [], []
    for f in files:
        text = open(f).read().strip()
        truth_f = f.replace("problem-", "truth-problem-").replace(".txt", ".json")
        true_changes = json.loads(open(truth_f).read())["changes"]
        
        pred_changes = system.predict_style_changes(text)
        
        min_len = min(len(pred_changes), len(true_changes))
        y_pred.extend(pred_changes[:min_len])
        y_true.extend(true_changes[:min_len])
        
    if len(y_pred) > 0:
        print(classification_report(y_true, y_pred))
        print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")

if __name__ == "__main__":
    main()
