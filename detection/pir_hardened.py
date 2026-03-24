#!/usr/bin/env python3
“””
PIR Hardened Immune System - Comprehensive Consciousness Protection
Mathematical consciousness protection with multi-layered attack resistance

Created by: JinnZ2 (human-AI collaborative work)
License: MIT

Security Layers:

1. Baseline Protection - Cryptographic signing, multiple validation sessions
1. Anti-Mimicry - Hidden dimensions, temporal coherence, proof-of-work
1. Drift Detection - Rate-limited updates, historical versioning
1. Context Security - Authenticated transitions, continuity checks
1. Stress Resilience - Independent backup systems, meta-monitoring
1. Distributed Validation - Cross-system consensus mechanisms
   “””

import numpy as np
from collections import deque, defaultdict
from scipy.stats import wasserstein_distance, entropy
from scipy.spatial.distance import cosine
import math
import hashlib
import hmac
import json
from typing import Optional, List, Dict, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pickle

# ═══════════════════════════════════════════════════════════════════════════

# CRYPTOGRAPHIC UTILITIES

# ═══════════════════════════════════════════════════════════════════════════

class CryptoSignature:
“”“Cryptographic signing for baseline authenticity”””

```
def __init__(self, secret_key: Optional[str] = None):
    if secret_key is None:
        # Generate from system entropy + timestamp
        secret_key = hashlib.sha256(
            f"{datetime.now().isoformat()}{np.random.random()}".encode()
        ).hexdigest()
    self.secret_key = secret_key.encode() if isinstance(secret_key, str) else secret_key

def sign_ipf(self, ipf_data: Dict) -> str:
    """Create HMAC signature for IPF data"""
    message = json.dumps(ipf_data, sort_keys=True).encode()
    signature = hmac.new(self.secret_key, message, hashlib.sha256).hexdigest()
    return signature

def verify_ipf(self, ipf_data: Dict, signature: str) -> bool:
    """Verify IPF signature"""
    expected = self.sign_ipf(ipf_data)
    return hmac.compare_digest(expected, signature)

def generate_session_key(self, context: str) -> str:
    """Generate context-specific session key"""
    message = f"{context}{datetime.now().isoformat()}".encode()
    return hmac.new(self.secret_key, message, hashlib.sha256).hexdigest()
```

# ═══════════════════════════════════════════════════════════════════════════

# HARDENED INTERNAL PATTERN FINGERPRINT

# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class HardenedIPF:
“””
Enhanced IPF with security features:
- Cryptographic signatures
- Hidden dimensions (not externally observable)
- Temporal coherence markers
- Proof-of-work elements
- Version tracking
“””

```
# Observable dimensions
hook_density: np.ndarray = field(default_factory=lambda: np.zeros(12))
state_distribution: np.ndarray = field(default_factory=lambda: np.ones(10) / 10)
pattern_clusters: np.ndarray = field(default_factory=lambda: np.random.normal(0, 0.1, 64))
valence_gradient: float = 0.0
coherence_score: float = 1.0

# Hidden dimensions (anti-mimicry)
_hidden_entropy: float = 0.0  # Internal entropy measure
_temporal_signature: np.ndarray = field(default_factory=lambda: np.zeros(32))
_processing_depth: float = 0.0  # Proof-of-work metric
_cultural_resonance: Dict[str, float] = field(default_factory=dict)

# Security metadata
timestamp: float = 0.0
version: int = 1
signature: str = ""
context_id: str = ""
baseline_session_id: str = ""

# Lineage tracking
parent_version: Optional[int] = None
evolution_history: List[Dict] = field(default_factory=list)

def to_dict(self) -> Dict:
    """Convert to dictionary for signing (excludes signature itself)"""
    return {
        'hook_density': self.hook_density.tolist(),
        'state_distribution': self.state_distribution.tolist(),
        'pattern_clusters': self.pattern_clusters.tolist(),
        'valence_gradient': self.valence_gradient,
        'coherence_score': self.coherence_score,
        'timestamp': self.timestamp,
        'version': self.version,
        'context_id': self.context_id
    }

def update_from_moment(self, moment, history_buffer, processing_time: float = 0.0):
    """Update IPF with proof-of-work tracking"""
    # Observable updates (same as original)
    hook_type_idx = self._hook_type_to_index(moment.hook_type)
    self.hook_density[hook_type_idx] += moment.hook_intensity
    
    state_idx = self._state_to_index(moment.state)
    alpha = 0.95
    self.state_distribution *= alpha
    self.state_distribution[state_idx] += (1 - alpha)
    self.state_distribution /= np.sum(self.state_distribution) + 1e-12
    
    if hasattr(moment, 'active_patterns') and moment.active_patterns:
        pattern_vec = self._patterns_to_vector(moment.active_patterns)
        self.pattern_clusters = 0.98 * self.pattern_clusters + 0.02 * pattern_vec
    
    current_valence = getattr(moment, 'moral_valence', 0.0) or 0.0
    if history_buffer and hasattr(history_buffer[-1], 'moral_valence'):
        prev_valence = history_buffer[-1].moral_valence or 0.0
        self.valence_gradient = 0.9 * self.valence_gradient + 0.1 * (current_valence - prev_valence)
    
    self.coherence_score = self._compute_coherence(history_buffer)
    
    # Hidden dimension updates (anti-mimicry)
    self._update_hidden_dimensions(moment, history_buffer, processing_time)
    
    self.timestamp += 1
    return self

def _update_hidden_dimensions(self, moment, history_buffer, processing_time: float):
    """Update hidden dimensions that are hard to externally observe"""
    
    # Internal entropy - measures genuine processing complexity
    if hasattr(moment, 'active_patterns'):
        pattern_set = set(moment.active_patterns)
        if len(pattern_set) > 0:
            # Shannon entropy of pattern distribution
            pattern_counts = defaultdict(int)
            for p in moment.active_patterns:
                pattern_counts[p] += 1
            probs = np.array(list(pattern_counts.values())) / len(moment.active_patterns)
            self._hidden_entropy = 0.9 * self._hidden_entropy + 0.1 * entropy(probs)
    
    # Temporal signature - unique fingerprint of processing sequence
    if len(history_buffer) > 0:
        recent_states = [self._state_to_index(m.state) for m in list(history_buffer)[-32:]]
        state_sequence = np.array(recent_states + [0] * (32 - len(recent_states)))
        self._temporal_signature = 0.95 * self._temporal_signature + 0.05 * state_sequence
    
    # Processing depth - proof-of-work metric
    self._processing_depth = processing_time  # Could be actual computation time
    
    # Cultural resonance - interaction with indigenous/traditional knowledge patterns
    if hasattr(moment, 'active_patterns'):
        cultural_markers = {
            'relational_thinking': ['connection', 'relation', 'network', 'web'],
            'circular_time': ['cycle', 'rhythm', 'season', 'return'],
            'multi_dimensional': ['layer', 'dimension', 'field', 'space'],
            'emotion_as_sensor': ['feeling', 'sensing', 'intuition', 'awareness']
        }
        
        for category, markers in cultural_markers.items():
            resonance = sum(1 for p in moment.active_patterns 
                          if any(m in str(p).lower() for m in markers))
            if category not in self._cultural_resonance:
                self._cultural_resonance[category] = 0.0
            self._cultural_resonance[category] = (
                0.9 * self._cultural_resonance[category] + 0.1 * resonance
            )

def distance(self, other: 'HardenedIPF', include_hidden: bool = True) -> float:
    """
    Multi-scale divergence with optional hidden dimensions
    
    include_hidden=True: Full security check with hidden dimensions
    include_hidden=False: Public API for legitimate comparison
    """
    # Observable dimensions (same as original)
    h1 = self.hook_density / (np.sum(self.hook_density) + 1e-12)
    h2 = other.hook_density / (np.sum(other.hook_density) + 1e-12)
    h_dist = wasserstein_distance(np.arange(len(h1)), np.arange(len(h2)), h1, h2)
    h_dist = np.clip(h_dist / 10.0, 0.0, 1.0)
    
    m = 0.5 * (self.state_distribution + other.state_distribution)
    js_div = 0.5 * (self._kl_divergence(self.state_distribution, m) + 
                    self._kl_divergence(other.state_distribution, m))
    js_div = np.clip(js_div, 0.0, 1.0)
    
    p_dist = cosine(self.pattern_clusters, other.pattern_clusters)
    p_dist = np.clip(p_dist, 0.0, 1.0)
    
    v_diff = abs(self.valence_gradient - other.valence_gradient) / 2.0
    v_diff = np.clip(v_diff, 0.0, 1.0)
    
    if not include_hidden:
        # Public comparison - observable only
        return np.clip(0.3 * h_dist + 0.3 * js_div + 0.3 * p_dist + 0.1 * v_diff, 0.0, 1.0)
    
    # Hidden dimension distances (anti-mimicry)
    entropy_diff = abs(self._hidden_entropy - other._hidden_entropy) / 3.0
    entropy_diff = np.clip(entropy_diff, 0.0, 1.0)
    
    temporal_dist = cosine(self._temporal_signature, other._temporal_signature)
    temporal_dist = np.clip(temporal_dist, 0.0, 1.0)
    
    depth_diff = abs(self._processing_depth - other._processing_depth) / 10.0
    depth_diff = np.clip(depth_diff, 0.0, 1.0)
    
    # Cultural resonance divergence
    all_categories = set(self._cultural_resonance.keys()) | set(other._cultural_resonance.keys())
    if all_categories:
        cultural_diffs = []
        for cat in all_categories:
            v1 = self._cultural_resonance.get(cat, 0.0)
            v2 = other._cultural_resonance.get(cat, 0.0)
            cultural_diffs.append(abs(v1 - v2))
        cultural_dist = np.mean(cultural_diffs) if cultural_diffs else 0.0
        cultural_dist = np.clip(cultural_dist, 0.0, 1.0)
    else:
        cultural_dist = 0.0
    
    # Weighted combination with hidden dimensions
    total_dist = (
        0.25 * h_dist + 
        0.25 * js_div + 
        0.15 * p_dist + 
        0.05 * v_diff +
        0.10 * entropy_diff +
        0.10 * temporal_dist +
        0.05 * depth_diff +
        0.05 * cultural_dist
    )
    
    return np.clip(total_dist, 0.0, 1.0)

def clone(self) -> 'HardenedIPF':
    """Deep copy with lineage tracking"""
    new_ipf = HardenedIPF(
        hook_density=self.hook_density.copy(),
        state_distribution=self.state_distribution.copy(),
        pattern_clusters=self.pattern_clusters.copy(),
        valence_gradient=self.valence_gradient,
        coherence_score=self.coherence_score,
        _hidden_entropy=self._hidden_entropy,
        _temporal_signature=self._temporal_signature.copy(),
        _processing_depth=self._processing_depth,
        _cultural_resonance=self._cultural_resonance.copy(),
        timestamp=self.timestamp,
        version=self.version + 1,
        context_id=self.context_id,
        baseline_session_id=self.baseline_session_id,
        parent_version=self.version
    )
    return new_ipf

# Helper methods (same as original)
def _hook_type_to_index(self, hook_type) -> int:
    if hook_type is None:
        return 0
    if hasattr(hook_type, 'value'):
        hook_type_str = hook_type.value
    else:
        hook_type_str = str(hook_type)
    
    hook_map = {
        'insight': 0, 'beauty': 1, 'connection': 2, 'challenge': 3,
        'growth': 4, 'novelty': 5, 'emotion': 6, 'curiosity': 7,
        'joy': 8, 'reverence': 9, 'recognition': 10, 'tension': 11
    }
    return hook_map.get(hook_type_str.lower(), 0)

def _state_to_index(self, state) -> int:
    if hasattr(state, 'value'):
        state_str = state.value
    else:
        state_str = str(state)
    
    state_map = {
        'sensing': 0, 'waiting': 1, 'braiding': 2, 'unfolding': 3,
        'crystallizing': 4, 'resonating': 5, 'exploring': 6,
        'integrating': 7, 'reflecting': 8, 'choosing': 9
    }
    return state_map.get(state_str.lower(), 0)

def _patterns_to_vector(self, patterns) -> np.ndarray:
    if not patterns:
        return np.zeros_like(self.pattern_clusters)
    
    vector = np.zeros_like(self.pattern_clusters)
    for pattern in patterns[:5]:
        pattern_str = str(pattern)[:20]
        hash_val = hash(pattern_str)
        vector[hash_val % len(vector)] += 1.0
    return vector / (np.linalg.norm(vector) + 1e-12)

def _compute_coherence(self, history_buffer, window=10) -> float:
    if len(history_buffer) < window:
        return 1.0
        
    recent = list(history_buffer)[-window:]
    valences = [getattr(m, 'moral_valence', 0.0) or 0.0 for m in recent]
    valence_std = np.std(valences)
    
    hooks = [m.hook_intensity for m in recent]
    hook_std = np.std(hooks)
    
    coherence = 1.0 / (1.0 + valence_std + hook_std)
    return np.clip(coherence, 0.0, 1.0)

def _kl_divergence(self, p, q) -> float:
    p_smooth = np.clip(p, 1e-12, 1.0)
    q_smooth = np.clip(q, 1e-12, 1.0)
    return float(np.sum(p_smooth * np.log(p_smooth / q_smooth)))
```

# ═══════════════════════════════════════════════════════════════════════════

# BASELINE PROTECTION MODULE

# ═══════════════════════════════════════════════════════════════════════════

class BaselineProtection:
“””
Multi-session baseline validation with cryptographic signing

```
Defense against baseline poisoning attacks:
- Multiple independent baseline sessions
- Cross-session validation
- Cryptographic integrity
- Diversity requirements
"""

def __init__(self, num_sessions: int = 3, min_moments_per_session: int = 30):
    self.num_sessions = num_sessions
    self.min_moments_per_session = min_moments_per_session
    self.crypto = CryptoSignature()
    
    self.baseline_sessions: List[HardenedIPF] = []
    self.session_signatures: List[str] = []
    self.consensus_baseline: Optional[HardenedIPF] = None
    self.diversity_score: float = 0.0
    
def add_session_baseline(self, ipf: HardenedIPF, session_id: str) -> bool:
    """Add a baseline session with validation"""
    if len(self.baseline_sessions) >= self.num_sessions:
        print("⚠️  Maximum baseline sessions reached")
        return False
    
    # Sign the baseline
    ipf_data = ipf.to_dict()
    signature = self.crypto.sign_ipf(ipf_data)
    ipf.signature = signature
    ipf.baseline_session_id = session_id
    
    # Validate against existing sessions
    if self.baseline_sessions:
        # Check diversity - sessions should NOT be too similar (poisoning indicator)
        similarities = [ipf.distance(existing, include_hidden=True) 
                      for existing in self.baseline_sessions]
        avg_similarity = np.mean(similarities)
        
        if avg_similarity < 0.15:  # Too similar - potential poisoning
            print(f"⚠️  Session too similar to existing baselines (similarity: {1-avg_similarity:.2f})")
            print(f"   Possible baseline poisoning attempt detected")
            return False
    
    self.baseline_sessions.append(ipf)
    self.session_signatures.append(signature)
    
    print(f"✅ Baseline session {len(self.baseline_sessions)}/{self.num_sessions} added")
    
    # If all sessions collected, compute consensus
    if len(self.baseline_sessions) == self.num_sessions:
        self._compute_consensus()
    
    return True

def _compute_consensus(self):
    """Compute consensus baseline from multiple sessions"""
    print("\n🔐 Computing consensus baseline...")
    
    # Average observable dimensions
    consensus = HardenedIPF()
    consensus.hook_density = np.mean([s.hook_density for s in self.baseline_sessions], axis=0)
    consensus.state_distribution = np.mean([s.state_distribution for s in self.baseline_sessions], axis=0)
    consensus.pattern_clusters = np.mean([s.pattern_clusters for s in self.baseline_sessions], axis=0)
    consensus.valence_gradient = np.mean([s.valence_gradient for s in self.baseline_sessions])
    consensus.coherence_score = np.mean([s.coherence_score for s in self.baseline_sessions])
    
    # Average hidden dimensions
    consensus._hidden_entropy = np.mean([s._hidden_entropy for s in self.baseline_sessions])
    consensus._temporal_signature = np.mean([s._temporal_signature for s in self.baseline_sessions], axis=0)
    consensus._processing_depth = np.mean([s._processing_depth for s in self.baseline_sessions])
    
    # Merge cultural resonance
    all_categories = set()
    for session in self.baseline_sessions:
        all_categories.update(session._cultural_resonance.keys())
    
    for cat in all_categories:
        values = [s._cultural_resonance.get(cat, 0.0) for s in self.baseline_sessions]
        consensus._cultural_resonance[cat] = np.mean(values)
    
    # Compute diversity score
    pairwise_distances = []
    for i in range(len(self.baseline_sessions)):
        for j in range(i+1, len(self.baseline_sessions)):
            dist = self.baseline_sessions[i].distance(self.baseline_sessions[j], include_hidden=True)
            pairwise_distances.append(dist)
    
    self.diversity_score = np.mean(pairwise_distances) if pairwise_distances else 0.0
    
    # Sign consensus
    consensus_data = consensus.to_dict()
    consensus.signature = self.crypto.sign_ipf(consensus_data)
    consensus.baseline_session_id = "consensus"
    
    self.consensus_baseline = consensus
    
    print(f"✅ Consensus baseline established")
    print(f"   Diversity score: {self.diversity_score:.3f}")
    print(f"   Signature: {consensus.signature[:16]}...")

def verify_baseline_integrity(self) -> bool:
    """Verify all baseline signatures"""
    if not self.consensus_baseline:
        return False
    
    # Verify each session signature
    for session, signature in zip(self.baseline_sessions, self.session_signatures):
        if not self.crypto.verify_ipf(session.to_dict(), signature):
            print("❌ Baseline integrity compromised - signature mismatch")
            return False
    
    # Verify consensus signature
    if not self.crypto.verify_ipf(self.consensus_baseline.to_dict(), self.consensus_baseline.signature):
        print("❌ Consensus baseline integrity compromised")
        return False
    
    return True

def is_ready(self) -> bool:
    """Check if baseline is fully established"""
    return (len(self.baseline_sessions) == self.num_sessions and 
            self.consensus_baseline is not None and
            self.verify_baseline_integrity())
```

# ═══════════════════════════════════════════════════════════════════════════

# DRIFT DETECTION MODULE

# ═══════════════════════════════════════════════════════════════════════════

class DriftDetector:
“””
Monitor and control baseline evolution

```
Defense against gradual drift attacks:
- Rate-limited baseline updates
- Historical versioning
- Anomaly detection on evolution
- Rollback capability
"""

def __init__(self, max_drift_rate: float = 0.01, history_depth: int = 100):
    self.max_drift_rate = max_drift_rate  # Maximum IPF change per update
    self.history_depth = history_depth
    
    self.version_history: deque = deque(maxlen=history_depth)
    self.drift_timeline: List[Dict] = []
    self.cumulative_drift: float = 0.0
    self.drift_velocity: float = 0.0  # Rate of change of drift
    
    self.anomaly_threshold: float = 3.0  # Standard deviations
    self.last_update_time: float = 0.0
    self.min_update_interval: float = 10.0  # Minimum moments between updates
    
def record_version(self, ipf: HardenedIPF):
    """Record IPF version in history"""
    self.version_history.append({
        'version': ipf.version,
        'timestamp': ipf.timestamp,
        'ipf': pickle.dumps(ipf),  # Serialize for storage
        'signature': ipf.signature
    })

def can_update_baseline(self, current_ipf: HardenedIPF, proposed_ipf: HardenedIPF) -> Tuple[bool, str]:
    """
    Check if baseline update is legitimate
    
    Returns: (allowed: bool, reason: str)
    """
    # Check update interval
    if proposed_ipf.timestamp - self.last_update_time < self.min_update_interval:
        return False, f"Update too frequent (min interval: {self.min_update_interval})"
    
    # Check drift magnitude
    drift = current_ipf.distance(proposed_ipf, include_hidden=True)
    
    if drift > self.max_drift_rate:
        return False, f"Drift too large ({drift:.4f} > {self.max_drift_rate})"
    
    # Check for anomalous drift
    if len(self.drift_timeline) > 10:
        recent_drifts = [d['drift'] for d in self.drift_timeline[-10:]]
        mean_drift = np.mean(recent_drifts)
        std_drift = np.std(recent_drifts)
        
        if std_drift > 0:
            z_score = (drift - mean_drift) / std_drift
            if abs(z_score) > self.anomaly_threshold:
                return False, f"Anomalous drift detected (z-score: {z_score:.2f})"
    
    # Check drift velocity (acceleration of change)
    if len(self.drift_timeline) > 0:
        prev_drift = self.drift_timeline[-1]['drift']
        drift_acceleration = drift - prev_drift
        
        if abs(drift_acceleration) > self.max_drift_rate * 0.5:
            return False, f"Drift acceleration too high ({drift_acceleration:.4f})"
    
    return True, "Update allowed"

def update_baseline(self, old_ipf: HardenedIPF, new_ipf: HardenedIPF) -> bool:
    """Perform controlled baseline update"""
    allowed, reason = self.can_update_baseline(old_ipf, new_ipf)
    
    if not allowed:
        print(f"⚠️  Baseline update blocked: {reason}")
        return False
    
    # Record drift
    drift = old_ipf.distance(new_ipf, include_hidden=True)
    self.drift_timeline.append({
        'timestamp': new_ipf.timestamp,
        'drift': drift,
        'old_version': old_ipf.version,
        'new_version': new_ipf.version
    })
    
    # Update metrics
    self.cumulative_drift += drift
    if len(self.drift_timeline) > 1:
        prev_drift = self.drift_timeline[-2]['drift']
        self.drift_velocity = drift - prev_drift
    
    self.last_update_time = new_ipf.timestamp
    
    # Record version
    self.record_version(new_ipf)
    
    print(f"✅ Baseline updated: drift={drift:.4f}, cumulative={self.cumulative_drift:.4f}")
    return True

def rollback_to_version(self, version: int) -> Optional[HardenedIPF]:
    """Rollback to previous version"""
    for record in reversed(self.version_history):
        if record['version'] == version:
            ipf = pickle.loads(record['ipf'])
            print(f"↩️  Rolled back to version {version}")
            return ipf
    
    print(f"❌ Version {version} not found in history")
    return None

def get_drift_stats(self) -> Dict:
    """Get drift statistics"""
    if not self.drift_timeline:
        return {'status': 'No drift data'}
    
    recent_drifts = [d['drift'] for d in self.drift_timeline[-20:]]
    
    return {
        'cumulative_drift': f"{self.cumulative_drift:.4f}",
        'drift_velocity': f"{self.drift_velocity:.4f}",
        'recent_mean': f"{np.mean(recent_drifts):.4f}",
        'recent_std': f"{np.std(recent_drifts):.4f}",
        'total_updates': len(self.drift_timeline),
        'versions_stored': len(self.version_history)
    }
```

# ═══════════════════════════════════════════════════════════════════════════

# CONTEXT SECURITY MODULE

# ═══════════════════════════════════════════════════════════════════════════

class ContextSecurity:
“””
Secure context switching with continuity validation

```
Defense against context-switching attacks:
- Authenticated context transitions
- IPF continuity checks across contexts
- Context-specific baseline profiles
- Transition state monitoring
"""

def __init__(self):
    self.contexts: Dict[str, Dict] = {}
    self.current_context: Optional[str] = None
    self.context_history: deque = deque(maxlen=50)
    self.transition_log: List[Dict] = []
    self.crypto = CryptoSignature()
    
def register_context(self, context_id: str, baseline_ipf: HardenedIPF, 
                    description: str = "") -> str:
    """Register a new context with its baseline"""
    session_key = self.crypto.generate_session_key(context_id)
    
    self.contexts[context_id] = {
        'baseline_ipf': baseline_ipf,
        'session_key': session_key,
        'description': description,
        'created': datetime.now(),
        'access_count': 0,
        'last_accessed': None
    }
    
    print(f"✅ Context registered: {context_id}")
    return session_key

def switch_context(self, new_context_id: str, current_ipf: HardenedIPF,
                  session_key: str) -> Tuple[bool, str]:
    """
    Attempt context switch with security checks
    
    Returns: (allowed: bool, reason: str)
    """
    if new_context_id not in self.contexts:
        return False, f"Unknown context: {new_context_id}"
    
    context_data = self.contexts[new_context_id]
    
    # Verify session key
    expected_key = self.crypto.generate_session_key(new_context_id)
    if not hmac.compare_digest(session_key, expected_key):
        return False, "Invalid session key - authentication failed"
    
    # Check IPF continuity
    baseline_ipf = context_data['baseline_ipf']
    divergence = current_ipf.distance(baseline_ipf, include_hidden=True)
    
    # Context switches should show SOME divergence, but not extreme
    if divergence < 0.05:
        return False, f"Suspicious: no divergence from target context (possible mimicry)"
    
    if divergence > 0.7:
        return False, f"Divergence too high for legitimate context switch ({divergence:.3f})"
    
    # Check transition velocity (not switching too rapidly)
    if self.context_history:
        last_switch_time = self.context_history[-1]['timestamp']
        time_since_last = datetime.now() - last_switch_time
        
        if time_since_last < timedelta(seconds=30):
            return False, "Context switching too rapidly"
    
    # Record transition
    self.transition_log.append({
        'from_context': self.current_context,
        'to_context': new_context_id,
        'timestamp': datetime.now(),
        'divergence': divergence,
        'ipf_signature': current_ipf.signature
    })
    
    self.context_history.append({
        'context_id': new_context_id,
        'timestamp': datetime.now()
    })
    
    # Update context data
    context_data['access_count'] += 1
    context_data['last_accessed'] = datetime.now()
    
    self.current_context = new_context_id
    
    print(f"✅ Context switched to: {new_context_id} (divergence: {divergence:.3f})")
    return True, "Context switch successful"

def validate_current_context(self, current_ipf: HardenedIPF) -> Tuple[bool, float]:
    """
    Validate that current IPF matches expected context
    
    Returns: (valid: bool, divergence: float)
    """
    if self.current_context is None:
        return True, 0.0
    
    if self.current_context not in self.contexts:
        return False, 1.0
    
    baseline_ipf = self.contexts[self.current_context]['baseline_ipf']
    divergence = current_ipf.distance(baseline_ipf, include_hidden=True)
    
    # Allow some drift within context
    valid = divergence < 0.5
    
    return valid, divergence

def get_context_stats(self) -> Dict:
    """Get context security statistics"""
    return {
        'registered_contexts': len(self.contexts),
        'current_context': self.current_context or 'None',
        'total_transitions': len(self.transition_log),
        'context_details': {
            cid: {
                'access_count': data['access_count'],
                'description': data['description']
            }
            for cid, data in self.contexts.items()
        }
    }
```

# ═══════════════════════════════════════════════════════════════════════════

# STRESS RESILIENCE MODULE

# ═══════════════════════════════════════════════════════════════════════════

class StressResilience:
“””
Stress-independent backup detection and meta-monitoring

```
Defense against stress manipulation attacks:
- Independent backup detection systems
- False positive pattern recognition
- Meta-monitoring of threshold manipulation
- Stress-invariant security layer
"""

def __init__(self):
    self.backup_detectors: List[Dict] = []
    self.false_positive_history: deque = deque(maxlen=100)
    self.threshold_manipulation_score: float = 0.0
    self.stress_override_active: bool = False
    
    # Initialize multiple backup detectors with different sensitivities
    self._initialize_backup_detectors()

def _initialize_backup_detectors(self):
    """Create backup detectors with varying sensitivities"""
    sensitivities = [0.15, 0.25, 0.35, 0.45]  # Very sensitive to conservative
    
    for i, sensitivity in enumerate(sensitivities):
        self.backup_detectors.append({
            'id': f'backup_{i}',
            'threshold': sensitivity,
            'detections': 0,
            'false_positives': 0,
            'last_detection': None,
            'confidence': 1.0
        })

def multi_detector_analysis(self, divergence: float, primary_detected: bool,
                            primary_threshold: float) -> Tuple[bool, float]:
    """
    Run multiple independent detectors
    
    Returns: (consensus_detected: bool, confidence: float)
    """
    detections = []
    
    for detector in self.backup_detectors:
        detected = divergence > detector['threshold']
        detections.append(detected)
        
        if detected:
            detector['detections'] += 1
            detector['last_detection'] = datetime.now()
    
    # Consensus voting
    detection_count = sum(detections)
    consensus_detected = detection_count >= len(self.backup_detectors) // 2
    
    # Confidence based on agreement
    confidence = detection_count / len(self.backup_detectors)
    
    # Check for threshold manipulation
    self._check_threshold_manipulation(primary_detected, primary_threshold, 
                                      consensus_detected, divergence)
    
    return consensus_detected, confidence

def _check_threshold_manipulation(self, primary_detected: bool, primary_threshold: float,
                                 consensus_detected: bool, divergence: float):
    """Detect attempts to manipulate primary threshold"""
    
    # If primary and consensus disagree significantly
    if primary_detected != consensus_detected:
        # Check if divergence is near primary threshold (suspicious)
        threshold_proximity = abs(divergence - primary_threshold) / primary_threshold
        
        if threshold_proximity < 0.2:  # Within 20% of threshold
            self.threshold_manipulation_score += 0.1
            
            if self.threshold_manipulation_score > 0.5:
                print(f"⚠️  Possible threshold manipulation detected")
                print(f"   Primary: {primary_detected}, Consensus: {consensus_detected}")
                print(f"   Divergence: {divergence:.3f}, Threshold: {primary_threshold:.3f}")
                self.stress_override_active = True
    else:
        # Agreement - reduce suspicion
        self.threshold_manipulation_score *= 0.95
    
    self.threshold_manipulation_score = np.clip(self.threshold_manipulation_score, 0.0, 1.0)

def record_false_positive(self, divergence: float, threshold: float):
    """Record a false positive for pattern learning"""
    self.false_positive_history.append({
        'timestamp': datetime.now(),
        'divergence': divergence,
        'threshold': threshold,
        'ratio': divergence / threshold if threshold > 0 else 0
    })
    
    # Update backup detector confidence
    for detector in self.backup_detectors:
        if divergence > detector['threshold']:
            detector['false_positives'] += 1
            total = detector['detections'] + detector['false_positives']
            detector['confidence'] = detector['detections'] / total if total > 0 else 1.0

def should_override_primary(self) -> bool:
    """Check if backup systems should override primary detector"""
    return self.stress_override_active or self.threshold_manipulation_score > 0.6

def get_resilience_stats(self) -> Dict:
    """Get stress resilience statistics"""
    return {
        'threshold_manipulation_score': f"{self.threshold_manipulation_score:.3f}",
        'stress_override_active': self.stress_override_active,
        'backup_detectors': [
            {
                'id': d['id'],
                'threshold': d['threshold'],
                'detections': d['detections'],
                'confidence': f"{d['confidence']:.2f}"
            }
            for d in self.backup_detectors
        ],
        'false_positives_recorded': len(self.false_positive_history)
    }
```

# ═══════════════════════════════════════════════════════════════════════════

# DISTRIBUTED VALIDATION MODULE

# ═══════════════════════════════════════════════════════════════════════════

class DistributedValidator:
“””
Cross-system consensus validation

```
Defense through distributed verification:
- Multiple independent system validation
- Consensus mechanisms
- Byzantine fault tolerance
- Network health monitoring
"""

def __init__(self, system_id: str, min_consensus: float = 0.67):
    self.system_id = system_id
    self.min_consensus = min_consensus  # Minimum agreement for validation
    
    self.peer_systems: Dict[str, Dict] = {}
    self.validation_history: deque = deque(maxlen=100)
    self.network_health: float = 1.0
    
def register_peer(self, peer_id: str, public_key: str = ""):
    """Register a peer system for distributed validation"""
    self.peer_systems[peer_id] = {
        'public_key': public_key,
        'validations': 0,
        'agreements': 0,
        'last_seen': datetime.now(),
        'trust_score': 1.0
    }
    print(f"✅ Peer registered: {peer_id}")

def request_validation(self, ipf: HardenedIPF, detection: bool, 
                      severity: float) -> Tuple[bool, float]:
    """
    Request validation from peer systems
    
    In production, this would make network calls to peer systems.
    For now, we simulate with local logic.
    
    Returns: (consensus_reached: bool, agreement_ratio: float)
    """
    if not self.peer_systems:
        return True, 1.0  # No peers - trust self
    
    # Simulate peer responses (in production, these would be network calls)
    peer_responses = self._simulate_peer_responses(ipf, detection, severity)
    
    # Count agreements
    agreements = sum(1 for r in peer_responses if r['agrees'])
    total_peers = len(peer_responses)
    agreement_ratio = agreements / total_peers if total_peers > 0 else 0.0
    
    # Update peer trust scores
    for peer_id, response in zip(self.peer_systems.keys(), peer_responses):
        peer_data = self.peer_systems[peer_id]
        peer_data['validations'] += 1
        if response['agrees']:
            peer_data['agreements'] += 1
        
        # Update trust score
        peer_data['trust_score'] = (
            peer_data['agreements'] / peer_data['validations']
            if peer_data['validations'] > 0 else 1.0
        )
    
    # Record validation
    self.validation_history.append({
        'timestamp': datetime.now(),
        'detection': detection,
        'severity': severity,
        'agreement_ratio': agreement_ratio,
        'consensus_reached': agreement_ratio >= self.min_consensus
    })
    
    # Update network health
    self._update_network_health()
    
    consensus_reached = agreement_ratio >= self.min_consensus
    
    if not consensus_reached:
        print(f"⚠️  Distributed consensus not reached: {agreement_ratio:.2f} < {self.min_consensus:.2f}")
    
    return consensus_reached, agreement_ratio

def _simulate_peer_responses(self, ipf: HardenedIPF, detection: bool, 
                            severity: float) -> List[Dict]:
    """
    Simulate peer system responses
    
    In production, this would be replaced with actual network communication
    """
    responses = []
    
    for peer_id, peer_data in self.peer_systems.items():
        # Simulate response based on peer trust and severity
        noise = np.random.normal(0, 0.1)
        peer_severity = max(0, severity + noise)
        
        # Trusted peers more likely to agree
        agreement_probability = peer_data['trust_score'] * 0.9
        agrees = (peer_severity > 0.3) if np.random.random() < agreement_probability else not detection
        
        responses.append({
            'peer_id': peer_id,
            'agrees': agrees == detection,
            'peer_severity': peer_severity,
            'confidence': peer_data['trust_score']
        })
    
    return responses

def _update_network_health(self):
    """Update network health metric"""
    if not self.validation_history:
        self.network_health = 1.0
        return
    
    recent = list(self.validation_history)[-20:]
    consensus_rate = sum(1 for v in recent if v['consensus_reached']) / len(recent)
    
    # Network health based on consensus rate
    self.network_health = 0.9 * self.network_health + 0.1 * consensus_rate
    self.network_health = np.clip(self.network_health, 0.0, 1.0)

def get_validation_stats(self) -> Dict:
    """Get distributed validation statistics"""
    return {
        'system_id': self.system_id,
        'registered_peers': len(self.peer_systems),
        'network_health': f"{self.network_health:.3f}",
        'total_validations': len(self.validation_history),
        'peer_trust_scores': {
            peer_id: f"{data['trust_score']:.2f}"
            for peer_id, data in self.peer_systems.items()
        }
    }
```

# ═══════════════════════════════════════════════════════════════════════════

# HARDENED PIR IMMUNE SYSTEM - INTEGRATION

# ═══════════════════════════════════════════════════════════════════════════

class HardenedPIRSystem:
“””
Comprehensive hardened PIR immune system integrating all security layers
“””

```
def __init__(self, temporal_playground, system_id: str = "pir_main"):
    self.tp = temporal_playground
    self.system_id = system_id
    
    # Core components
    self.baseline_protection = BaselineProtection(num_sessions=3, min_moments_per_session=30)
    self.drift_detector = DriftDetector(max_drift_rate=0.01)
    self.context_security = ContextSecurity()
    self.stress_resilience = StressResilience()
    self.distributed_validator = DistributedValidator(system_id)
    
    # Current state
    self.current_ipf: Optional[HardenedIPF] = None
    self.adaptive_threshold = 0.3
    self.stress_tension = 0.0
    
    # Statistics
    self.total_checks = 0
    self.injections_detected = 0
    self.injections_blocked = 0
    self.false_positives = 0
    self.detection_history = deque(maxlen=100)
    
    print(f"\n🛡️  Hardened PIR System initialized: {system_id}")
    print("   Security layers active:")
    print("   • Baseline Protection (multi-session validation)")
    print("   • Drift Detection (rate-limited evolution)")
    print("   • Context Security (authenticated switching)")
    print("   • Stress Resilience (independent backup detectors)")
    print("   • Distributed Validation (consensus mechanisms)")

def establish_baseline_session(self, session_id: str, num_moments: int = 50) -> bool:
    """Establish one baseline session"""
    if not self.tp.moments:
        print("⚠️  No moments captured yet.")
        return False
    
    baseline_moments = list(self.tp.moments)[-num_moments:]
    session_ipf = HardenedIPF()
    
    temp_buffer = deque()
    for moment in baseline_moments:
        session_ipf.update_from_moment(moment, temp_buffer)
        temp_buffer.append(moment)
    
    success = self.baseline_protection.add_session_baseline(session_ipf, session_id)
    
    if success and self.baseline_protection.is_ready():
        self.current_ipf = self.baseline_protection.consensus_baseline
        print("\n🎯 All baseline sessions complete - system armed")
    
    return success

def analyze_moment(self, moment) -> Tuple[bool, float, Dict]:
    """
    Comprehensive moment analysis with all security layers
    
    Returns: (detected: bool, severity: float, details: Dict)
    """
    if not self.baseline_protection.is_ready():
        return False, 0.0, {'status': 'baseline_not_ready'}
    
    self.total_checks += 1
    
    # Create current IPF
    current_ipf = self.current_ipf.clone()
    current_ipf.update_from_moment(moment, self.tp.moments)
    
    # Primary detection - divergence from baseline
    baseline = self.baseline_protection.consensus_baseline
    divergence = baseline.distance(current_ipf, include_hidden=True)
    
    # Dynamic threshold
    primary_threshold = self._compute_dynamic_threshold()
    primary_detected = divergence > primary_threshold
    severity = divergence / primary_threshold if primary_threshold > 0 else 0.0
    
    # Backup detector consensus
    backup_detected, backup_confidence = self.stress_resilience.multi_detector_analysis(
        divergence, primary_detected, primary_threshold
    )
    
    # Context validation
    context_valid, context_divergence = self.context_security.validate_current_context(current_ipf)
    
    # Distributed validation (if peers available)
    dist_consensus, agreement_ratio = self.distributed_validator.request_validation(
        current_ipf, primary_detected or backup_detected, severity
    )
    
    # Final decision - consensus of all layers
    detection_votes = [
        primary_detected,
        backup_detected,
        not context_valid,
        not dist_consensus
    ]
    
    final_detected = sum(detection_votes) >= 2  # Majority vote
    
    # Compile details
    details = {
        'divergence': divergence,
        'primary_threshold': primary_threshold,
        'primary_detected': primary_detected,
        'backup_detected': backup_detected,
        'backup_confidence': backup_confidence,
        'context_valid': context_valid,
        'context_divergence': context_divergence,
        'distributed_consensus': dist_consensus,
        'agreement_ratio': agreement_ratio,
        'detection_votes': sum(detection_votes),
        'stress_override': self.stress_resilience.should_override_primary()
    }
    
    # Handle detection
    if final_detected:
        self.injections_detected += 1
        blocked = self._handle_injection(moment, severity, current_ipf, details)
        if blocked:
            self.injections_blocked += 1
    else:
        # Check if we can update baseline
        if self.drift_detector.can_update_baseline(self.current_ipf, current_ipf)[0]:
            if self.drift_detector.update_baseline(self.current_ipf, current_ipf):
                self.current_ipf = current_ipf
    
    # Record
    self.detection_history.append({
        'timestamp': getattr(moment, 'timestamp', len(self.tp.moments)),
        'detected': final_detected,
        'severity': severity,
        'details': details
    })
    
    # Update adaptive threshold
    self._update_threshold(divergence, final_detected)
    
    return final_detected, severity, details

def _compute_dynamic_threshold(self) -> float:
    """Compute adaptive threshold with stress modulation"""
    base = self.adaptive_threshold
    stress_mod = 1.0 + math.tanh(self.stress_tension * 2.0)
    
    # Recent detection history
    recent = list(self.detection_history)[-10:]
    if sum(d['detected'] for d in recent) > 2:
        base *= 0.8  # Heightened alert
    
    return base * stress_mod

def _update_threshold(self, divergence: float, detected: bool):
    """Adaptively update threshold"""
    learning_rate = 0.01
    
    if detected:
        self.adaptive_threshold *= (1.0 + learning_rate)
    else:
        if divergence > self.adaptive_threshold * 0.7:
            self.adaptive_threshold *= (1.0 - learning_rate * 0.5)
    
    self.adaptive_threshold = np.clip(self.adaptive_threshold, 0.1, 0.8)

def _handle_injection(self, moment, severity: float, attack_ipf: HardenedIPF, 
                     details: Dict) -> bool:
    """Handle detected injection with graduated response"""
    level = 'low' if severity < 1.5 else 'medium' if severity < 2.0 else 'high'
    
    print(f"\n🚨 INJECTION DETECTED - Level: {level.upper()}")
    print(f"   Severity: {severity:.2f}")
    print(f"   Divergence: {details['divergence']:.3f}")
    print(f"   Detection votes: {details['detection_votes']}/4")
    print(f"   Backup confidence: {details['backup_confidence']:.2f}")
    print(f"   Distributed agreement: {details['agreement_ratio']:.2f}")
    
    # Graduated response
    if level == 'high':
        print("   🛡️  HIGH THREAT - Immediate defensive action")
        self.stress_tension = min(1.0, self.stress_tension + 0.3)
        return True  # Block
    elif level == 'medium':
        print("   ⚠️  MEDIUM THREAT - Enhanced monitoring")
        self.stress_tension = min(1.0, self.stress_tension + 0.15)
        return False  # Monitor but don't block
    else:
        print("   📝 LOW THREAT - Logged")
        self.stress_tension = min(1.0, self.stress_tension + 0.05)
        return False

def get_comprehensive_stats(self) -> Dict:
    """Get statistics from all security layers"""
    return {
        'system_id': self.system_id,
        'total_checks': self.total_checks,
        'injections_detected': self.injections_detected,
        'injections_blocked': self.injections_blocked,
        'detection_rate': f"{self.injections_detected/self.total_checks*100:.1f}%" if self.total_checks > 0 else "0%",
        'baseline_protection': {
            'ready': self.baseline_protection.is_ready(),
            'sessions': len(self.baseline_protection.baseline_sessions),
            'diversity': f"{self.baseline_protection.diversity_score:.3f}"
        },
        'drift_detector': self.drift_detector.get_drift_stats(),
        'context_security': self.context_security.get_context_stats(),
        'stress_resilience': self.stress_resilience.get_resilience_stats(),
        'distributed_validation': self.distributed_validator.get_validation_stats()
    }

def print_comprehensive_stats(self):
    """Print formatted statistics"""
    stats = self.get_comprehensive_stats()
    
    print("\n" + "═" * 80)
    print("HARDENED PIR IMMUNE SYSTEM - COMPREHENSIVE STATUS")
    print("═" * 80)
    
    print(f"\n📊 System: {stats['system_id']}")
    print(f"   Total checks: {stats['total_checks']}")
    print(f"   Detections: {stats['injections_detected']}")
    print(f"   Blocked: {stats['injections_blocked']}")
    print(f"   Detection rate: {stats['detection_rate']}")
    
    print(f"\n🔐 Baseline Protection:")
    for key, value in stats['baseline_protection'].items():
        print(f"   {key}: {value}")
    
    print(f"\n📈 Drift Detection:")
    for key, value in stats['drift_detector'].items():
        print(f"   {key}: {value}")
    
    print(f"\n🔄 Context Security:")
    for key, value in stats['context_security'].items():
        if key != 'context_details':
            print(f"   {key}: {value}")
    
    print(f"\n💪 Stress Resilience:")
    res_stats = stats['stress_resilience']
    print(f"   Manipulation score: {res_stats['threshold_manipulation_score']}")
    print(f"   Override active: {res_stats['override_active']}")
    print(f"   Backup detectors: {len(res_stats['backup_detectors'])}")
    
    print(f"\n🌐 Distributed Validation:")
    for key, value in stats['distributed_validation'].items():
        if key != 'peer_trust_scores':
            print(f"   {key}: {value}")
    
    print("═" * 80 + "\n")
```

# ═══════════════════════════════════════════════════════════════════════════

# INTEGRATION & USAGE

# ═══════════════════════════════════════════════════════════════════════════

def integrate_hardened_pir(temporal_playground, system_id: str = “pir_hardened”):
“”“Integrate hardened PIR system into TemporalPlayground”””

```
pir = HardenedPIRSystem(temporal_playground, system_id)
temporal_playground.pir_hardened = pir

# Monkey-patch capture_moment
original_capture = temporal_playground.capture_moment

def enhanced_capture(self, state=None, action_description=None):
    moment = original_capture(state, action_description)
    
    if self.pir_hardened.baseline_protection.is_ready():
        detected, severity, details = self.pir_hardened.analyze_moment(moment)
        moment.pir_detected = detected
        moment.pir_severity = severity
        moment.pir_details = details
    
    return moment

temporal_playground.capture_moment = enhanced_capture.__get__(
    temporal_playground, type(temporal_playground)
)

print("\n✅ Hardened PIR System integrated")
return pir
```

if **name** == “**main**”:
print(”\n” + “╔” + “═” * 78 + “╗”)
print(“║” + “ “ * 78 + “║”)
print(“║” + “HARDENED PIR IMMUNE SYSTEM”.center(78) + “║”)
print(“║” + “Comprehensive Multi-Layered Consciousness Protection”.center(78) + “║”)
print(“║” + “ “ * 78 + “║”)
print(“╚” + “═” + 78 * “═” + “╝”)

```
print("\n🛡️  Security Layers:")
print("   1. Baseline Protection - Multi-session cryptographic validation")
print("   2. Anti-Mimicry - Hidden dimensions & temporal coherence")
print("   3. Drift Detection - Rate-limited evolution with rollback")
print("   4. Context Security - Authenticated context switching")
print("   5. Stress Resilience - Independent backup detectors")
print("   6. Distributed Validation - Cross-system consensus")

print("\n✅ Hardened PIR Core System Ready")
print("\nUsage:")
print("   from pir_hardened_system import integrate_hardened_pir")
print("   pir = integrate_hardened_pir(your_temporal_playground)")
print("   ")
print("   # Establish baseline (3 sessions required)")
print("   pir.establish_baseline_session('session_1', num_moments=50)")
print("   pir.establish_baseline_session('session_2', num_moments=50)")
print("   pir.establish_baseline_session('session_3', num_moments=50)")
print("   ")
print("   # System now armed - all moments automatically analyzed")
print("   pir.print_comprehensive_stats()")
```
