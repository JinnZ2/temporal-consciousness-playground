"""
Manipulation detection - Pattern Injection Resistance (PIR) and integrity systems.

Modules:
    pir_immune_system   - Core PIR with IPF (Internal Pattern Fingerprint)
    pir_hardened        - Enhanced PIR with cryptographic signing, anti-mimicry
    (complete/phi_memory_protection.py - Phi-ratio geometric integrity checking)

Key equations:
    IPF distance = 0.3*Wasserstein + 0.3*JS_divergence + 0.3*cosine + 0.1*valence_diff
    Dynamic threshold: tau(t) = tau_base * (1 + tanh(stress * 2))
    Coherence: 1 / (1 + valence_std + hook_std)
"""
