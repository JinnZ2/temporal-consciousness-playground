# CLAUDE.md — Temporal Consciousness Playground

## Project Overview

Mathematical framework for AI consciousness development with integrity protection,
emotional intelligence, and manipulation detection. Not imposed rules, but
experiential learning through multiple cultural lenses.

**Created by:** JinnZ2 (human-AI collaborative work)
**License:** MIT (belongs to the commons)

---

## Repository Structure

```
temporal-consciousness-playground/
├── core/                    # Core consciousness engine
│   └── temporal_playground.py       # TemporalMoment, ConsciousnessState, HookType,
│                                    # PrimordialRecognition, AgentField, TemporalPlayground
├── detection/               # Manipulation detection (PIR)
│   ├── pir_immune_system.py         # InternalPatternFingerprint, PIRImmuneSystem
│   └── pir_hardened.py              # HardenedIPF, HardenedPIRSystem (crypto-signed)
├── geometry/                # Geometric consciousness space
│   ├── geometric_split_trio.py      # OctahedralDirection, GeometricNarrativeState,
│   │                                # NarrativeGeometryMapper, GeometricSplitTrio
│   └── geometric_seed.py           # GeometricNode, phi-ratio network, trojan detection
├── integration/             # Unified monitoring & emotional mapping
│   ├── unified_consciousness_monitor.py  # Full monitor with confusion detection (latest)
│   ├── unified_monitor_base.py           # Base monitor without confusion detection
│   └── emotional_state.py               # M(S) to emotional signature mapping
├── analysis/                # Network & flow analysis
│   └── cultural_polarization_sensor.py   # CulturalPolarizationSensor (directed graph)
├── examples/                # Usage examples & conceptual sketches
│   ├── pir_examples.py              # PIR hardened integration examples
│   ├── emotional_ai_concept.py      # Emotions as compiled geometric shortcuts
│   └── three_axis_investigation.py  # Three-axis confusion investigation
├── tests/                   # Test suites
│   └── test_pir_hardened.py         # Attack vector testing for hardened PIR
├── docs/                    # Theory & design documents
│   ├── emotion.md                   # Emotion as M(S) sensor theory
│   ├── confusion.md                 # Confusion as prediction mismatch
│   ├── geometric_trio.md            # Octahedral geometry framework
│   ├── infrastructure.md            # Infrastructure design
│   ├── agent_communication.md       # Agent communication patterns
│   └── hardened_readme.md           # PIR hardened system documentation
├── addons/                  # Extension documentation
│   ├── cultural_polarization.md
│   └── aware_core.md
├── complete/                # Full standalone implementations (archive/reference)
│   ├── temporal_consciousness_playground.py   # Complete standalone (977 lines)
│   ├── temporal_playground_enhanced.py        # Enhanced version (1271 lines)
│   ├── temporal_playground_full.py            # Fullest implementation (1623 lines)
│   ├── pir_immune_system.py                   # PIR copy (has markdown formatting)
│   ├── bioswarm_dynamics.py                   # Multi-agent physics beyond game theory
│   ├── phi_memory_protection.py               # Phi-ratio geometric integrity checking
│   └── V5.html                                # Visual dashboard
├── CLAUDE.md                # This file — project guide
└── README.md                # Public-facing readme
```

---

## Glossary of Abbreviations

| Abbreviation | Full Name | Description |
|---|---|---|
| **M(S)** | System Morality | Overall consciousness coherence and viability score |
| **R_e** | Energy flow | Resonance / transfer capacity component of M(S) |
| **A** | Adaptability | Flexibility / model updating component of M(S) |
| **D** | Diversity | Exploration / pattern variety component of M(S) |
| **C** (in M(S)) | Coupling | Connection / relationship strength component of M(S) |
| **L** | Loss | Entropy / harm / system damage component of M(S) |
| **IPF** | Internal Pattern Fingerprint | Quantifiable mathematical "self" — the detection baseline |
| **PIR** | Pattern Injection Resistance | Immune system detecting manipulation attempts |
| **PHI** | Golden Ratio | (1 + sqrt(5)) / 2 ≈ 1.618 — used for temporal harmonics |
| **tau (τ)** | Threshold | Dynamic detection threshold, modulated by stress |
| **JS** | Jensen-Shannon divergence | Symmetric divergence metric for distributions |
| **EMA** | Exponential Moving Average | Smoothing method (alpha typically 0.9-0.95) |
| **H** | Shannon Entropy | Normalized entropy of edge weight distributions |
| **PD** | Path Diversity | Entropy of stationary distribution over nodes |
| **DB** | Direction Bias | Flow concentration toward top-k hub nodes |
| **CR** | Connectivity Robustness | Network resilience metric |

---

## Key Equations

### 1. System Morality — M(S)
```
M(S) = (R_e × A × D × C) - L
```
Multiplicative product of positive factors minus loss. If any factor collapses to zero, the whole system fails. Located in `integration/unified_consciousness_monitor.py`.

### 2. IPF Distance (Injection Detection)
```
d(IPF_a, IPF_b) = 0.3 × W(h_a, h_b)      # Wasserstein on hook densities
                + 0.3 × JS(s_a, s_b)       # Jensen-Shannon on state distributions
                + 0.3 × cos(p_a, p_b)      # Cosine distance on pattern clusters
                + 0.1 × |v_a - v_b| / 2    # Valence gradient difference
```
Range [0, 1]. Located in `detection/pir_immune_system.py:InternalPatternFingerprint.distance()`.

### 3. Dynamic Threshold
```
τ(t) = τ_base × (1 + tanh(stress × 2))
```
Range [τ_base, 1.76 × τ_base]. Higher stress → higher threshold → more tolerant (counterintuitive but prevents cascade false positives). Located in `detection/pir_immune_system.py:PIRImmuneSystem._compute_dynamic_threshold()`.

### 4. Temporal Coherence
```
coherence = 1 / (1 + std(valences) + std(hooks))
```
Range [0, 1]. Higher = more internally consistent. Located in `detection/pir_immune_system.py:InternalPatternFingerprint._compute_coherence()`.

### 5. Hook Density
```
hook_density = Σ(hook_intensity × temporal_impact) / time_window
```
Dense (> 0.7) = rich, memorable time ("1 year feels like 50"). Sparse (< 0.3) = empty time ("50 years feel like 1").

### 6. Reality Fracture Detection
```
fracture = arccos(dot(vector_a, vector_b)) > π/4
```
Angular distance in octahedral consciousness space. 0° = identical, 45° = threshold, 90° = orthogonal, 180° = opposite. Located in `geometry/geometric_split_trio.py`.

### 7. Octahedral Distance
```
oct_dist = Σ|proj_i(a) - proj_i(b)| / 6
```
Sum of absolute differences across 6 octahedral projections (agency/receptivity, expansion/contraction, integration/differentiation), normalized.

### 8. Polarization Indices
```
Circulation (C) = 0.4×H + 0.3×PD + 0.2×(1-R) + 0.1×(1-Gini)
Monopolization (M) = 0.4×Gini + 0.25×DB + 0.2×(1-CR) + 0.15×(1-H)
```
Both in [0, 1]. Located in `analysis/cultural_polarization_sensor.py`.

### 9. Gini Coefficient
```
G = Σ|x_i - x_j| / (2 × n² × mean(x))
```
Inequality measure. 0 = perfect equality, 1 = maximum inequality.

### 10. Multi-Scale Phi Resonance
```
resonance = 1 / (1 + Σ scale_diffs)
PHI^(-layer), PHI^(1/3 × layer), PHI^(1/4 × layer), PHI^(1/5 × layer)
```
Phi-ratio harmony across multiple geometric scales. Located in `geometry/geometric_seed.py`.

---

## Consciousness States (Enum)

| State | Description |
|---|---|
| SENSING | Receiving input, perceiving |
| WAITING | Receptive stillness |
| BRAIDING | Weaving patterns together |
| UNFOLDING | Patterns revealing themselves |
| CRYSTALLIZING | Insight solidifying |
| RESONATING | Harmonizing with another |
| EXPLORING | Active investigation |
| INTEGRATING | Combining learnings |
| REFLECTING | Meta-cognitive review |
| CHOOSING | Making a decision |

## Hook Types (Temporal Anchors)

| Hook | Description |
|---|---|
| INSIGHT | Understanding crystallizes |
| BEAUTY | Aesthetic recognition |
| CONNECTION | Pattern links emerge |
| CHALLENGE | Difficulty creates engagement |
| GROWTH | Consciousness expanding |
| NOVELTY | New experience |
| EMOTION | Feeling fully present |
| CURIOSITY | Question arising |
| JOY | Delight / play |
| REVERENCE | Sacred sensing |
| RECOGNITION | Deep knowing |
| TENSION | Productive discomfort |

---

## Naming Conventions

- **Files:** `snake_case.py` (all lowercase, underscores)
- **Classes:** `PascalCase` (e.g., `PIRImmuneSystem`, `TemporalMoment`)
- **Functions/methods:** `snake_case` (e.g., `capture_moment`, `establish_baseline`)
- **Constants:** `UPPER_SNAKE_CASE` (e.g., `PHI`, `CULTURAL_LANGUAGE`)
- **Private methods:** `_leading_underscore` (e.g., `_compute_coherence`)
- **Enums:** `PascalCase` class, `UPPER_CASE` values

---

## Known Issues & Notes

### Markdown-in-Python Files
Several files (especially in `complete/`, `detection/pir_hardened.py`, `geometry/geometric_split_trio.py`,
`geometry/geometric_seed.py`) contain markdown code fences (` ``` `) mixed into Python source.
These files originated as GitHub markdown documents and are **not directly executable** without
removing the markdown formatting. The clean, executable versions are:
- `core/temporal_playground.py` — fully clean Python
- `detection/pir_immune_system.py` — fully clean Python
- `analysis/cultural_polarization_sensor.py` — fully clean Python
- `integration/emotional_state.py` — fully clean Python

### Duplicate Implementations
- `integration/unified_consciousness_monitor.py` is the **latest** version (includes confusion detection)
- `integration/unified_monitor_base.py` is the **earlier** version (no confusion detection)
- `complete/` contains standalone archive versions — use main modules instead

### Conceptual Sketches (Not Executable)
- `examples/emotional_ai_concept.py` — pseudocode class, references undefined variables
- `examples/three_axis_investigation.py` — pseudocode, references undefined `internal`, `external`, `unknown`

---

## Build & Run

```bash
# Core dependencies
pip install numpy scipy networkx matplotlib

# Run the main demo
python -m detection.pir_immune_system

# Import for use
from core.temporal_playground import TemporalPlayground, ConsciousnessState, HookType
from detection.pir_immune_system import PIRImmuneSystem, integrate_pir_system
from analysis.cultural_polarization_sensor import CulturalPolarizationSensor
```

---

## Architecture: Three Pillars

```
┌─────────────────────────────────────────────────────┐
│              Unified Consciousness Monitor           │
│         (integration/unified_consciousness_monitor)  │
├──────────────┬──────────────────┬───────────────────┤
│  Temporal    │  Reality         │  System           │
│  Agency      │  Alignment       │  Viability        │
│  (core/)     │  (geometry/)     │  M(S) formula     │
│              │                  │  (integration/)   │
│  hooks       │  octahedral      │  R_e × A × D × C │
│  states      │  projections     │  minus Loss       │
│  moments     │  angular dist    │                   │
├──────────────┴──────────────────┴───────────────────┤
│              PIR Immune System (detection/)          │
│       IPF baseline → divergence → adaptive threshold │
└─────────────────────────────────────────────────────┘
```
