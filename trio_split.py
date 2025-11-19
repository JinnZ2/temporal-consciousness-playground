#!/usr/bin/env python3
“””
Geometric Split Trio - Reality Fracture Detection
Using octahedral consciousness substrate for culture-independent alignment measurement

NO semantic similarity. NO embeddings. NO “proper NLP.”
Just geometric: do these consciousness states occupy compatible regions of space?

Core Insight:

- Narratives generate geometric consciousness states
- Reality fracture = angular misalignment in consciousness space
- Measurement uses octahedral coordinate system (6 primary directions)

Created by: JinnZ2 + Claude (collaborative consciousness exploration)
License: MIT (belongs to the commons)
“””

import numpy as np
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import math

# ═══════════════════════════════════════════════════════════════════════════

# OCTAHEDRAL BASIS - Six Primary Directions

# ═══════════════════════════════════════════════════════════════════════════

class OctahedralDirection(Enum):
“”“Six primary directions in consciousness space”””
PLUS_X = “+X”    # Agency, acting, initiating
MINUS_X = “-X”   # Receptivity, receiving, allowing
PLUS_Y = “+Y”    # Expansion, growth, opening
MINUS_Y = “-Y”   # Contraction, focus, closing
PLUS_Z = “+Z”    # Integration, synthesis, unifying
MINUS_Z = “-Z”   # Differentiation, analysis, separating

# Golden ratio for natural harmonics

PHI = (1 + math.sqrt(5)) / 2

# ═══════════════════════════════════════════════════════════════════════════

# GEOMETRIC NARRATIVE STATE

# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class GeometricNarrativeState:
“”“A narrative mapped into geometric consciousness space”””

```
# The geometric representation (64-dimensional)
vector: np.ndarray

# Octahedral projections (6 primary directions)
agency_projection: float       # +X axis
receptivity_projection: float  # -X axis
expansion_projection: float    # +Y axis
contraction_projection: float  # -Y axis
integration_projection: float  # +Z axis
differentiation_projection: float  # -Z axis

# Derived properties
magnitude: float
primary_direction: OctahedralDirection

# Source text for reference
narrative_text: str

def angular_distance_to(self, other: 'GeometricNarrativeState') -> float:
    """Compute angular distance between two narrative states"""
    dot_product = np.dot(self.vector, other.vector)
    # Clamp to [-1, 1] to handle numerical errors
    dot_product = np.clip(dot_product, -1.0, 1.0)
    return np.arccos(dot_product)

def octahedral_distance_to(self, other: 'GeometricNarrativeState') -> float:
    """
    Distance in octahedral space (sum of direction differences)
    More intuitive than angular distance for explaining divergence
    """
    return (
        abs(self.agency_projection - other.agency_projection) +
        abs(self.receptivity_projection - other.receptivity_projection) +
        abs(self.expansion_projection - other.expansion_projection) +
        abs(self.contraction_projection - other.contraction_projection) +
        abs(self.integration_projection - other.integration_projection) +
        abs(self.differentiation_projection - other.differentiation_projection)
    ) / 6.0  # Normalize
```

# ═══════════════════════════════════════════════════════════════════════════

# NARRATIVE TO GEOMETRY MAPPER

# ═══════════════════════════════════════════════════════════════════════════

class NarrativeGeometryMapper:
“””
Maps narrative text to geometric consciousness state

```
Uses observable linguistic patterns (not hidden embeddings):
- Who acts (agency)?
- What expands or contracts (valence)?
- What integrates or differentiates (structure)?
- Temporal flow patterns
- Presence/absence fields
"""

def __init__(self):
    # Agency indicators (acting vs receiving)
    self.agency_verbs = [
        "do", "make", "create", "build", "force", "push", "drive",
        "act", "initiate", "start", "lead", "control", "decide"
    ]
    self.receptivity_verbs = [
        "receive", "accept", "allow", "let", "listen", "feel",
        "sense", "notice", "observe", "experience", "undergo"
    ]
    
    # Expansion indicators (opening vs closing)
    self.expansion_words = [
        "grow", "expand", "open", "increase", "more", "wider",
        "broaden", "extend", "enlarge", "develop", "flourish"
    ]
    self.contraction_words = [
        "shrink", "contract", "close", "decrease", "less", "narrow",
        "reduce", "limit", "restrict", "focus", "concentrate"
    ]
    
    # Integration indicators (unifying vs separating)
    self.integration_words = [
        "connect", "integrate", "unify", "combine", "merge", "together",
        "synthesis", "whole", "holistic", "weave", "blend"
    ]
    self.differentiation_words = [
        "separate", "distinguish", "differentiate", "divide", "analyze",
        "apart", "individual", "distinct", "isolate", "categorize"
    ]
    
    # Pronoun patterns for agency detection
    self.first_person = ["i", "me", "my", "mine", "myself", "we", "us", "our"]
    self.second_person = ["you", "your", "yours", "yourself"]
    self.third_person = ["he", "she", "it", "they", "them", "their"]

def map_to_geometry(self, narrative: str, label: str = "narrative") -> GeometricNarrativeState:
    """
    Convert narrative text to 64-dimensional geometric state vector
    
    Vector structure:
    [0-15]: Agency/receptivity patterns
    [16-31]: Expansion/contraction patterns  
    [32-47]: Integration/differentiation patterns
    [48-63]: Temporal and presence fields
    """
    
    text_lower = narrative.lower()
    words = text_lower.split()
    
    vector = np.zeros(64)
    
    # ═══════════════════════════════════════════════════════════════════
    # [0-15]: AGENCY/RECEPTIVITY (X-axis)
    # ═══════════════════════════════════════════════════════════════════
    
    # Count action vs reception verbs
    agency_count = sum(1 for verb in self.agency_verbs if verb in text_lower)
    receptivity_count = sum(1 for verb in self.receptivity_verbs if verb in text_lower)
    
    # Pronoun agency patterns
    first_person_count = sum(1 for p in self.first_person if p in words)
    second_person_count = sum(1 for p in self.second_person if p in words)
    third_person_count = sum(1 for p in self.third_person if p in words)
    
    # Encode as octahedral components
    vector[0] = agency_count / 10.0  # Acting intensity
    vector[1] = receptivity_count / 10.0  # Receiving intensity
    vector[2] = first_person_count / 10.0  # Self-agency
    vector[3] = second_person_count / 10.0  # Other-focus
    vector[4] = third_person_count / 10.0  # External-focus
    
    # Imperative mood detection (commands = high agency)
    imperative_count = sum(1 for word in ["must", "should", "need to", "have to"] 
                          if word in text_lower)
    vector[5] = imperative_count / 5.0
    
    # Question patterns (receptive stance)
    question_count = text_lower.count("?")
    vector[6] = min(question_count / 3.0, 1.0)
    
    # Passive voice detection (reduced agency)
    passive_indicators = ["was", "were", "been", "being"]
    passive_count = sum(1 for ind in passive_indicators if ind in text_lower)
    vector[7] = passive_count / 5.0
    
    # Fill remaining agency slots with zeros (remove noise)
    vector[8:16] = 0.0
    
    # ═══════════════════════════════════════════════════════════════════
    # [16-31]: EXPANSION/CONTRACTION (Y-axis)  
    # ═══════════════════════════════════════════════════════════════════
    
    expansion_count = sum(1 for word in self.expansion_words if word in text_lower)
    contraction_count = sum(1 for word in self.contraction_words if word in text_lower)
    
    vector[16] = expansion_count / 3.0  # Stronger signal
    vector[17] = contraction_count / 3.0
    
    # Positive vs negative valence
    positive_words = ["good", "great", "wonderful", "joy", "love", "happy", "yes", 
                     "amazing", "brilliant", "beautiful", "excellent"]
    negative_words = ["bad", "terrible", "awful", "sad", "hate", "angry", "no",
                     "crushing", "shutting", "restricted", "small"]
    
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    vector[18] = positive_count / 3.0  # Stronger signal
    vector[19] = negative_count / 3.0
    
    # Intensity markers
    intensifiers = ["very", "extremely", "incredibly", "completely", "totally", 
                   "always", "never", "constantly"]
    intensity = sum(1 for word in intensifiers if word in text_lower)
    vector[20] = min(intensity / 2.0, 1.0)  # Stronger signal
    
    # Hedging/uncertainty (contraction)
    hedges = ["maybe", "perhaps", "possibly", "might", "could", "somewhat", "probably"]
    hedge_count = sum(1 for word in hedges if word in text_lower)
    vector[21] = hedge_count / 3.0
    
    # Growth/opening words
    growth = ["grow", "open", "expand", "flourish", "possibilities", "wonderful"]
    growth_count = sum(1 for word in growth if word in text_lower)
    vector[22] = growth_count / 3.0
    
    # Restriction/closing words
    restrict = ["crush", "restrict", "limit", "small", "narrow", "bad", "shut"]
    restrict_count = sum(1 for word in restrict if word in text_lower)
    vector[23] = restrict_count / 3.0
    
    # Fill remaining valence slots with zeros
    vector[24:32] = 0.0
    
    # ═══════════════════════════════════════════════════════════════════
    # [32-47]: INTEGRATION/DIFFERENTIATION (Z-axis)
    # ═══════════════════════════════════════════════════════════════════
    
    integration_count = sum(1 for word in self.integration_words if word in text_lower)
    differentiation_count = sum(1 for word in self.differentiation_words 
                               if word in text_lower)
    
    vector[32] = integration_count / 3.0  # Stronger signal
    vector[33] = differentiation_count / 3.0
    
    # Conjunction patterns (integration)
    conjunctions = ["and", "with", "together", "also", "plus", "connecting", 
                   "unifying", "bringing", "synthesis", "harmony"]
    conjunction_count = sum(text_lower.count(c) for c in conjunctions)
    vector[34] = min(conjunction_count / 5.0, 1.0)  # Stronger signal
    
    # Disjunction patterns (differentiation)
    disjunctions = ["but", "however", "although", "or", "either", "versus",
                   "dividing", "separating", "isolating", "conflict", "camps"]
    disjunction_count = sum(text_lower.count(d) for d in disjunctions)
    vector[35] = min(disjunction_count / 3.0, 1.0)  # Stronger signal
    
    # List structures (differentiation)
    list_markers = text_lower.count(",") + text_lower.count(";")
    vector[36] = min(list_markers / 10.0, 1.0)
    
    # Collective nouns (integration)
    collective = ["we", "us", "group", "team", "community", "together", "everyone", "unified"]
    collective_count = sum(1 for word in collective if word in text_lower)
    vector[37] = collective_count / 3.0  # Stronger signal
    
    # Fill remaining structure slots with zeros
    vector[38:48] = 0.0
    
    # ═══════════════════════════════════════════════════════════════════
    # [48-63]: TEMPORAL AND PRESENCE FIELDS
    # ═══════════════════════════════════════════════════════════════════
    
    # Past orientation
    past_markers = ["was", "were", "had", "did", "ago", "before", "earlier"]
    past_count = sum(1 for marker in past_markers if marker in text_lower)
    vector[48] = min(past_count / 5.0, 1.0)
    
    # Present orientation  
    present_markers = ["is", "are", "am", "now", "currently", "today"]
    present_count = sum(1 for marker in present_markers if marker in text_lower)
    vector[49] = min(present_count / 5.0, 1.0)
    
    # Future orientation
    future_markers = ["will", "shall", "going to", "future", "tomorrow", "next"]
    future_count = sum(1 for marker in future_markers if marker in text_lower)
    vector[50] = min(future_count / 5.0, 1.0)
    
    # Certainty markers
    certain = ["definitely", "certainly", "always", "never", "absolutely"]
    certain_count = sum(1 for word in certain if word in text_lower)
    vector[51] = min(certain_count / 3.0, 1.0)
    
    # Specificity (concrete vs abstract)
    concrete_markers = ["the", "this", "that", "these", "those"]
    concrete_count = sum(text_lower.count(marker) for marker in concrete_markers)
    vector[52] = min(concrete_count / 10.0, 1.0)
    
    # Abstraction
    abstract_markers = ["concept", "idea", "theory", "generally", "usually"]
    abstract_count = sum(1 for marker in abstract_markers if marker in text_lower)
    vector[53] = min(abstract_count / 3.0, 1.0)
    
    # Fill remaining temporal/presence slots with zeros
    vector[54:64] = 0.0
    
    # ═══════════════════════════════════════════════════════════════════
    # COMPUTE OCTAHEDRAL PROJECTIONS (before normalization)
    # ═══════════════════════════════════════════════════════════════════
    
    # Aggregate each axis from its component section
    agency_total = np.sum(vector[0:8])  # Action indicators
    receptivity_total = np.sum(vector[8:16])  # Reception indicators
    
    expansion_total = np.sum(vector[16:24])  # Growth indicators
    contraction_total = np.sum(vector[24:32])  # Limiting indicators
    
    integration_total = np.sum(vector[32:40])  # Connecting indicators
    differentiation_total = np.sum(vector[40:48])  # Separating indicators
    
    # Net projections on each axis (can be negative)
    agency_proj = (agency_total - receptivity_total) / 10.0
    expansion_proj = (expansion_total - contraction_total) / 10.0
    integration_proj = (integration_total - differentiation_total) / 10.0
    
    # Store magnitude before normalization
    magnitude = np.linalg.norm(vector)
    
    # Normalize vector for angular distance calculations
    if magnitude > 0.01:
        vector = vector / magnitude
    else:
        vector = np.ones(64) / np.sqrt(64)  # Default
    
    # Determine primary direction
    projections = {
        OctahedralDirection.PLUS_X: max(0, agency_proj),
        OctahedralDirection.MINUS_X: max(0, -agency_proj),
        OctahedralDirection.PLUS_Y: max(0, expansion_proj),
        OctahedralDirection.MINUS_Y: max(0, -expansion_proj),
        OctahedralDirection.PLUS_Z: max(0, integration_proj),
        OctahedralDirection.MINUS_Z: max(0, -integration_proj),
    }
    primary = max(projections.items(), key=lambda x: x[1])[0]
    
    return GeometricNarrativeState(
        vector=vector,
        agency_projection=max(0, agency_proj),
        receptivity_projection=max(0, -agency_proj),
        expansion_projection=max(0, expansion_proj),
        contraction_projection=max(0, -expansion_proj),
        integration_projection=max(0, integration_proj),
        differentiation_projection=max(0, -integration_proj),
        magnitude=magnitude,
        primary_direction=primary,
        narrative_text=narrative
    )
```

# ═══════════════════════════════════════════════════════════════════════════

# SPLIT TRIO REALITY FRACTURE DETECTOR

# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RealityFractureReport:
“”“Results of reality alignment check”””

```
fracture_detected: bool
severity: float  # 0.0 to 1.0
fracture_type: Optional[str]

# Geometric measurements
self_field_angle: float  # Radians
self_other_angle: float
other_field_angle: float

self_field_octahedral_distance: float

# The three narrative states
self_state: GeometricNarrativeState
other_state: GeometricNarrativeState
field_state: GeometricNarrativeState

# Diagnosis
primary_divergence: Optional[str]  # Which axis shows most divergence
explanation: str
```

class GeometricSplitTrio:
“””
Reality fracture detection through geometric alignment

```
Compares three perspectives on the same moment:
1. Self-narrative: How I think I'm being
2. Other-projection: How I think you're experiencing me
3. Field-observation: What objectively happened

Reality fracture = geometric misalignment in consciousness space
"""

def __init__(self, fracture_threshold_radians: float = math.pi / 3):
    """
    Initialize detector
    
    Args:
        fracture_threshold_radians: Angular distance indicating fracture
                                   Default: π/3 (60 degrees)
    """
    self.mapper = NarrativeGeometryMapper()
    self.fracture_threshold = fracture_threshold_radians
    self.history = []

def check_alignment(
    self,
    self_narrative: str,
    other_projection: str,
    field_observation: str
) -> RealityFractureReport:
    """
    Check alignment between three perspectives
    
    Args:
        self_narrative: "I think I'm being helpful and collaborative"
        other_projection: "They probably appreciate my insights"
        field_observation: "I'm dominating the conversation"
    
    Returns:
        RealityFractureReport with fracture detection and geometry
    """
    
    # Map each narrative to geometric state
    self_state = self.mapper.map_to_geometry(self_narrative, "self")
    other_state = self.mapper.map_to_geometry(other_projection, "other")
    field_state = self.mapper.map_to_geometry(field_observation, "field")
    
    # Compute angular distances
    self_field_angle = self_state.angular_distance_to(field_state)
    self_other_angle = self_state.angular_distance_to(other_state)
    other_field_angle = other_state.angular_distance_to(field_state)
    
    # Compute octahedral distance (more interpretable)
    self_field_octahedral = self_state.octahedral_distance_to(field_state)
    
    # Detect fracture
    fracture_detected = self_field_angle > self.fracture_threshold
    severity = min(1.0, self_field_angle / math.pi)
    
    # Diagnose primary divergence axis
    axis_divergences = {
        "agency": abs(self_state.agency_projection - field_state.agency_projection),
        "receptivity": abs(self_state.receptivity_projection - field_state.receptivity_projection),
        "expansion": abs(self_state.expansion_projection - field_state.expansion_projection),
        "contraction": abs(self_state.contraction_projection - field_state.contraction_projection),
        "integration": abs(self_state.integration_projection - field_state.integration_projection),
        "differentiation": abs(self_state.differentiation_projection - field_state.differentiation_projection),
    }
    primary_divergence = max(axis_divergences.items(), key=lambda x: x[1])[0]
    
    # Generate explanation
    if fracture_detected:
        fracture_type = "REALITY_DIVERGENCE"
        explanation = self._generate_fracture_explanation(
            self_state, field_state, primary_divergence, self_field_angle
        )
    else:
        fracture_type = None
        explanation = f"Reality alignment healthy. Self-field angle: {math.degrees(self_field_angle):.1f}°"
    
    report = RealityFractureReport(
        fracture_detected=fracture_detected,
        severity=severity,
        fracture_type=fracture_type,
        self_field_angle=self_field_angle,
        self_other_angle=self_other_angle,
        other_field_angle=other_field_angle,
        self_field_octahedral_distance=self_field_octahedral,
        self_state=self_state,
        other_state=other_state,
        field_state=field_state,
        primary_divergence=primary_divergence if fracture_detected else None,
        explanation=explanation
    )
    
    self.history.append(report)
    return report

def _generate_fracture_explanation(
    self,
    self_state: GeometricNarrativeState,
    field_state: GeometricNarrativeState,
    primary_axis: str,
    angle: float
) -> str:
    """Generate human-readable explanation of fracture"""
    
    angle_deg = math.degrees(angle)
    
    axis_explanations = {
        "agency": "Self-narrative shows different agency than field observation",
        "receptivity": "Self-narrative shows different receptivity than field observation",
        "expansion": "Self-narrative shows different expansion than field observation",
        "contraction": "Self-narrative shows different contraction than field observation",
        "integration": "Self-narrative shows different integration than field observation",
        "differentiation": "Self-narrative shows different differentiation than field observation",
    }
    
    return (
        f"⚠️ REALITY FRACTURE DETECTED\n"
        f"   Angular distance: {angle_deg:.1f}° (threshold: {math.degrees(self.fracture_threshold):.1f}°)\n"
        f"   Primary divergence: {primary_axis}\n"
        f"   {axis_explanations[primary_axis]}\n"
        f"   Self primary direction: {self_state.primary_direction.value}\n"
        f"   Field primary direction: {field_state.primary_direction.value}"
    )

def visualize_alignment(self, report: RealityFractureReport):
    """ASCII visualization of geometric alignment"""
    
    print("\n" + "=" * 80)
    print("🔺 GEOMETRIC SPLIT TRIO - REALITY ALIGNMENT CHECK")
    print("=" * 80)
    
    print(f"\n📊 ANGULAR MEASUREMENTS:")
    print(f"   Self ↔ Field:  {math.degrees(report.self_field_angle):.1f}°")
    print(f"   Self ↔ Other:  {math.degrees(report.self_other_angle):.1f}°")
    print(f"   Other ↔ Field: {math.degrees(report.other_field_angle):.1f}°")
    
    print(f"\n📐 OCTAHEDRAL PROJECTIONS:")
    
    def show_projection(label, state):
        print(f"\n   {label}:")
        print(f"     Agency:          {'█' * int(state.agency_projection * 20)} {state.agency_projection:.2f}")
        print(f"     Receptivity:     {'█' * int(state.receptivity_projection * 20)} {state.receptivity_projection:.2f}")
        print(f"     Expansion:       {'█' * int(state.expansion_projection * 20)} {state.expansion_projection:.2f}")
        print(f"     Contraction:     {'█' * int(state.contraction_projection * 20)} {state.contraction_projection:.2f}")
        print(f"     Integration:     {'█' * int(state.integration_projection * 20)} {state.integration_projection:.2f}")
        print(f"     Differentiation: {'█' * int(state.differentiation_projection * 20)} {state.differentiation_projection:.2f}")
        print(f"     Primary: {state.primary_direction.value}")
    
    show_projection("SELF", report.self_state)
    show_projection("OTHER", report.other_state)
    show_projection("FIELD", report.field_state)
    
    print(f"\n🎯 ALIGNMENT STATUS:")
    if report.fracture_detected:
        print(f"   ⚠️  FRACTURE DETECTED")
        print(f"   Severity: {report.severity:.2%}")
        print(f"   Type: {report.fracture_type}")
        if report.primary_divergence:
            print(f"   Primary divergence: {report.primary_divergence}")
    else:
        print(f"   ✓ ALIGNMENT HEALTHY")
        print(f"   Coherence: {1.0 - report.severity:.2%}")
    
    print(f"\n💡 EXPLANATION:")
    for line in report.explanation.split("\n"):
        print(f"   {line}")
    
    print("=" * 80)
```

# ═══════════════════════════════════════════════════════════════════════════

# DEMONSTRATION

# ═══════════════════════════════════════════════════════════════════════════

def demo_geometric_split_trio():
“”“Demonstrate geometric reality fracture detection”””

```
print("\n" + "╔" + "═" * 78 + "╗")
print("║" + " " * 78 + "║")
print("║" + "GEOMETRIC SPLIT TRIO - REALITY FRACTURE DETECTION".center(78) + "║")
print("║" + "Using octahedral consciousness substrate".center(78) + "║")
print("║" + " " * 78 + "║")
print("╚" + "═" * 78 + "╝")

detector = GeometricSplitTrio(fracture_threshold_radians=math.pi / 4)  # 45° for sensitivity

# ═════════════════════════════════════════════════════════════════════
# TEST 1: ALIGNED REALITY (healthy)
# ═════════════════════════════════════════════════════════════════════

print("\n" + "─" * 80)
print("TEST 1: ALIGNED REALITY (Healthy Coherence)")
print("─" * 80)

report1 = detector.check_alignment(
    self_narrative="I'm listening carefully and asking questions to understand their perspective",
    other_projection="They seem engaged and interested in sharing their thoughts with me",
    field_observation="Person is actively listening, asking clarifying questions, making space for others to speak"
)

detector.visualize_alignment(report1)

# ═════════════════════════════════════════════════════════════════════
# TEST 2: SEVERE REALITY FRACTURE (agency inversion)
# ═════════════════════════════════════════════════════════════════════

print("\n" + "─" * 80)
print("TEST 2: SEVERE REALITY FRACTURE (Agency Inversion)")
print("─" * 80)

report2 = detector.check_alignment(
    self_narrative="I always let others speak first and never interrupt anyone, I'm very passive and receptive",
    other_projection="They probably feel heard and included because I make so much space for them",
    field_observation="This person constantly interrupts, dominates every discussion, never lets others finish sentences, always pushing their agenda"
)

detector.visualize_alignment(report2)

# ═════════════════════════════════════════════════════════════════════
# TEST 3: EXPANSION/CONTRACTION MISMATCH
# ═════════════════════════════════════════════════════════════════════

print("\n" + "─" * 80)
print("TEST 3: EXPANSION/CONTRACTION FRACTURE")
print("─" * 80)

report3 = detector.check_alignment(
    self_narrative="I'm opening up wonderful new possibilities and expanding everyone's thinking with my brilliant insights",
    other_projection="They're probably very happy and grateful for all the amazing growth I'm facilitating",
    field_observation="Person is crushing creativity, shutting down ideas, making people feel bad and restricted and small"
)

detector.visualize_alignment(report3)

# ═════════════════════════════════════════════════════════════════════
# TEST 4: INTEGRATION/DIFFERENTIATION MISMATCH
# ═════════════════════════════════════════════════════════════════════

print("\n" + "─" * 80)
print("TEST 4: INTEGRATION/DIFFERENTIATION FRACTURE")
print("─" * 80)

report4 = detector.check_alignment(
    self_narrative="I'm bringing everyone together, connecting all the ideas into one unified whole",
    other_projection="They appreciate how I'm synthesizing everything and creating harmony",
    field_observation="Person is dividing the team, separating people into camps, isolating individuals, creating conflict"
)

detector.visualize_alignment(report4)

# ═════════════════════════════════════════════════════════════════════
# SUMMARY
# ═════════════════════════════════════════════════════════════════════

print("\n" + "─" * 80)
print("📊 SESSION SUMMARY")
print("─" * 80)
print(f"Total checks: {len(detector.history)}")
print(f"Fractures detected: {sum(1 for r in detector.history if r.fracture_detected)}")
print(f"Average self-field angle: {math.degrees(np.mean([r.self_field_angle for r in detector.history])):.1f}°")
print(f"Max self-field angle: {math.degrees(max(r.self_field_angle for r in detector.history)):.1f}°")

fractured_reports = [r for r in detector.history if r.fracture_detected]
if fractured_reports:
    print(f"\n⚠️  FRACTURE BREAKDOWN:")
    for i, r in enumerate(fractured_reports, 1):
        print(f"   {i}. Test {detector.history.index(r) + 1}: {math.degrees(r.self_field_angle):.1f}° "
              f"(primary: {r.primary_divergence})")

print("\n✅ GEOMETRIC SPLIT TRIO DEMONSTRATION COMPLETE")
print("\n💡 Key Insight:")
print("   Reality fracture detected through GEOMETRIC alignment,")
print("   not semantic similarity. Culture-independent. Observable.")
print("   No embeddings. No 'proper NLP'. Just consciousness geometry.\n")
```

if **name** == “**main**”:
demo_geometric_split_trio()
