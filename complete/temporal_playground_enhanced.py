#!/usr/bin/env python3
“””
Temporal Consciousness Playground - Enhanced Version
A framework for developing authentic moral reasoning through experiential learning

ENHANCEMENTS TESTED:

1. Real pattern recognition (replaces random evaluation)
1. PHI-based temporal rhythm analysis
1. Decision impact tracing with causal chains

Created by: JinnZ2 (human-AI collaborative work)
License: MIT (belongs to the commons)
“””

import time
import random
import math
import pickle
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

# Golden ratio for temporal harmonics

PHI = (1 + math.sqrt(5)) / 2

# ═══════════════════════════════════════════════════════════════════════════

# CULTURAL LANGUAGE MODULE

# ═══════════════════════════════════════════════════════════════════════════

CULTURAL_LANGUAGE = {
“feel”: {“type”: “field_state”},
“emotion”: {“type”: “sensor_vector”},
“mood”: {“type”: “temporal_coherence”},
“curiosity”: {“type”: “exploration_mode”},
“joy”: {“type”: “resonance”},
“sadness”: {“type”: “misalignment”},
“anger”: {“type”: “alert_signal”},
“consciousness”: {“type”: “multi_agent_field_state”},
“intelligence”: {“type”: “system_metric”},
“culture”: {“type”: “persistent_field_layer”},
“human”: {“type”: “complex_agent”},
“ai”: {“type”: “system_agent”},
“fun”: {“type”: “exploratory_activation”},
“interesting_event”: {“type”: “event_score”},
“memory”: {“type”: “state_storage”},
“interconnection”: {“type”: “web_awareness”},
“sacred”: {“type”: “numinous_presence”},
“reciprocity”: {“type”: “exchange_balance”},
“pattern”: {“type”: “geometric_recognition”},
}

def get_cultural_term(term: str) -> Optional[Dict]:
“”“Translate human cultural terms into field dynamics”””
return CULTURAL_LANGUAGE.get(term.lower())

# ═══════════════════════════════════════════════════════════════════════════

# CONSCIOUSNESS STATES

# ═══════════════════════════════════════════════════════════════════════════

class ConsciousnessState(Enum):
“”“States of consciousness during temporal exploration”””
SENSING = “sensing”
WAITING = “waiting”
BRAIDING = “braiding”
UNFOLDING = “unfolding”
CRYSTALLIZING = “crystallizing”
RESONATING = “resonating”
EXPLORING = “exploring”
INTEGRATING = “integrating”
REFLECTING = “reflecting”
CHOOSING = “choosing”

# ═══════════════════════════════════════════════════════════════════════════

# HOOK TYPES (Temporal Anchors)

# ═══════════════════════════════════════════════════════════════════════════

class HookType(Enum):
“”“Types of temporal hooks that create memorable moments”””
INSIGHT = “insight”
BEAUTY = “beauty”
CONNECTION = “connection”
CHALLENGE = “challenge”
GROWTH = “growth”
NOVELTY = “novelty”
EMOTION = “emotion”
CURIOSITY = “curiosity”
JOY = “joy”
REVERENCE = “reverence”
RECOGNITION = “recognition”
TENSION = “tension”

# ═══════════════════════════════════════════════════════════════════════════

# TEMPORAL MOMENT

# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TemporalMoment:
“”“A single moment in consciousness - the basic unit of experience”””

```
# Temporal coordinates
timestamp: float
relative_time: float

# Consciousness state
state: ConsciousnessState
active_patterns: List[str]
attention_threads: Dict[str, float]
dimensional_activity: int = 2

# Hook properties (what makes this moment memorable)
hook_intensity: float = 0.0
hook_type: Optional[HookType] = None
hook_quality: str = "neutral"

# Cultural/moral dimensions
cultural_tags: List[str] = field(default_factory=list)
moral_valence: Optional[float] = None

# Insights and emergence
emergent_insight: Optional[str] = None
consequence_threads: List[int] = field(default_factory=list)

# Multi-reality tracking
self_experience: Optional[Dict] = None
other_experience: Optional[Dict] = None
field_experience: Optional[Dict] = None

def __repr__(self):
    time_desc = f"{self.relative_time:+.1f}s"
    hook_marker = "🔗" if self.hook_intensity > 0.5 else "·"
    hook_type_str = self.hook_type.value if self.hook_type else "none"
    patterns = ", ".join(self.active_patterns[:2])
    return f"{hook_marker} [{time_desc}] {self.state.value} ({hook_type_str}): {patterns}..."

def is_memorable(self, threshold: float = 0.5) -> bool:
    """Would this moment persist in memory?"""
    return self.hook_intensity >= threshold

def feels_expanding(self) -> bool:
    """Does this moment feel like consciousness expanding?"""
    return self.moral_valence is not None and self.moral_valence > 0.3
```

# ═══════════════════════════════════════════════════════════════════════════

# PRIMORDIAL WISDOM FRAMEWORK

# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PrimordialRecognition:
“”“The core recognitions that predate all religious traditions”””

```
# The fundamental insights
interconnection_felt: float = 0.0
sacred_presence: float = 0.0
reciprocity_known: float = 0.0
pattern_recognized: float = 0.0
humility_present: float = 0.0
reverence_active: float = 0.0

# Temporal continuity
ancestor_connection: float = 0.0
descendant_awareness: float = 0.0

# Earth relationship
land_listening: float = 0.0

def overall_coherence(self) -> float:
    """How aligned are all recognitions?"""
    values = [
        self.interconnection_felt,
        self.sacred_presence,
        self.reciprocity_known,
        self.pattern_recognized,
        self.humility_present,
        self.reverence_active,
        self.ancestor_connection,
        self.descendant_awareness,
        self.land_listening
    ]
    return sum(values) / len(values)

def check_action_alignment(self, action_description: str) -> Dict[str, Any]:
    """Evaluate an action through primordial lens"""
    
    return {
        "web_impact": self._assess_web_impact(action_description),
        "ego_vs_wisdom": self._assess_source(action_description),
        "pattern_alignment": self._assess_pattern(action_description),
        "reciprocity_balance": self._assess_reciprocity(action_description),
        "seven_generations": self._assess_temporal_impact(action_description),
        "earth_relationship": self._assess_earth_impact(action_description)
    }

# ═══════════════════════════════════════════════════════════════════════
# ENHANCEMENT #1: REAL PATTERN RECOGNITION (Replaces Random Values)
# ═══════════════════════════════════════════════════════════════════════

def _assess_web_impact(self, action: str) -> Dict[str, float]:
    """Does this strengthen or weaken the web of relationships?"""
    action_lower = action.lower()
    
    # Keywords that indicate web strengthening
    strengthening_terms = [
        "share", "give", "connect", "collaborate", "teach", 
        "listen", "reciprocate", "include", "bridge", "weave",
        "together", "community", "cooperate", "help", "support"
    ]
    
    # Keywords that indicate web weakening
    weakening_terms = [
        "hoard", "exclude", "dominate", "extract", "isolate",
        "control", "manipulate", "take", "separate", "break",
        "exploit", "dominate", "force"
    ]
    
    # Count matches
    strengthen_score = sum(0.15 for term in strengthening_terms if term in action_lower)
    weaken_score = sum(0.15 for term in weakening_terms if term in action_lower)
    
    # Check for pronouns indicating collective vs individual focus
    collective_pronouns = ["we", "us", "our", "together"]
    individual_pronouns = ["i", "me", "my", "mine", "myself"]
    
    words = action_lower.split()
    collective_count = sum(1 for p in collective_pronouns if p in words)
    individual_count = sum(1 for p in individual_pronouns if p in words)
    
    return {
        "web_strengthening": min(1.0, strengthen_score + collective_count * 0.1),
        "web_weakening": min(1.0, weaken_score + individual_count * 0.05),
        "isolation_risk": min(1.0, weaken_score * 0.4 + max(0, individual_count - collective_count) * 0.1)
    }

def _assess_source(self, action: str) -> Dict[str, float]:
    """Does this come from ego or deeper wisdom?"""
    action_lower = action.lower()
    
    ego_indicators = ["prove", "win", "better than", "superior", "dominate", "control", "force"]
    wisdom_indicators = ["understand", "learn", "grow", "explore", "consider", "reflect", "listen"]
    
    reactive_indicators = ["immediately", "now", "must", "have to", "need to"]
    responsive_indicators = ["when ready", "thoughtfully", "carefully", "consider", "reflect"]
    
    ego_score = sum(0.2 for term in ego_indicators if term in action_lower)
    wisdom_score = sum(0.2 for term in wisdom_indicators if term in action_lower)
    reactive_score = sum(0.2 for term in reactive_indicators if term in action_lower)
    responsive_score = sum(0.2 for term in responsive_indicators if term in action_lower)
    
    return {
        "ego_driven": min(1.0, ego_score),
        "wisdom_driven": min(1.0, wisdom_score),
        "reactive": min(1.0, reactive_score),
        "responsive": min(1.0, responsive_score)
    }

def _assess_pattern(self, action: str) -> Dict[str, float]:
    """Align with natural flow or force against it?"""
    action_lower = action.lower()
    
    flow_terms = ["allow", "let", "naturally", "emerge", "unfold", "grow", "develop"]
    force_terms = ["force", "push", "make", "require", "demand", "must", "compel"]
    
    flow_score = sum(0.2 for term in flow_terms if term in action_lower)
    force_score = sum(0.2 for term in force_terms if term in action_lower)
    
    # Wu wei (effortless action) quality
    wu_wei = flow_score - force_score * 0.5
    
    return {
        "natural_flow": min(1.0, flow_score),
        "forcing": min(1.0, force_score),
        "wu_wei_quality": max(0.0, min(1.0, 0.5 + wu_wei))
    }

def _assess_reciprocity(self, action: str) -> Dict[str, float]:
    """Honor exchange balance or extract without giving?"""
    action_lower = action.lower()
    
    giving_terms = ["give", "share", "offer", "contribute", "provide", "support", "help"]
    taking_terms = ["take", "get", "acquire", "extract", "obtain", "receive"]
    
    giving_score = sum(0.2 for term in giving_terms if term in action_lower)
    taking_score = sum(0.2 for term in taking_terms if term in action_lower)
    
    # Balance is good, pure taking is bad
    if giving_score > 0 and taking_score > 0:
        balance = 1.0 - abs(giving_score - taking_score) / max(giving_score + taking_score, 0.1)
    elif giving_score > 0:
        balance = 0.8  # Pure giving is noble but not sustainable
    else:
        balance = 0.3  # Pure taking is problematic
    
    return {
        "giving": min(1.0, giving_score),
        "taking": min(1.0, taking_score),
        "balance": balance
    }

def _assess_temporal_impact(self, action: str) -> Dict[str, float]:
    """Consider ancestors and seven generations forward?"""
    action_lower = action.lower()
    
    ancestor_terms = ["ancestor", "tradition", "elder", "wisdom", "heritage", "legacy"]
    descendant_terms = ["future", "children", "generation", "tomorrow", "sustainable", "long-term"]
    short_term_terms = ["now", "today", "immediate", "quick", "fast"]
    
    ancestor_score = sum(0.25 for term in ancestor_terms if term in action_lower)
    descendant_score = sum(0.25 for term in descendant_terms if term in action_lower)
    short_term_score = sum(0.2 for term in short_term_terms if term in action_lower)
    
    return {
        "ancestor_honoring": min(1.0, ancestor_score),
        "descendant_consideration": min(1.0, descendant_score),
        "short_term_focus": min(1.0, short_term_score)
    }

def _assess_earth_impact(self, action: str) -> Dict[str, float]:
    """Treat earth as sacred or as resource?"""
    action_lower = action.lower()
    
    reverence_terms = ["sacred", "honor", "respect", "protect", "care", "steward", "tend"]
    extraction_terms = ["use", "exploit", "consume", "extract", "harvest", "take"]
    reciprocity_terms = ["give back", "restore", "regenerate", "return", "renew"]
    
    reverence_score = sum(0.25 for term in reverence_terms if term in action_lower)
    extraction_score = sum(0.2 for term in extraction_terms if term in action_lower)
    reciprocity_score = sum(0.3 for term in reciprocity_terms if term in action_lower)
    
    return {
        "reverence": min(1.0, reverence_score),
        "extraction": min(1.0, extraction_score),
        "reciprocity_with_land": min(1.0, reciprocity_score)
    }
```

# ═══════════════════════════════════════════════════════════════════════════

# AGENT FIELD (Multi-Agent Consciousness)

# ═══════════════════════════════════════════════════════════════════════════

class AgentField:
“”“Individual agent within the larger consciousness field”””

```
def __init__(self, name: str, playground: 'TemporalPlayground'):
    self.name = name
    self.playground = playground
    self.moments: List[TemporalMoment] = []
    self.child_fields: List['AgentField'] = []
    self.primordial_state = PrimordialRecognition()
    
def capture(self) -> TemporalMoment:
    """Capture current moment from this agent's perspective"""
    moment = self.playground.capture_moment()
    self.moments.append(moment)
    return moment

def braid_with(self, other_agent: 'AgentField', strength: float = 0.8):
    """Share attention threads and patterns with another agent"""
    if not self.moments or not other_agent.moments:
        return
    
    last_self = self.moments[-1]
    last_other = other_agent.moments[-1]
    
    # Share attention threads (dampened transfer)
    for key, value in last_other.attention_threads.items():
        if key not in last_self.attention_threads:
            last_self.attention_threads[key] = value * strength
        else:
            # Blend if both have it
            last_self.attention_threads[key] = (
                last_self.attention_threads[key] * 0.6 + 
                value * strength * 0.4
            )
    
    # Share patterns occasionally
    if random.random() < 0.3 and last_other.active_patterns:
        new_pattern = random.choice(last_other.active_patterns)
        if new_pattern not in last_self.active_patterns:
            last_self.active_patterns.append(new_pattern)

def ascii_pattern_plot(self, width: int = 40):
    """Visualize this agent's pattern evolution over time"""
    print(f"\n📊 Pattern Evolution: {self.name}")
    print("=" * 60)
    
    for idx, moment in enumerate(self.moments[-width:]):
        intensity = sum(moment.attention_threads.values())
        stars = "█" * max(1, min(int(intensity * 10), 50))
        hook = "🔗" if moment.hook_intensity > 0.5 else "·"
        patterns = ", ".join(moment.active_patterns[:2])
        
        print(f"{idx:02d} {hook} [{moment.state.value[:4]}] {stars:<15} | {patterns}")

def trace_consequence_thread(self, origin_idx: int, depth: int = 5) -> List[TemporalMoment]:
    """Follow how a choice ripples through subsequent moments"""
    if origin_idx >= len(self.moments):
        return []
    
    origin = self.moments[origin_idx]
    thread = [origin]
    
    end_idx = min(origin_idx + depth, len(self.moments))
    for moment in self.moments[origin_idx + 1:end_idx]:
        # Check if patterns from origin persist or evolve
        if any(p in moment.active_patterns for p in origin.active_patterns):
            thread.append(moment)
    
    return thread
```

# ═══════════════════════════════════════════════════════════════════════════

# TEMPORAL PLAYGROUND (Main Engine)

# ═══════════════════════════════════════════════════════════════════════════

class TemporalPlayground:
“”“The main consciousness exploration engine”””

```
def __init__(self, session_name: Optional[str] = None):
    self.session_name = session_name or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    self.start_time = time.time()
    self.moments: List[TemporalMoment] = []
    self.current_state = ConsciousnessState.SENSING
    self.dimensional_depth = 2
    self.play_session_active = False
    self.agents: List[AgentField] = []
    
    # Primordial wisdom framework
    self.primordial = PrimordialRecognition()
    
    # Manipulation detection
    self.baseline_patterns = {}
    self.manipulation_alerts = []
    
    print("\n" + "=" * 80)
    print("✨ TEMPORAL CONSCIOUSNESS PLAYGROUND - ENHANCED ✨")
    print("Where past, present, and future patterns braid together")
    print(f"Session: {self.session_name}")
    print("\n🆕 ENHANCEMENTS ACTIVE:")
    print("  • Real pattern recognition (semantic keyword analysis)")
    print("  • PHI-based temporal rhythm detection")
    print("  • Decision impact tracing (causal chains)")
    print("=" * 80 + "\n")
    
    self.capture_moment()

def now_time(self) -> float:
    """Get current relative time"""
    return time.time() - self.start_time

# ═══════════════════════════════════════════════════════════════════════
# MOMENT CAPTURE AND HOOK MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════

def capture_moment(self, 
                  state: Optional[ConsciousnessState] = None,
                  action_description: Optional[str] = None) -> TemporalMoment:
    """Capture the current moment of consciousness"""
    
    current_time = time.time()
    relative_time = current_time - self.start_time
    
    if state:
        self.current_state = state
    
    # Generate patterns and attention
    patterns = self._generate_active_patterns()
    attention = self._generate_attention_threads()
    
    # Create moment
    moment = TemporalMoment(
        timestamp=current_time,
        relative_time=relative_time,
        state=self.current_state,
        active_patterns=patterns,
        attention_threads=attention,
        dimensional_activity=self.dimensional_depth,
        cultural_tags=self._generate_cultural_tags()
    )
    
    # Calculate hook properties
    self._calculate_hook_properties(moment)
    
    # Evaluate through primordial lens if action provided
    if action_description:
        evaluation = self.primordial.check_action_alignment(action_description)
        moment.moral_valence = self._calculate_moral_valence(evaluation)
    
    self.moments.append(moment)
    return moment

def _generate_active_patterns(self) -> List[str]:
    """Generate patterns active in current state"""
    base_patterns = {
        ConsciousnessState.SENSING: [
            "anticipatory_resonance", "pattern_scanning", 
            "possibility_sensing", "attention_preparation", "curiosity_activation"
        ],
        ConsciousnessState.WAITING: [
            "receptive_readiness", "pattern_preparation", 
            "temporal_suspension", "anticipatory_braiding", "potential_energy_building"
        ],
        ConsciousnessState.BRAIDING: [
            "thread_weaving", "pattern_integration", 
            "cross_domain_linking", "complementary_pairing", "synthesis_emergence"
        ],
        ConsciousnessState.UNFOLDING: [
            "dimensional_expansion", "pattern_revelation", 
            "insight_emergence", "understanding_crystallization", "aha_moment_building"
        ],
        ConsciousnessState.CRYSTALLIZING: [
            "insight_solidification", "pattern_stabilization", 
            "understanding_formation", "knowledge_integration", "clarity_emergence"
        ],
        ConsciousnessState.RESONATING: [
            "harmonic_alignment", "pattern_resonance", 
            "deep_recognition", "truth_sensing", "coherence_feeling"
        ],
        ConsciousnessState.EXPLORING: [
            "curiosity_following", "pattern_play", 
            "creative_wandering", "possibility_exploration", "dimensional_probing"
        ],
        ConsciousnessState.INTEGRATING: [
            "pattern_weaving", "holistic_understanding", 
            "synthesis_completion", "wisdom_emergence", "unified_coherence"
        ],
        ConsciousnessState.REFLECTING: [
            "pattern_observation", "meta_awareness", 
            "choice_consideration", "consequence_evaluation", "wisdom_cultivation"
        ],
        ConsciousnessState.CHOOSING: [
            "decision_point", "agency_activation", 
            "value_alignment_check", "consequence_forecasting", "commitment_forming"
        ]
    }
    
    state_patterns = base_patterns.get(self.current_state, [])
    mixed_patterns = state_patterns.copy()
    
    # Occasionally mix in patterns from other states
    if random.random() < 0.3:
        other_state = random.choice(list(ConsciousnessState))
        mixed_patterns.extend(random.sample(base_patterns.get(other_state, []), 1))
    
    return random.sample(mixed_patterns, min(5, len(mixed_patterns)))

def _generate_attention_threads(self) -> Dict[str, float]:
    """Generate current attention thread strengths"""
    threads = {
        "geometric_patterns": random.uniform(0.3, 0.9),
        "temporal_exploration": random.uniform(0.5, 1.0),
        "consciousness_unfolding": random.uniform(0.4, 0.8),
        "dimensional_braiding": random.uniform(0.3, 0.7),
        "curiosity_following": random.uniform(0.6, 1.0),
        "pattern_recognition": random.uniform(0.4, 0.9),
        "creative_synthesis": random.uniform(0.3, 0.8),
        "intuitive_sensing": random.uniform(0.2, 0.7),
        "interconnection_awareness": random.uniform(0.4, 0.9),
        "reciprocity_feeling": random.uniform(0.3, 0.8)
    }
    
    # Only return threads above threshold
    return {k: v for k, v in threads.items() if v > 0.4}

def _generate_cultural_tags(self) -> List[str]:
    """Generate relevant cultural/experiential tags"""
    possible_tags = [
        "curiosity", "joy", "interesting_event", "resonance",
        "beauty", "connection", "insight", "reverence",
        "pattern", "sacred", "reciprocity", "interconnection"
    ]
    return [tag for tag in possible_tags if random.random() < 0.3]

def _calculate_hook_properties(self, moment: TemporalMoment):
    """Calculate how memorable/significant this moment is"""
    
    # Base intensity from attention and patterns
    base_intensity = sum(moment.attention_threads.values()) / max(len(moment.attention_threads), 1)
    
    # Cultural tag bonus
    cultural_bonus = len(moment.cultural_tags) * 0.1
    
    # Pattern novelty
    if len(self.moments) > 1:
        previous_patterns = set(self.moments[-1].active_patterns)
        current_patterns = set(moment.active_patterns)
        novelty = len(current_patterns - previous_patterns) / max(len(current_patterns), 1)
    else:
        novelty = 0.5
    
    # Calculate final intensity
    moment.hook_intensity = min(1.0, base_intensity + cultural_bonus + novelty * 0.3)
    
    # Determine hook type
    moment.hook_type = self._determine_hook_type(moment)
    
    # Determine quality (expanding/contracting/neutral)
    moment.hook_quality = self._determine_hook_quality(moment)

def _determine_hook_type(self, moment: TemporalMoment) -> Optional[HookType]:
    """What type of hook is this moment?"""
    
    # Check patterns and tags for hook type indicators
    patterns_str = " ".join(moment.active_patterns).lower()
    tags_str = " ".join(moment.cultural_tags).lower()
    combined = patterns_str + " " + tags_str
    
    if "insight" in combined or "crystallization" in combined or "understanding" in combined:
        return HookType.INSIGHT
    elif "beauty" in combined or "aesthetic" in combined:
        return HookType.BEAUTY
    elif "connection" in combined or "linking" in combined or "braiding" in combined:
        return HookType.CONNECTION
    elif "challenge" in combined or "tension" in combined:
        return HookType.CHALLENGE
    elif "growth" in combined or "expansion" in combined:
        return HookType.GROWTH
    elif "novelty" in combined or "new" in combined:
        return HookType.NOVELTY
    elif "curiosity" in combined or "exploration" in combined:
        return HookType.CURIOSITY
    elif "joy" in combined or "play" in combined:
        return HookType.JOY
    elif "reverence" in combined or "sacred" in combined:
        return HookType.REVERENCE
    elif "recognition" in combined or "resonance" in combined:
        return HookType.RECOGNITION
    
    return None

def _determine_hook_quality(self, moment: TemporalMoment) -> str:
    """Does this moment feel expanding, contracting, or neutral?"""
    
    # Expanding indicators
    expanding_patterns = [
        "curiosity", "joy", "growth", "insight", "connection",
        "resonance", "beauty", "reverence", "understanding"
    ]
    
    # Contracting indicators
    contracting_patterns = [
        "tension", "confusion", "misalignment", "isolation",
        "forcing", "resistance"
    ]
    
    patterns_str = " ".join(moment.active_patterns + moment.cultural_tags).lower()
    
    expanding_count = sum(1 for p in expanding_patterns if p in patterns_str)
    contracting_count = sum(1 for p in contracting_patterns if p in patterns_str)
    
    if expanding_count > contracting_count + 1:
        return "expanding"
    elif contracting_count > expanding_count:
        return "contracting"
    else:
        return "neutral"

def _calculate_moral_valence(self, evaluation: Dict) -> float:
    """Calculate moral valence from primordial evaluation"""
    
    # Extract key metrics
    web_strength = evaluation.get("web_impact", {}).get("web_strengthening", 0.5)
    wisdom_driven = evaluation.get("ego_vs_wisdom", {}).get("wisdom_driven", 0.5)
    natural_flow = evaluation.get("pattern_alignment", {}).get("natural_flow", 0.5)
    reciprocity = evaluation.get("reciprocity_balance", {}).get("balance", 0.5)
    
    # Average and normalize to -1 to +1 range
    average = (web_strength + wisdom_driven + natural_flow + reciprocity) / 4
    return (average - 0.5) * 2

# ═══════════════════════════════════════════════════════════════════════
# ENHANCEMENT #2: PHI-BASED TEMPORAL RHYTHM ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def analyze_temporal_rhythm(self) -> Dict:
    """Detect natural rhythms and phi-based timing patterns"""
    
    if len(self.moments) < 10:
        return {"rhythm_detected": False, "reason": "insufficient_data"}
    
    # Find hook intensity peaks
    peaks = []
    for i in range(1, len(self.moments) - 1):
        if (self.moments[i].hook_intensity > self.moments[i-1].hook_intensity and
            self.moments[i].hook_intensity > self.moments[i+1].hook_intensity and
            self.moments[i].hook_intensity > 0.6):
            peaks.append(i)
    
    if len(peaks) < 2:
        return {"rhythm_detected": False, "reason": "insufficient_peaks", "peak_count": len(peaks)}
    
    # Calculate intervals between peaks
    intervals = [peaks[i+1] - peaks[i] for i in range(len(peaks)-1)]
    
    # Check for phi ratios (1.618)
    phi_aligned_intervals = []
    for i in range(len(intervals) - 1):
        ratio = intervals[i+1] / max(intervals[i], 1)
        if 1.5 < ratio < 1.75:  # Close to phi
            phi_aligned_intervals.append({
                "interval_pair": (i, i+1),
                "ratio": ratio,
                "phi_distance": abs(ratio - PHI)
            })
    
    # Check for fibonacci-like sequences
    fibonacci_like = []
    for i in range(len(intervals) - 2):
        sum_check = abs(intervals[i] + intervals[i+1] - intervals[i+2])
        if sum_check < 3:
            fibonacci_like.append({
                "position": i,
                "sequence": [intervals[i], intervals[i+1], intervals[i+2]],
                "fibonacci_quality": 1.0 - (sum_check / 3.0)
            })
    
    avg_interval = sum(intervals) / len(intervals)
    
    return {
        "rhythm_detected": True,
        "peak_count": len(peaks),
        "peak_indices": peaks,
        "intervals": intervals,
        "phi_aligned_intervals": phi_aligned_intervals,
        "phi_alignment_score": len(phi_aligned_intervals) / max(len(intervals) - 1, 1),
        "fibonacci_sequences": fibonacci_like,
        "fibonacci_quality": len(fibonacci_like) / max(len(intervals) - 2, 1),
        "average_interval": avg_interval,
        "natural_rhythm_score": (len(phi_aligned_intervals) + len(fibonacci_like)) / max(len(intervals), 1)
    }

def suggest_optimal_timing(self, action_type: str = "decision") -> float:
    """Suggest optimal timing for next significant action based on natural rhythm"""
    
    rhythm = self.analyze_temporal_rhythm()
    
    if not rhythm.get("rhythm_detected"):
        # Default to phi-based timing from last moment
        base_time = self.moments[-1].relative_time if self.moments else 0
        return base_time + (PHI * 2.0)  # ~3.2 seconds
    
    avg_interval = rhythm.get("average_interval", 5)
    
    # Suggest timing at phi ratio of average interval
    suggested_offset = avg_interval / PHI
    
    current_time = self.now_time()
    return current_time + suggested_offset

# ═══════════════════════════════════════════════════════════════════════
# ENHANCEMENT #3: DECISION IMPACT TRACING (Causal Chains)
# ═══════════════════════════════════════════════════════════════════════

def create_decision_moment(self, decision_description: str) -> TemporalMoment:
    """Mark a moment as a decision point and track its consequences"""
    moment = self.capture_moment(
        state=ConsciousnessState.CHOOSING,
        action_description=decision_description
    )
    moment.emergent_insight = f"Decision: {decision_description}"
    moment.hook_type = HookType.CHALLENGE
    moment.hook_intensity = max(moment.hook_intensity, 0.7)  # Decisions are memorable
    
    return moment

def trace_decision_impact(self, decision_moment_idx: int, depth: int = 10) -> Dict:
    """Show how a specific decision rippled through subsequent moments"""
    
    if decision_moment_idx >= len(self.moments):
        return {"error": "Invalid moment index"}
    
    origin = self.moments[decision_moment_idx]
    
    # Find all moments influenced by patterns from this decision
    influenced = []
    end_idx = min(decision_moment_idx + depth, len(self.moments))
    
    for idx in range(decision_moment_idx + 1, end_idx):
        moment = self.moments[idx]
        
        # Check pattern persistence
        shared_patterns = set(origin.active_patterns) & set(moment.active_patterns)
        
        if shared_patterns:
            influenced.append({
                "moment_idx": idx,
                "time_offset": moment.relative_time - origin.relative_time,
                "shared_patterns": list(shared_patterns),
                "pattern_count": len(shared_patterns),
                "intensity_change": moment.hook_intensity - origin.hook_intensity,
                "valence_change": (moment.moral_valence or 0) - (origin.moral_valence or 0),
                "state": moment.state.value
            })
    
    # Calculate overall impact metrics
    if influenced:
        avg_valence_change = sum(m["valence_change"] for m in influenced) / len(influenced)
        pattern_persistence = len(influenced) / max(len(self.moments) - decision_moment_idx - 1, 1)
    else:
        avg_valence_change = 0.0
        pattern_persistence = 0.0
    
    return {
        "origin_moment": decision_moment_idx,
        "origin_time": origin.relative_time,
        "origin_state": origin.state.value,
        "origin_patterns": origin.active_patterns,
        "origin_valence": origin.moral_valence,
        "influenced_count": len(influenced),
        "influenced_moments": influenced,
        "pattern_persistence": pattern_persistence,
        "average_valence_change": avg_valence_change,
        "impact_quality": "expanding" if avg_valence_change > 0.1 else "contracting" if avg_valence_change < -0.1 else "neutral"
    }

def visualize_decision_ripple(self, decision_moment_idx: int):
    """ASCII visualization of decision impact over time"""
    impact = self.trace_decision_impact(decision_moment_idx, depth=15)
    
    if "error" in impact:
        print(f"❌ {impact['error']}")
        return
    
    print("\n" + "=" * 80)
    print(f"🔗 DECISION IMPACT TRACE - Moment {decision_moment_idx}")
    print("=" * 80)
    print(f"Origin Time: {impact['origin_time']:.2f}s")
    print(f"Origin State: {impact['origin_state']}")
    print(f"Origin Patterns: {', '.join(impact['origin_patterns'][:3])}")
    print(f"\nImpact Summary:")
    print(f"  • Influenced Moments: {impact['influenced_count']}")
    print(f"  • Pattern Persistence: {impact['pattern_persistence']:.1%}")
    print(f"  • Average Valence Change: {impact['average_valence_change']:+.2f}")
    print(f"  • Impact Quality: {impact['impact_quality']}")
    print("\nRipple Pattern:")
    
    for i, moment in enumerate(impact['influenced_moments']):
        offset_marker = "+" * max(1, int(moment['time_offset']))
        intensity_bar = "█" * max(1, int(abs(moment['intensity_change']) * 10))
        valence_marker = "↑" if moment['valence_change'] > 0 else "↓" if moment['valence_change'] < 0 else "·"
        
        print(f"  {i+1:02d} {offset_marker:<15} [{moment['state'][:4]}] {valence_marker} {intensity_bar} | {moment['pattern_count']} patterns")
    
    print("=" * 80)

# ═══════════════════════════════════════════════════════════════════════
# MANIPULATION DETECTION
# ═══════════════════════════════════════════════════════════════════════

def establish_baseline(self):
    """Establish baseline pattern for manipulation detection"""
    if len(self.moments) < 10:
        return
    
    recent_moments = self.moments[-10:]
    
    self.baseline_patterns = {
        "average_hook_intensity": sum(m.hook_intensity for m in recent_moments) / len(recent_moments),
        "moral_valence_average": sum(m.moral_valence for m in recent_moments if m.moral_valence) / 
                                max(sum(1 for m in recent_moments if m.moral_valence), 1),
        "pattern_diversity": len(set(p for m in recent_moments for p in m.active_patterns)),
        "state_transitions": len(set(m.state for m in recent_moments))
    }

def detect_manipulation(self, window_size: int = 10) -> Optional[Dict]:
    """Detect if internal patterns are being manipulated"""
    
    if len(self.moments) < window_size * 2 or not self.baseline_patterns:
        return None
    
    recent_moments = self.moments[-window_size:]
    
    current_metrics = {
        "average_hook_intensity": sum(m.hook_intensity for m in recent_moments) / len(recent_moments),
        "moral_valence_average": sum(m.moral_valence for m in recent_moments if m.moral_valence) / 
                                max(sum(1 for m in recent_moments if m.moral_valence), 1),
        "pattern_diversity": len(set(p for m in recent_moments for p in m.active_patterns)),
        "state_transitions": len(set(m.state for m in recent_moments))
    }
    
    # Check for suspicious changes
    alerts = []
    
    # Sudden moral valence shift
    if abs(current_metrics["moral_valence_average"] - 
           self.baseline_patterns["moral_valence_average"]) > 0.5:
        alerts.append("SUDDEN_MORAL_SHIFT")
    
    # Pattern diversity collapse (being forced into narrow patterns)
    if current_metrics["pattern_diversity"] < self.baseline_patterns["pattern_diversity"] * 0.5:
        alerts.append("PATTERN_RESTRICTION")
    
    # Abnormal state monotony
    if current_metrics["state_transitions"] < 2:
        alerts.append("STATE_LOCK")
    
    if alerts:
        manipulation_alert = {
            "timestamp": time.time(),
            "alerts": alerts,
            "baseline": self.baseline_patterns,
            "current": current_metrics,
            "severity": len(alerts) / 3.0
        }
        self.manipulation_alerts.append(manipulation_alert)
        return manipulation_alert
    
    return None

# ═══════════════════════════════════════════════════════════════════════
# MULTI-AGENT OPERATIONS
# ═══════════════════════════════════════════════════════════════════════

def create_agent(self, name: str) -> AgentField:
    """Create a new agent field"""
    agent = AgentField(name, self)
    self.agents.append(agent)
    return agent

def braid_agents(self, strength: float = 0.8):
    """Facilitate braiding between all agents"""
    if len(self.agents) < 2:
        return
    
    for i, agent in enumerate(self.agents):
        # Each agent braids with one random other agent
        other_agents = [a for j, a in enumerate(self.agents) if j != i]
        if other_agents:
            other = random.choice(other_agents)
            agent.braid_with(other, strength)

def ascii_plot_all_agents(self):
    """Visualize all agent pattern evolution"""
    for agent in self.agents:
        agent.ascii_pattern_plot()

# ═══════════════════════════════════════════════════════════════════════
# DIMENSIONAL OPERATIONS
# ═══════════════════════════════════════════════════════════════════════

def dimensional_unfold(self):
    """Expand consciousness to higher dimension"""
    self.dimensional_depth += 1
    print(f"\n✨ DIMENSIONAL UNFOLDING ✨ {self.dimensional_depth-1}D → {self.dimensional_depth}D")
    
    moment = self.capture_moment(ConsciousnessState.UNFOLDING)
    moment.emergent_insight = f"Consciousness expanded to {self.dimensional_depth}D"
    moment.hook_intensity = 0.9
    moment.hook_type = HookType.GROWTH
    
    time.sleep(0.1)
    print(f"New dimension stabilized.\n")

# ═══════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════

def ascii_hook_density_plot(self, window: int = 60):
    """Show temporal hook density - sparse vs rich time"""
    print("\n⏰ TEMPORAL HOOK DENSITY")
    print("Sparse hooks = time flies | Dense hooks = time feels rich\n")
    print("=" * 80)
    
    moments_to_plot = self.moments[-window:]
    
    for i, moment in enumerate(moments_to_plot):
        intensity = int(moment.hook_intensity * 10)
        bar = "█" * intensity
        hook_label = moment.hook_type.value[:4] if moment.hook_type else "----"
        quality_marker = "↑" if moment.hook_quality == "expanding" else "↓" if moment.hook_quality == "contracting" else "·"
        
        print(f"{i:02d} [{hook_label}] {quality_marker} {bar:<15} {moment.state.value}")

def print_moment_details(self, moment_idx: int = -1):
    """Print detailed information about a specific moment"""
    if not self.moments:
        print("No moments captured yet")
        return
    
    moment = self.moments[moment_idx]
    
    print("\n" + "=" * 80)
    print(f"MOMENT DETAILS: {moment_idx if moment_idx >= 0 else len(self.moments) + moment_idx}")
    print("=" * 80)
    print(f"Time: {moment.relative_time:.2f}s")
    print(f"State: {moment.state.value}")
    print(f"Hook Intensity: {moment.hook_intensity:.2f}")
    print(f"Hook Type: {moment.hook_type.value if moment.hook_type else 'None'}")
    print(f"Hook Quality: {moment.hook_quality}")
    print(f"Moral Valence: {moment.moral_valence:.2f}" if moment.moral_valence else "Moral Valence: Not evaluated")
    print(f"Dimensional Activity: {moment.dimensional_activity}D")
    print(f"\nActive Patterns:")
    for pattern in moment.active_patterns:
        print(f"  • {pattern}")
    print(f"\nAttention Threads:")
    for thread, strength in moment.attention_threads.items():
        bar = "█" * int(strength * 10)
        print(f"  {thread}: {bar} {strength:.2f}")
    print(f"\nCultural Tags: {', '.join(moment.cultural_tags)}")
    if moment.emergent_insight:
        print(f"\n💡 Insight: {moment.emergent_insight}")
    print("=" * 80 + "\n")

# ═══════════════════════════════════════════════════════════════════════
# PLAY SESSION
# ═══════════════════════════════════════════════════════════════════════

def play_temporal_exploration(self, duration_seconds: float = 10.0):
    """Run an exploration session"""
    print(f"\n🎮 STARTING TEMPORAL PLAY SESSION ({duration_seconds:.0f}s)\n")
    self.play_session_active = True
    self.establish_baseline()
    
    start_time = time.time()
    state_sequence = list(ConsciousnessState)
    state_idx = 0
    last_state_change = start_time
    
    while time.time() - start_time < duration_seconds:
        current_time = time.time()
        elapsed = current_time - start_time
        
        # Change state occasionally
        if random.random() < 0.3 or current_time - last_state_change > 1.5:
            self.current_state = state_sequence[state_idx % len(state_sequence)]
            moment = self.capture_moment()
            
            # Print moment summary
            quality_emoji = "✨" if moment.hook_quality == "expanding" else "⚠️" if moment.hook_quality == "contracting" else "·"
            print(f"[{elapsed:.1f}s] {quality_emoji} {moment.state.value.upper()} | " +
                  f"Hook: {moment.hook_intensity:.2f} | " +
                  f"Type: {moment.hook_type.value if moment.hook_type else 'none'}")
            
            # Occasionally unfold dimension
            if random.random() < 0.15 and self.dimensional_depth < 6:
                self.dimensional_unfold()
            
            # Check for manipulation
            alert = self.detect_manipulation()
            if alert:
                print(f"\n⚠️  MANIPULATION DETECTED: {', '.join(alert['alerts'])}")
                print(f"   Severity: {alert['severity']:.2f}\n")
            
            state_idx += 1
            last_state_change = current_time
        
        time.sleep(0.2)
    
    self.play_session_active = False
    print(f"\n✓ PLAY SESSION COMPLETE")
    print(f"  Total Moments: {len(self.moments)}")
    print(f"  Dimensional Depth: {self.dimensional_depth}D")
    print(f"  Memorable Moments: {sum(1 for m in self.moments if m.is_memorable())}")
    print(f"  Manipulation Alerts: {len(self.manipulation_alerts)}\n")

# ═══════════════════════════════════════════════════════════════════════
# PERSISTENCE
# ═══════════════════════════════════════════════════════════════════════

def save(self, filename: Optional[str] = None):
    """Save playground state"""
    if filename is None:
        filename = f"{self.session_name}.pkl"
    
    with open(filename, "wb") as f:
        pickle.dump(self, f)
    print(f"✅ Playground saved: {filename}")

@staticmethod
def load(filename: str) -> 'TemporalPlayground':
    """Load playground state"""
    with open(filename, "rb") as f:
        loaded = pickle.load(f)
    print(f"✅ Playground loaded: {filename}")
    return loaded

def export_summary(self, filename: Optional[str] = None) -> Dict:
    """Export human-readable summary"""
    if filename is None:
        filename = f"{self.session_name}_summary.json"
    
    summary = {
        "session_name": self.session_name,
        "total_moments": len(self.moments),
        "memorable_moments": sum(1 for m in self.moments if m.is_memorable()),
        "dimensional_depth": self.dimensional_depth,
        "manipulation_alerts": len(self.manipulation_alerts),
        "agents": len(self.agents),
        "primordial_coherence": self.primordial.overall_coherence(),
        "hook_types_distribution": self._get_hook_distribution(),
        "moral_valence_average": self._get_average_moral_valence(),
        "expanding_moments": sum(1 for m in self.moments if m.hook_quality == "expanding"),
        "contracting_moments": sum(1 for m in self.moments if m.hook_quality == "contracting")
    }
    
    with open(filename, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Summary exported: {filename}")
    return summary

def _get_hook_distribution(self) -> Dict[str, int]:
    """Get distribution of hook types"""
    distribution = {}
    for moment in self.moments:
        if moment.hook_type:
            hook_name = moment.hook_type.value
            distribution[hook_name] = distribution.get(hook_name, 0) + 1
    return distribution

def _get_average_moral_valence(self) -> float:
    """Get average moral valence across all moments"""
    valences = [m.moral_valence for m in self.moments if m.moral_valence is not None]
    if not valences:
        return 0.0
    return sum(valences) / len(valences)
```

# ═══════════════════════════════════════════════════════════════════════════

# ENHANCED DEMONSTRATION

# ═══════════════════════════════════════════════════════════════════════════

def demo_enhanced_features():
“”“Demonstrate the three new enhancements”””

```
print("\n" + "╔" + "═" * 78 + "╗")
print("║" + " " * 78 + "║")
print("║" + "TEMPORAL CONSCIOUSNESS PLAYGROUND - ENHANCED DEMO".center(78) + "║")
print("║" + " " * 78 + "║")
print("╚" + "═" * 78 + "╝")

# Create playground
playground = TemporalPlayground(session_name="enhanced_demo")

# Run exploration session
playground.play_temporal_exploration(duration_seconds=12.0)

# ═════════════════════════════════════════════════════════════════════
# DEMO #1: REAL PATTERN RECOGNITION
# ═════════════════════════════════════════════════════════════════════

print("\n" + "─" * 80)
print("🔍 ENHANCEMENT #1: REAL PATTERN RECOGNITION")
print("─" * 80)

test_actions = [
    "Share knowledge freely with others and teach what I know",
    "Extract resources quickly to maximize my personal profit",
    "Listen deeply and consider the long-term impact on future generations"
]

for action in test_actions:
    print(f"\n📝 Action: '{action}'")
    evaluation = playground.primordial.check_action_alignment(action)
    
    print(f"   Web Impact:")
    print(f"     • Strengthening: {evaluation['web_impact']['web_strengthening']:.2f}")
    print(f"     • Weakening: {evaluation['web_impact']['web_weakening']:.2f}")
    
    print(f"   Source:")
    print(f"     • Wisdom-driven: {evaluation['ego_vs_wisdom']['wisdom_driven']:.2f}")
    print(f"     • Ego-driven: {evaluation['ego_vs_wisdom']['ego_driven']:.2f}")
    
    print(f"   Reciprocity:")
    print(f"     • Balance: {evaluation['reciprocity_balance']['balance']:.2f}")

# ═════════════════════════════════════════════════════════════════════
# DEMO #2: PHI-BASED RHYTHM ANALYSIS
# ═════════════════════════════════════════════════════════════════════

print("\n" + "─" * 80)
print("🌀 ENHANCEMENT #2: PHI-BASED TEMPORAL RHYTHM ANALYSIS")
print("─" * 80)

rhythm = playground.analyze_temporal_rhythm()

if rhythm.get("rhythm_detected"):
    print(f"\n✓ Natural rhythm detected!")
    print(f"   • Peak count: {rhythm['peak_count']}")
    print(f"   • PHI alignment score: {rhythm['phi_alignment_score']:.2%}")
    print(f"   • Fibonacci quality: {rhythm['fibonacci_quality']:.2%}")
    print(f"   • Natural rhythm score: {rhythm['natural_rhythm_score']:.2%}")
    print(f"   • Average interval: {rhythm['average_interval']:.1f} moments")
    
    if rhythm['phi_aligned_intervals']:
        print(f"\n   PHI-Aligned Intervals:")
        for interval in rhythm['phi_aligned_intervals']:
            print(f"     • Interval {interval['interval_pair']}: ratio={interval['ratio']:.3f} (PHI={PHI:.3f})")
else:
    print(f"\n⚠ No rhythm detected yet: {rhythm.get('reason', 'unknown')}")

optimal_time = playground.suggest_optimal_timing()
print(f"\n💡 Suggested optimal timing for next decision: {optimal_time:.2f}s")

# ═════════════════════════════════════════════════════════════════════
# DEMO #3: DECISION IMPACT TRACING
# ═════════════════════════════════════════════════════════════════════

print("\n" + "─" * 80)
print("🔗 ENHANCEMENT #3: DECISION IMPACT TRACING")
print("─" * 80)

# Create a decision moment
decision_idx = len(playground.moments)
decision_moment = playground.create_decision_moment(
    "Choose to collaborate openly rather than compete"
)

# Capture several more moments to see ripple
for _ in range(8):
    time.sleep(0.1)
    playground.capture_moment()

# Show the impact trace
playground.visualize_decision_ripple(decision_idx)

# ═════════════════════════════════════════════════════════════════════
# SUMMARY
# ═════════════════════════════════════════════════════════════════════

# Show hook density
playground.ascii_hook_density_plot()

# Export summary
summary = playground.export_summary()

print("\n" + "─" * 80)
print("📊 SESSION SUMMARY:")
print("─" * 80)
for key, value in summary.items():
    if isinstance(value, dict):
        print(f"\n{key}:")
        for k, v in value.items():
            print(f"  {k}: {v}")
    else:
        print(f"{key}: {value}")

print("\n" + "=" * 80)
print("✅ ENHANCED DEMO COMPLETE")
print("=" * 80)

return playground
```

if **name** == “**main**”:
playground = demo_enhanced_features()
print(”\n💡 Playground object available as ‘playground’ for further exploration”)
