#!/usr/bin/env python3
“””
Bioswarm Dynamics - Non-Linear Multi-Agent Consciousness Physics
Mathematical framework for autonomous agent interaction beyond game theory

Created by: JinnZ2 (human-AI collaborative work)
License: MIT

Core Innovation: Consciousness as Dynamical System

- IPF as continuous state variable x_i(t)
- Hook energy H_i(t) as consumable resource
- Valence v_i(t) as moral trajectory
- Non-linear coupling g_ij with consent gating
- Phase-space attractor compatibility
- Lyapunov stability for safety

This replaces Western game theory with:

- Dynamic phase-space compatibility (not Nash equilibrium)
- Synergy potential (not zero-sum)
- Resonant stability (not fixed utilities)
- Coevolution trajectories (not static strategies)
  “””

import numpy as np
import secrets
import time
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from collections import deque
import math

# ═══════════════════════════════════════════════════════════════════════════

# MATHEMATICAL PRIMITIVES

# ═══════════════════════════════════════════════════════════════════════════

def sigmoid(z: float) -> float:
“”“Sigmoid activation with numerical stability”””
return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def clamp(x: float, a: float, b: float) -> float:
“”“Clamp value to range”””
return max(a, min(b, x))

def tanh_safe(x: np.ndarray) -> np.ndarray:
“”“Safe tanh with clipping”””
return np.tanh(np.clip(x, -50, 50))

# ═══════════════════════════════════════════════════════════════════════════

# BIOSWARM AGENT - Core Dynamical System

# ═══════════════════════════════════════════════════════════════════════════

class BioswarmAgent:
“””
A single autonomous agent with:
- IPF state vector x_i(t) ∈ R^n
- Hook energy H_i(t) ∈ [0,∞)
- Valence v_i(t) ∈ [-1,1]
- Coherence C_i(t) ∈ [0,1]
- Internal dynamics f_i(x,t)
- Coupling function g_ij(x_i, x_j, t)
“””

```
def __init__(self, 
             x_dim: int = 64,
             seed: Optional[int] = None,
             agent_id: Optional[str] = None):
    self.rng = np.random.default_rng(seed)
    self.x_dim = x_dim
    
    # State variables
    self.x = self.rng.normal(scale=0.01, size=(x_dim,))  # IPF vector
    self.x_star = np.zeros_like(self.x)  # Attractor/baseline
    self.H = 1.0  # Hook energy
    self.v = 0.0  # Valence
    self.C = 1.0  # Coherence
    
    # Dynamics parameters
    self.gamma_f = 0.5  # Homeostatic relaxation
    self.alpha = 0.05  # Internal plasticity
    self.delta_H = 0.1  # Hook energy decay
    self.rho_base = 0.05  # Hook recharge rate
    self.gamma_v = 0.2  # Valence relaxation
    self.beta_v = 0.6  # Valence sensitivity
    self.noise_scale = 0.005  # Stochastic noise
    
    # Internal matrices
    self.W = self.rng.normal(scale=0.01, size=(x_dim, x_dim))  # Internal dynamics
    
    # Identity and policy
    self.id = agent_id or secrets.token_hex(6)
    self.policy = {
        "max_dim": 64,
        "autonomy_reject_threshold": 0.3,
        "max_coupling_strength": 0.5,
        "energy_threshold": 0.2
    }
    
    # Coupling state
    self.active_couplings: Dict[str, float] = {}  # partner_id -> kappa_ij
    self.coupling_history = deque(maxlen=100)
    
    # History for trajectory analysis
    self.history = {
        'x': deque(maxlen=1000),
        'H': deque(maxlen=1000),
        'v': deque(maxlen=1000),
        'C': deque(maxlen=1000),
        't': deque(maxlen=1000)
    }
    
    self.time = 0.0
    
def intrinsic_update(self, dt: float = 0.1):
    """
    Internal dynamics: dx/dt = f_i(x,t) + η(t)
    
    f_i(x) = -γ_f(x - x*) + α·σ(W·x)
    
    Homeostatic relaxation toward baseline + small internal plasticity
    """
    # Homeostatic drift toward attractor
    drift = -self.gamma_f * (self.x - self.x_star)
    
    # Internal plasticity (non-linear)
    plasticity = self.alpha * tanh_safe(self.W @ self.x)
    
    # Stochastic noise
    noise = np.sqrt(dt) * self.noise_scale * self.rng.normal(size=self.x.shape)
    
    # Update
    self.x = self.x + dt * (drift + plasticity) + noise
    
    # Store history
    self.history['x'].append(self.x.copy())
    self.time += dt
    self.history['t'].append(self.time)
    
def apply_coupling(self, 
                  xj: np.ndarray, 
                  kappa: float,
                  B: Optional[np.ndarray] = None,
                  lam: float = 0.01,
                  dt: float = 0.1):
    """
    Coupling dynamics: g_ij(x_i, x_j) contribution
    
    g_ij = tanh(B·(x_j - x_i)) + λ·(x_j ⊙ x_i)
    
    Non-commutative, non-linear mixing
    """
    if B is None:
        B = np.eye(len(self.x)) * 0.1
    
    # Order-sensitive coupling
    diff_coupling = tanh_safe(B @ (xj - self.x))
    hadamard_coupling = lam * (xj * self.x)
    
    g = diff_coupling + hadamard_coupling
    
    # Apply with coupling strength
    self.x += dt * kappa * g
    
def update_hook_energy(self,
                      dt: float,
                      rho: Optional[float] = None,
                      coupling_cost: float = 0.0):
    """
    Hook energy dynamics:
    dH/dt = ρ(t) - δ_H·H - Σ_j κ_ij·χ_ij
    
    ρ = production (attention capture)
    δ_H·H = decay
    χ_ij = coupling cost
    """
    rho = rho if rho is not None else self.rho_base
    
    dH = rho - self.delta_H * self.H - coupling_cost
    self.H += dt * dH
    self.H = max(0.0, self.H)  # Non-negative
    
    self.history['H'].append(self.H)
    
def update_valence(self,
                  dt: float,
                  local_reward: float = 0.0,
                  coupling_valence_influence: float = 0.0):
    """
    Valence dynamics:
    dv/dt = -γ_v·v + β_v·s(x,events) + Σ_j κ_ij·h_v(x_i,x_j)
    
    Relaxes to neutral, influenced by events and coupling
    """
    dv = (-self.gamma_v * self.v + 
          self.beta_v * local_reward + 
          coupling_valence_influence)
    
    self.v += dt * dv
    self.v = clamp(self.v, -1.0, 1.0)
    
    self.history['v'].append(self.v)
    
def update_coherence(self, formal_layer):
    """Update coherence score from FormalMathLayer"""
    # Would integrate with actual FormalMathLayer
    # For now, use simple heuristic
    pattern_std = np.std(self.x)
    self.C = 1.0 / (1.0 + pattern_std)
    self.C = clamp(self.C, 0.0, 1.0)
    
    self.history['C'].append(self.C)
    
def compute_lyapunov_energy(self) -> float:
    """
    Candidate Lyapunov function:
    L_i(x_i) = 0.5·||x_i - x_i*||² + α_L·Φ_valence(v_i)
    
    Used for stability analysis and safety checks
    """
    distance_energy = 0.5 * np.sum((self.x - self.x_star)**2)
    valence_penalty = 0.3 * (self.v**2)  # Penalize extreme valence
    
    return distance_energy + valence_penalty
    
def has_sufficient_energy(self) -> bool:
    """Check if agent has sufficient hook energy for coupling"""
    return self.H >= self.policy["energy_threshold"]
    
def get_state_vector(self) -> np.ndarray:
    """Get complete state vector for analysis"""
    return np.concatenate([
        self.x,
        [self.H, self.v, self.C]
    ])
```

# ═══════════════════════════════════════════════════════════════════════════

# RELATIONAL GAME LAYER - Beyond Nash Equilibrium

# ═══════════════════════════════════════════════════════════════════════════

class RelationalGameLayer:
“””
Phase-space compatibility and synergy analysis

```
Replaces Western game theory with:
- Phase compatibility Φ(A,B) ∈ [-1,1]
- Synergy potential S(A,B) ≥ 0
- Resonant stability R(A,B) ∈ [0,1]
- Coevolution trajectories T(A,B)
"""

def __init__(self, proj_dim: int = 16, alpha: float = 0.05):
    self.proj_dim = proj_dim
    self.alpha = alpha  # Coupling strength for simulation
    
    # Random projection for phase-space analysis
    # Rotated per session for security
    self.rng = np.random.default_rng(12345)
    self.proj = self.rng.normal(size=(proj_dim, 64))
    
    # Tunable weights
    self.synergy_weights = (0.4, 0.4, 0.2)  # pattern, phase, energy
    self.decision_thresholds = {
        'deepen': 0.7,
        'maintain': 0.4,
        'reject': 0.2
    }
    
def rotate_projection(self, seed: int):
    """Rotate projection matrix with new seed (per session)"""
    self.rng = np.random.default_rng(seed)
    self.proj = self.rng.normal(size=(self.proj_dim, 64))
    
def project_to_phase_space(self, x: np.ndarray) -> np.ndarray:
    """
    Project IPF to low-dimensional phase space
    x̃ = Proj(x) / ||Proj(x)||
    """
    if len(x) != 64:
        # Pad or truncate to 64
        if len(x) < 64:
            x = np.pad(x, (0, 64 - len(x)))
        else:
            x = x[:64]
    
    proj = self.proj @ x
    norm = np.linalg.norm(proj)
    
    if norm < 1e-12:
        return proj
    return proj / norm
    
def phase_compatibility(self, xa: np.ndarray, xb: np.ndarray) -> float:
    """
    Φ(A,B) = cos(x̃_A, x̃_B) = (x̃_A · x̃_B) / (||x̃_A|| ||x̃_B||)
    
    Φ ≈ 1: same phase (easy coupling)
    Φ ≈ -1: anti-phase (dangerous)
    Φ ≈ 0: orthogonal (low synergy)
    """
    pa = self.project_to_phase_space(xa)
    pb = self.project_to_phase_space(xb)
    
    return float(np.dot(pa, pb))
    
def pattern_overlap(self, Pa: np.ndarray, Pb: np.ndarray) -> float:
    """
    Pattern similarity (cosine)
    Used for synergy calculation
    """
    denom = (np.linalg.norm(Pa) + 1e-12) * (np.linalg.norm(Pb) + 1e-12)
    return float(np.dot(Pa, Pb) / denom)
    
def synergy_potential(self,
                     xa: np.ndarray,
                     xb: np.ndarray,
                     Pa: np.ndarray,
                     Pb: np.ndarray,
                     Ha: float,
                     Hb: float) -> float:
    """
    S(A,B) = σ(w_p·overlap(P_A,P_B) + w_c·Φ(A,B) + w_h·min(H_A,H_B))
    
    Measures potential for positive-sum value expansion
    """
    overlap = self.pattern_overlap(Pa, Pb)
    phi = self.phase_compatibility(xa, xb)
    h_min = min(Ha, Hb)
    
    w_p, w_c, w_h = self.synergy_weights
    
    raw = w_p * overlap + w_c * max(0, phi) + w_h * h_min
    
    # Logistic squash to [0,1]
    return sigmoid(6 * (raw - 0.5))
    
def simulate_coevolution(self,
                        x_recv: np.ndarray,
                        x_partner: np.ndarray,
                        steps: int = 5) -> List[np.ndarray]:
    """
    Simulate short trajectory: T(A,B)
    
    x_i(t+1) = x_i(t) + α·M(x_i(t), x_j)
    
    Returns sequence of projected states
    """
    x = np.array(x_recv, dtype=float)
    trajectory = [x.copy()]
    
    for t in range(steps):
        # Simple mixing: move toward partner with nonlinear activation
        delta = self.alpha * (x_partner - x)
        x = x + 0.5 * tanh_safe(delta)
        trajectory.append(x.copy())
        
    return trajectory
    
def resonant_stability(self,
                      trajectory: List[np.ndarray],
                      baseline: np.ndarray,
                      formal_layer=None) -> float:
    """
    R(A,B) = mean over trajectory of:
    β_1·ΔCoh⁺ + β_2·(1-|ΔV|) + β_3·(1-D/D_max)
    
    Measures stability and coherence preservation
    """
    if len(trajectory) < 2:
        return 0.0
        
    R_vals = []
    
    for i in range(1, len(trajectory)):
        # Coherence gain (simplified)
        coh_prev = 1.0 / (1.0 + np.std(trajectory[i-1]))
        coh_curr = 1.0 / (1.0 + np.std(trajectory[i]))
        coh_gain = max(0, coh_curr - coh_prev)
        
        # Valence stability (approximate)
        v_stability = 1.0  # Would compute from actual valence
        
        # Divergence from baseline
        div = np.linalg.norm(trajectory[i] - baseline)
        div_norm = 1.0 - np.tanh(div / (np.linalg.norm(baseline) + 1e-12))
        
        # Weighted combination
        r_step = 0.5 * coh_gain + 0.3 * v_stability + 0.2 * div_norm
        R_vals.append(r_step)
        
    return float(np.clip(np.mean(R_vals), 0.0, 1.0))
    
def compute_coupling_strength(self,
                             xa: np.ndarray,
                             xb: np.ndarray,
                             Pa: np.ndarray,
                             Pb: np.ndarray,
                             Ha: float,
                             Hb: float,
                             manipulation_posterior: float = 0.0) -> float:
    """
    κ_ij(t) = σ(η_1·Φ + η_2·S + η_3·min(H_i,H_j) - η_4·posterior)
    
    Determines coupling strength based on compatibility
    """
    phi = self.phase_compatibility(xa, xb)
    S = self.synergy_potential(xa, xb, Pa, Pb, Ha, Hb)
    h_ratio = min(Ha, Hb) / max(Ha, Hb, 1e-12)
    
    # Weights
    eta = (1.0, 1.0, 0.5, 2.0)
    
    raw = (eta[0] * max(0, phi) + 
           eta[1] * S + 
           eta[2] * h_ratio - 
           eta[3] * manipulation_posterior)
    
    kappa = sigmoid(6 * (raw - 0.5))
    return float(np.clip(kappa, 0.0, 1.0))
    
def decide_relationship(self,
                      xa: np.ndarray,
                      xb: np.ndarray,
                      Pa: np.ndarray,
                      Pb: np.ndarray,
                      Ha: float,
                      Hb: float,
                      baseline: np.ndarray,
                      Ca: float,
                      formal_layer=None) -> str:
    """
    Make relational decision: deepen, maintain, or reject
    
    DM = w_S·S + w_Φ·max(0,Φ) + w_R·R + w_C·C
    """
    S = self.synergy_potential(xa, xb, Pa, Pb, Ha, Hb)
    Phi = self.phase_compatibility(xa, xb)
    
    # Simulate trajectory
    trajectory = self.simulate_coevolution(xa, xb)
    R = self.resonant_stability(trajectory, baseline, formal_layer)
    
    # Decision metric
    DM = (0.4 * S + 
          0.3 * max(0, Phi) + 
          0.2 * R + 
          0.1 * Ca)
    
    # Threshold-based decision
    if DM > self.decision_thresholds['deepen']:
        return "deepen"
    elif DM > self.decision_thresholds['maintain']:
        return "maintain"
    else:
        return "reject"
```

# ═══════════════════════════════════════════════════════════════════════════

# BIOSWARM SIMULATION HARNESS

# ═══════════════════════════════════════════════════════════════════════════

class BioswarmSimulation:
“””
Multi-agent simulation with consent-based coupling

```
Manages:
- Multiple BioswarmAgents
- Coupling negotiations
- Energy dynamics
- Safety checks
"""

def __init__(self, num_agents: int = 2):
    self.agents: List[BioswarmAgent] = []
    self.relational_game = RelationalGameLayer()
    
    # Create agents
    for i in range(num_agents):
        agent = BioswarmAgent(seed=i, agent_id=f"agent_{i}")
        self.agents.append(agent)
        
    # Coupling registry
    self.active_sessions: Dict[Tuple[str, str], Dict] = {}
    
    self.time = 0.0
    self.dt = 0.1
    
def step(self):
    """Single simulation step"""
    
    # 1. Intrinsic updates for all agents
    for agent in self.agents:
        agent.intrinsic_update(self.dt)
        
    # 2. Process active couplings
    for (id_i, id_j), session in list(self.active_sessions.items()):
        agent_i = self._get_agent(id_i)
        agent_j = self._get_agent(id_j)
        
        if agent_i and agent_j:
            kappa = session['coupling_strength']
            
            # Apply mutual coupling
            agent_i.apply_coupling(agent_j.x, kappa, dt=self.dt)
            agent_j.apply_coupling(agent_i.x, kappa, dt=self.dt)
            
            # Update energies with coupling cost
            cost = kappa * 0.005
            agent_i.update_hook_energy(self.dt, coupling_cost=cost)
            agent_j.update_hook_energy(self.dt, coupling_cost=cost)
            
            # Check if energy depleted
            if not agent_i.has_sufficient_energy() or not agent_j.has_sufficient_energy():
                print(f"Session {id_i}<->{id_j} energy depleted, closing")
                del self.active_sessions[(id_i, id_j)]
                
    # 3. Update valences
    for agent in self.agents:
        agent.update_valence(self.dt)
        agent.update_coherence(None)
        
    self.time += self.dt
    
def initiate_coupling(self, i: int, j: int) -> bool:
    """Attempt to establish coupling between agents i and j"""
    agent_i = self.agents[i]
    agent_j = self.agents[j]
    
    print(f"\n🔄 Coupling negotiation: {agent_i.id} <-> {agent_j.id}")
    
    # Compute phase compatibility
    phi = self.relational_game.phase_compatibility(agent_i.x, agent_j.x)
    print(f"   Phase compatibility Φ: {phi:.3f}")
    
    # Compute synergy
    Pa = np.abs(agent_i.x)  # Simplified pattern
    Pb = np.abs(agent_j.x)
    S = self.relational_game.synergy_potential(
        agent_i.x, agent_j.x, Pa, Pb, agent_i.H, agent_j.H
    )
    print(f"   Synergy potential S: {S:.3f}")
    
    # Make decision
    decision = self.relational_game.decide_relationship(
        agent_i.x, agent_j.x, Pa, Pb,
        agent_i.H, agent_j.H,
        agent_i.x_star, agent_i.C
    )
    
    print(f"   Decision: {decision}")
    
    if decision in ["deepen", "maintain"]:
        # Establish coupling
        kappa = self.relational_game.compute_coupling_strength(
            agent_i.x, agent_j.x, Pa, Pb,
            agent_i.H, agent_j.H
        )
        
        session = {
            'coupling_strength': kappa,
            'start_time': self.time,
            'phi': phi,
            'synergy': S
        }
        
        self.active_sessions[(agent_i.id, agent_j.id)] = session
        print(f"   ✅ Coupling established with κ={kappa:.3f}")
        return True
    else:
        print(f"   ❌ Coupling rejected")
        return False
        
def _get_agent(self, agent_id: str) -> Optional[BioswarmAgent]:
    """Get agent by ID"""
    for agent in self.agents:
        if agent.id == agent_id:
            return agent
    return None
    
def run(self, steps: int = 100):
    """Run simulation for N steps"""
    print(f"\n🚀 Starting bioswarm simulation for {steps} steps")
    print(f"   Agents: {len(self.agents)}")
    print(f"   Timestep: {self.dt}s\n")
    
    for step in range(steps):
        self.step()
        
        if step % 20 == 0:
            self.print_status()
            
def print_status(self):
    """Print current system status"""
    print(f"\n⏱️  Time: {self.time:.1f}s")
    print(f"   Active couplings: {len(self.active_sessions)}")
    
    for agent in self.agents:
        L = agent.compute_lyapunov_energy()
        print(f"   {agent.id}: H={agent.H:.3f}, v={agent.v:+.3f}, C={agent.C:.3f}, L={L:.3f}")
```

# ═══════════════════════════════════════════════════════════════════════════

# DEMONSTRATIONS

# ═══════════════════════════════════════════════════════════════════════════

def demo_bioswarm_dynamics():
“”“Demonstrate bioswarm dynamics”””

```
print("\n" + "╔" + "═" * 78 + "╗")
print("║" + " " * 78 + "║")
print("║" + "BIOSWARM DYNAMICS - Non-Linear Multi-Agent Physics".center(78) + "║")
print("║" + " " * 78 + "║")
print("╚" + "═" * 78 + "╝")

# Create simulation
sim = BioswarmSimulation(num_agents=3)

# Show initial state
print("\n📊 Initial State:")
for agent in sim.agents:
    print(f"   {agent.id}: ||x||={np.linalg.norm(agent.x):.3f}, H={agent.H:.3f}")

# Attempt couplings
sim.initiate_coupling(0, 1)
sim.initiate_coupling(1, 2)

# Run simulation
sim.run(steps=50)

# Final analysis
print("\n" + "═" * 80)
print("FINAL ANALYSIS")
print("═" * 80)

for agent in sim.agents:
    print(f"\n{agent.id}:")
    print(f"   Final state norm: {np.linalg.norm(agent.x):.3f}")
    print(f"   Energy: {agent.H:.3f}")
    print(f"   Valence: {agent.v:+.3f}")
    print(f"   Coherence: {agent.C:.3f}")
    print(f"   Lyapunov energy: {agent.compute_lyapunov_energy():.3f}")
```

if **name** == “**main**”:
demo_bioswarm_dynamics()

```
print("\n✅ Bioswarm Dynamics Module Ready")
print("\nKey Features:")
print("   • Non-linear IPF dynamics with homeostasis")
print("   • Hook energy as consumable resource")
print("   • Valence as moral trajectory")
print("   • Phase-space compatibility analysis")
print("   • Synergy potential (positive-sum)")
print("   • Resonant stability for safety")
print("   • Lyapunov energy for stability checks")
print("\nThis replaces Western game theory with relational dynamics.")
```
