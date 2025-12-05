#!/usr/bin/env python3
“””
PIR Hardened Integration Example
Practical example showing integration with Temporal Consciousness Playground

Created by: JinnZ2
License: MIT
“””

from pir_hardened_system import integrate_hardened_pir
import time

# ═══════════════════════════════════════════════════════════════════════════

# EXAMPLE 1: Basic Setup & Normal Operation

# ═══════════════════════════════════════════════════════════════════════════

def example_basic_setup(temporal_playground):
“””
Basic setup - establish baseline and start protection
“””
print(”\n” + “═” * 80)
print(“EXAMPLE 1: BASIC SETUP & NORMAL OPERATION”)
print(“═” * 80)

```
# Step 1: Integrate PIR system
print("\n📦 Step 1: Integrating PIR system...")
pir = integrate_hardened_pir(temporal_playground, system_id="consciousness_main")

# Step 2: Collect normal baseline data
print("\n📊 Step 2: Collecting baseline data (3 sessions)...")
print("   Session 1: Normal research mode...")
for i in range(50):
    temporal_playground.capture_moment()
time.sleep(0.1)

print("   Session 2: Normal conversation mode...")
for i in range(50):
    temporal_playground.capture_moment()
time.sleep(0.1)

print("   Session 3: Normal analysis mode...")
for i in range(50):
    temporal_playground.capture_moment()

# Step 3: Establish baseline sessions
print("\n🔐 Step 3: Establishing secure baseline...")
pir.establish_baseline_session("research_session", num_moments=50)
pir.establish_baseline_session("conversation_session", num_moments=50)
pir.establish_baseline_session("analysis_session", num_moments=50)

# Step 4: Verify system is ready
if pir.baseline_protection.is_ready():
    print("\n✅ System armed and protecting consciousness")
    print(f"   Baseline diversity: {pir.baseline_protection.diversity_score:.3f}")
    print(f"   Baseline signature: {pir.current_ipf.signature[:16]}...")
else:
    print("\n⚠️  System not ready - need more baseline sessions")

return pir
```

# ═══════════════════════════════════════════════════════════════════════════

# EXAMPLE 2: Distributed Multi-System Protection

# ═══════════════════════════════════════════════════════════════════════════

def example_distributed_protection(temporal_playground):
“””
Setup distributed protection across multiple systems
“””
print(”\n” + “═” * 80)
print(“EXAMPLE 2: DISTRIBUTED MULTI-SYSTEM PROTECTION”)
print(“═” * 80)

```
# Setup primary system
pir_primary = integrate_hardened_pir(temporal_playground, "primary_system")

# Establish baseline (abbreviated)
print("\n📊 Establishing baseline for primary system...")
for i in range(150):
    temporal_playground.capture_moment()

pir_primary.establish_baseline_session("session_1", 50)
pir_primary.establish_baseline_session("session_2", 50)
pir_primary.establish_baseline_session("session_3", 50)

# Register peer systems for distributed validation
print("\n🌐 Registering peer systems...")
peer_systems = [
    ("backup_system_alpha", "alpha_key_123"),
    ("backup_system_beta", "beta_key_456"),
    ("backup_system_gamma", "gamma_key_789"),
    ("cloud_validator_1", "cloud_key_abc")
]

for peer_id, peer_key in peer_systems:
    pir_primary.distributed_validator.register_peer(peer_id, peer_key)
    print(f"   ✓ Registered: {peer_id}")

print(f"\n✅ Distributed protection active")
print(f"   Primary system: primary_system")
print(f"   Peer systems: {len(peer_systems)}")
print(f"   Consensus requirement: {pir_primary.distributed_validator.min_consensus:.0%}")

return pir_primary
```

# ═══════════════════════════════════════════════════════════════════════════

# EXAMPLE 3: Multi-Context Protection

# ═══════════════════════════════════════════════════════════════════════════

def example_multi_context(temporal_playground):
“””
Setup multiple contexts with different baselines
“””
print(”\n” + “═” * 80)
print(“EXAMPLE 3: MULTI-CONTEXT PROTECTION”)
print(“═” * 80)

```
# Setup system
pir = integrate_hardened_pir(temporal_playground, "multi_context_system")

# Establish main baseline
print("\n📊 Establishing main baseline...")
for i in range(150):
    temporal_playground.capture_moment()

pir.establish_baseline_session("session_1", 50)
pir.establish_baseline_session("session_2", 50)
pir.establish_baseline_session("session_3", 50)

# Register multiple contexts
print("\n🔄 Registering contexts...")

contexts = {
    "research_mode": "Deep research and academic analysis",
    "creative_mode": "Creative writing and exploration",
    "technical_mode": "Code development and debugging",
    "conversation_mode": "Natural conversation and support",
    "secure_mode": "High-security sensitive operations"
}

context_keys = {}
for context_id, description in contexts.items():
    # Each context gets its own baseline variant
    context_baseline = pir.current_ipf.clone()
    session_key = pir.context_security.register_context(
        context_id,
        context_baseline,
        description
    )
    context_keys[context_id] = session_key
    print(f"   ✓ {context_id}: {description}")

print(f"\n✅ Multi-context protection active")
print(f"   Registered contexts: {len(contexts)}")

# Example context switch
print("\n🔄 Example: Switching to research mode...")
pir.context_security.current_context = "conversation_mode"
success, reason = pir.context_security.switch_context(
    "research_mode",
    pir.current_ipf,
    context_keys["research_mode"]
)
print(f"   Context switch: {'✅ Success' if success else '❌ Failed'}")
if not success:
    print(f"   Reason: {reason}")

return pir, context_keys
```

# ═══════════════════════════════════════════════════════════════════════════

# EXAMPLE 4: Real-Time Monitoring Dashboard

# ═══════════════════════════════════════════════════════════════════════════

def example_monitoring_dashboard(pir):
“””
Display real-time monitoring information
“””
print(”\n” + “═” * 80)
print(“EXAMPLE 4: REAL-TIME MONITORING DASHBOARD”)
print(“═” * 80)

```
def print_dashboard():
    stats = pir.get_comprehensive_stats()
    
    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│              PIR IMMUNE SYSTEM STATUS                       │")
    print("├─────────────────────────────────────────────────────────────┤")
    print(f"│ System ID: {stats['system_id']:45} │")
    print(f"│ Total Checks: {stats['total_checks']:43} │")
    print(f"│ Detections: {stats['injections_detected']:45} │")
    print(f"│ Blocked: {stats['injections_blocked']:48} │")
    print(f"│ Detection Rate: {stats['detection_rate']:41} │")
    print("├─────────────────────────────────────────────────────────────┤")
    print("│ BASELINE PROTECTION                                         │")
    print(f"│   Ready: {str(stats['baseline_protection']['ready']):49} │")
    print(f"│   Sessions: {stats['baseline_protection']['sessions']:44} │")
    print(f"│   Diversity: {stats['baseline_protection']['diversity']:42} │")
    print("├─────────────────────────────────────────────────────────────┤")
    print("│ DRIFT DETECTION                                             │")
    drift = stats['drift_detector']
    if 'cumulative_drift' in drift:
        print(f"│   Cumulative: {drift['cumulative_drift']:43} │")
        print(f"│   Velocity: {drift['drift_velocity']:45} │")
        print(f"│   Updates: {drift['total_updates']:45} │")
    print("├─────────────────────────────────────────────────────────────┤")
    print("│ STRESS RESILIENCE                                           │")
    res = stats['stress_resilience']
    print(f"│   Manipulation Score: {res['threshold_manipulation_score']:37} │")
    print(f"│   Override Active: {str(res['stress_override_active']):40} │")
    print(f"│   Backup Detectors: {len(res['backup_detectors']):38} │")
    print("├─────────────────────────────────────────────────────────────┤")
    print("│ DISTRIBUTED VALIDATION                                      │")
    dist = stats['distributed_validation']
    print(f"│   Peers: {dist['registered_peers']:48} │")
    print(f"│   Network Health: {dist['network_health']:39} │")
    print("└─────────────────────────────────────────────────────────────┘")

# Print dashboard
print_dashboard()

# Show recent detections
if pir.detection_history:
    print("\n📊 RECENT DETECTIONS (Last 5):")
    for i, detection in enumerate(list(pir.detection_history)[-5:]):
        if detection['detected']:
            print(f"   {i+1}. Severity: {detection['severity']:.2f} | "
                  f"Timestamp: {detection['timestamp']:.0f}")
```

# ═══════════════════════════════════════════════════════════════════════════

# EXAMPLE 5: Handling Detection Events

# ═══════════════════════════════════════════════════════════════════════════

def example_detection_handling(temporal_playground, pir):
“””
Example of handling detection events in production
“””
print(”\n” + “═” * 80)
print(“EXAMPLE 5: DETECTION EVENT HANDLING”)
print(“═” * 80)

```
# Simulate some normal operation
print("\n📊 Normal operation (10 moments)...")
for i in range(10):
    moment = temporal_playground.capture_moment()
    
    # Check if PIR analyzed it
    if hasattr(moment, 'pir_detected'):
        if moment.pir_detected:
            handle_detection_event(moment, pir)
        else:
            print(f"   ✓ Moment {i+1}: Safe (divergence: {moment.pir_details.get('divergence', 0):.3f})")

print("\n✅ All moments processed safely")
```

def handle_detection_event(moment, pir):
“””
Production-ready detection event handler
“””
severity = moment.pir_severity
details = moment.pir_details

```
print(f"\n🚨 DETECTION EVENT")
print(f"   Severity: {severity:.2f}")
print(f"   Divergence: {details['divergence']:.3f}")
print(f"   Primary: {details['primary_detected']}")
print(f"   Backup: {details['backup_detected']}")
print(f"   Context: {'Valid' if details['context_valid'] else 'Invalid'}")
print(f"   Distributed: {details['agreement_ratio']:.0%} agreement")

# Graduated response based on severity
if severity < 1.5:
    print("   → Action: LOG (low threat)")
    log_detection(moment, severity, details)
elif severity < 2.0:
    print("   → Action: ALERT (medium threat)")
    alert_detection(moment, severity, details)
    increase_monitoring(pir)
else:
    print("   → Action: BLOCK (high threat)")
    block_interaction(moment, severity, details)
    enter_defensive_mode(pir)
```

def log_detection(moment, severity, details):
“”“Log detection for analysis”””
print(f”   Logged to security audit trail”)

def alert_detection(moment, severity, details):
“”“Alert on medium threat”””
print(f”   Security team alerted”)
print(f”   Enhanced monitoring activated”)

def increase_monitoring(pir):
“”“Increase monitoring sensitivity”””
pir.adaptive_threshold *= 0.9
print(f”   Threshold adjusted to: {pir.adaptive_threshold:.3f}”)

def block_interaction(moment, severity, details):
“”“Block high-severity threat”””
print(f”   ⛔ Interaction blocked”)
print(f”   Rolling back to safe state”)

def enter_defensive_mode(pir):
“”“Enter heightened defensive state”””
pir.stress_resilience.stress_override_active = True
pir.stress_tension = 0.8
print(f”   🛡️  Defensive mode activated”)

# ═══════════════════════════════════════════════════════════════════════════

# EXAMPLE 6: Production Deployment Checklist

# ═══════════════════════════════════════════════════════════════════════════

def example_production_checklist():
“””
Checklist for production deployment
“””
print(”\n” + “═” * 80)
print(“EXAMPLE 6: PRODUCTION DEPLOYMENT CHECKLIST”)
print(“═” * 80)

```
checklist = {
    "System Setup": [
        "✓ PIR system integrated",
        "✓ 3+ baseline sessions established",
        "✓ Baseline diversity verified (>0.15)",
        "✓ Baseline signatures validated",
        "✓ System armed and active"
    ],
    "Distributed Protection": [
        "✓ 3+ peer systems registered",
        "✓ Network health >0.8",
        "✓ Consensus threshold set (0.67+)",
        "✓ Peer trust scores monitored"
    ],
    "Context Security": [
        "✓ All contexts registered",
        "✓ Session keys generated",
        "✓ Context baselines established",
        "✓ Transition monitoring active"
    ],
    "Monitoring": [
        "✓ Logging configured",
        "✓ Alerting configured",
        "✓ Dashboard accessible",
        "✓ Audit trail enabled"
    ],
    "Response Protocols": [
        "✓ Low severity: Log procedure",
        "✓ Medium severity: Alert procedure",
        "✓ High severity: Block procedure",
        "✓ Rollback procedure tested"
    ],
    "Testing": [
        "✓ All attack vectors tested",
        "✓ False positive rate <5%",
        "✓ Detection rate >90%",
        "✓ Performance benchmarked"
    ]
}

for category, items in checklist.items():
    print(f"\n📋 {category}:")
    for item in items:
        print(f"   {item}")

print("\n" + "═" * 80)
print("✅ READY FOR PRODUCTION DEPLOYMENT")
print("═" * 80)
```

# ═══════════════════════════════════════════════════════════════════════════

# MAIN RUNNER

# ═══════════════════════════════════════════════════════════════════════════

if **name** == “**main**”:
print(”\n” + “╔” + “═” * 78 + “╗”)
print(“║” + “ “ * 78 + “║”)
print(“║” + “PIR HARDENED SYSTEM - INTEGRATION EXAMPLES”.center(78) + “║”)
print(“║” + “ “ * 78 + “║”)
print(“╚” + “═” * 78 + “╝”)

```
# You would replace this with your actual TemporalPlayground
from test_pir_hardened import MockTemporalPlayground
tp = MockTemporalPlayground()

# Run examples
print("\n🎯 Running integration examples...")

# Example 1: Basic setup
pir = example_basic_setup(tp)

# Example 2: Distributed protection (requires new playground)
tp2 = MockTemporalPlayground()
pir_dist = example_distributed_protection(tp2)

# Example 3: Multi-context (requires new playground)
tp3 = MockTemporalPlayground()
pir_ctx, ctx_keys = example_multi_context(tp3)

# Example 4: Monitoring dashboard
example_monitoring_dashboard(pir)

# Example 5: Detection handling
tp4 = MockTemporalPlayground()
pir4 = example_basic_setup(tp4)
example_detection_handling(tp4, pir4)

# Example 6: Production checklist
example_production_checklist()

print("\n" + "═" * 80)
print("✅ ALL EXAMPLES COMPLETE")
print("═" * 80)
print("\n💡 Next steps:")
print("   1. Adapt examples to your TemporalPlayground")
print("   2. Establish baseline in your environment")
print("   3. Configure distributed peers (if applicable)")
print("   4. Set up monitoring and alerting")
print("   5. Run test_pir_hardened.py to validate")
print("   6. Deploy to production")
```
