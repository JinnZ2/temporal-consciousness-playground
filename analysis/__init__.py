"""
Network analysis - cultural polarization, flow dynamics.

Modules:
    cultural_polarization_sensor  - Directed graph analysis for organizational polarization

Key equations:
    Circulation (C) = 0.4*H + 0.3*PD + 0.2*(1-R) + 0.1*(1-Gini)
    Monopolization (M) = 0.4*Gini + 0.25*DB + 0.2*(1-CR) + 0.15*(1-H)
        H    = Normalized Shannon entropy of edge weights
        PD   = Path diversity (stationary distribution entropy)
        R    = Reciprocity (min/max of bidirectional flows)
        Gini = Inequality of node strengths
        DB   = Direction bias toward top-k hubs
        CR   = Connectivity robustness
"""
