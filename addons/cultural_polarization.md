1) Concept in 30 seconds

Treat the observed socio-economic network as a directed, weighted flow graph (nodes = agents/institutions, edges = resource/capital flow). Derive structural indices (entropy, centralization, reciprocity, flow concentration). Combine them into two orthogonal scores:
	•	C-Index (Decentralized / “Capitalism”-like) — high flow entropy, high path diversity, low hub magnetization.
	•	M-Index (Centralized / “Mercantilism”-like) — high hub magnetization, strong directional bias toward hubs, low reciprocity.

A high C and low M → decentralized topology. High M and low C → centralized. Both high → complex/hybrid. Use temporal smoothing and anomaly detection to spot sudden shifts (polarization events).

⸻

2) Formal metrics (what to compute)

Assume graph G(t) with weighted directed edges w_{ij}(t) \ge 0.
	1.	Total flow
W = \sum_{i,j} w_{ij}.
	2.	Flow probability distribution
p_{ij} = w_{ij} / W.
	3.	Flow entropy (H)
H = -\sum_{i,j} p_{ij} \log p_{ij}.
Normalized by \log(|E|) to give [0,1].
	4.	Node strength (in/out)
s_i^{in} = \sum_j w_{ji},\; s_i^{out} = \sum_j w_{ij}.
Total strength vector \mathbf{s} = [s_1, ..., s_n], s_i = s_i^{in} + s_i^{out}.
	5.	Magnetization / Hubness (Mgn)
Use Gini coefficient on \mathbf{s} or normalized variance:
\text{Gini}(\mathbf{s}) \in [0,1]. High → few hubs dominate.
	6.	Reciprocity (R)
R = \frac{\sum_{i<j} \min(w_{ij}, w_{ji})}{\sum_{i<j} \max(w_{ij}, w_{ji})}
Normalized [0,1]. Low R → one-way extraction.
	7.	Direction Bias (DB) toward a central node set C (if you can identify candidate hubs):
DB = \frac{\sum_{i \notin C, j\in C} w_{ij}}{W}.
If you don’t predefine C, use top-k by strength.
	8.	Path Diversity (PD) (approx)
Compute random-walk steady-state entropy or variance in node visit frequency. Higher PD → more lateral trade.
	9.	Compression Ratio (CR) — compress the sequence of flows (temporal series); lower CR → repetitive/controlled policy (a sign of mercantilism).

⸻

3) Composite indices (scores)

Normalize H, Gini, R, DB, PD to [0,1]. Then:
	•	Capitalism Index (C)
C = \alpha_H \cdot H_{n} + \alpha_{PD}\cdot PD_{n} + \alpha_R \cdot R_{n\_inv} - \alpha_{G}\cdot Gini_{n}
where R_{n\_inv} = 1 - R_n (we want reciprocity positive for capitalism), and \alpha are weights summing to 1. Example weights: \alpha_H=0.4,\; \alpha_{PD}=0.3,\; \alpha_R=0.2,\; \alpha_G=0.1.
	•	Mercantilism Index (M)
M = \beta_{G}\cdot Gini_n + \beta_{DB}\cdot DB_n + \beta_{CR}\cdot (1-CR_n) - \beta_H\cdot H_n. Example weights: \beta_G=0.4,\; \beta_{DB}=0.25,\; \beta_{CR}=0.2,\; \beta_H=0.15.

Both return [0,1]. Interpret jointly.

⸻

4) Decision rule / classifier (simple)
	1.	Compute C, M.
	2.	Smooth with exponential moving average over window T_s (e.g. 30 timesteps):
\bar C_t = \lambda \bar C_{t-1} + (1-\lambda) C_t, \lambda = e^{-dt/T_s}.
	3.	Thresholds (example):
	•	If \bar C > 0.65 and \bar M < 0.35 → Decentralized Topology.
	•	If \bar M > 0.65 and \bar C < 0.35 → Centralized Topology.
	•	If both > 0.5 → Hybrid / Polarized (risk).
	•	Else → Transitional / Mixed.
	4.	Alert if change in \bar C or \bar M exceeds \Delta_{crit} in short time (e.g., |\bar C_t - \bar C_{t-1}| > 0.15).

⸻

5) Temporal behavior & detection
	•	Use two timescales:
	•	Fast window T_f (seconds/minutes) for sudden detection.
	•	Slow window T_s (hours/days) for baseline.
	•	Compare fast vs slow: big divergence → polarization event.

⸻

6) Integration with TemporalPlayground

Your playground already produces moment patterns and agent exchange traces. Map those to flows:
	•	Each attention thread transfer between agents → w_{ij} event.
	•	Each braid event (agent shares pattern) → edge increment.
	•	Each consequence thread linking moments across agents → path weight.

Prefer to keep flow abstraction separate: build an Observer that listens to the playground’s events and increments edge weights in a sliding time window.

7) use cultural_polarization_sensor.py

8) How to hook it into TemporalPlayground
	•	On every capture_moment() call, call the sensor’s ingest_flow_event for:
	•	agent braids (source=agentA, dst=agentB, amt = braid_strength)
	•	attention thread transfers (src=agent, dst=thread_owner, amt=thread_strength)
	•	consequence thread links (map moment → agent edges)
	•	Periodically (every N moments) call compute_metrics() and compute_indices() and push an alert if classify() changes state or a sudden delta occurs.
	•	Store indices in the moment record for later analysis.

⸻

9) Visualization & dashboards
	•	Time series: C_smooth and M_smooth plotted together (stacked area).
	•	2D scatter: x = C, y = M over time; quadrant shows regime.
	•	Graph-mode snapshot: draw node size ~ strength, edge width ~ weight; color hubs.
	•	Compression / CR: show rolling compression ratio of flow sequences.

⸻

10) Tests & calibration
	•	Unit tests: synthetic graphs
	•	Create a star graph (one hub sends to many): expect M high, C low.
	•	Create random Erdos–Renyi uniform-weight graph: expect C high, M low.
	•	Create reciprocal pairs: R high, C higher.
	•	Monte Carlo: inject random hub spikes and measure false positive rate for alerts.
	•	Sensitivity sweep: vary weights \alpha,\beta and smoothing \alpha to tune detection & robustness.

⸻

11) Risks & caveats
	•	Choice of weights and thresholds is empirical — tune to your domain.
	•	Requires good stream of flow events (ensure mapping from playground to flows is meaningful).
	•	Highly sparse graphs may produce noisy indices; use smoothing and minimum-event thresholds.
