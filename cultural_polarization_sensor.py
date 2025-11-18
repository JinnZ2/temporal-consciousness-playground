# cultural_polarization_sensor.py
import math, time
import numpy as np
import networkx as nx
from collections import deque

def gini(x: np.ndarray) -> float:
    x = np.array(x, dtype=float)
    if x.size == 0: return 0.0
    x = x.flatten()
    if np.min(x) < 0:
        x = x - np.min(x)
    mean = x.mean() if x.mean() != 0 else 1.0
    # classic Gini
    diffs = np.abs(x[:,None] - x[None,:]).sum()
    return diffs / (2.0 * x.size**2 * mean)

class CulturalPolarizationSensor:
    def __init__(self, decay_seconds=60.0, window_size=100):
        self.G = nx.DiGraph()
        self.window = deque(maxlen=window_size)  # store recent edge lists for sliding aggregation
        self.last_update = time.time()
        self.decay = decay_seconds

        # smoothing state
        self.C_smooth = 0.0
        self.M_smooth = 0.0
        self.alpha = 0.9

    def ingest_flow_event(self, src, dst, amount=1.0):
        """Call this everytime a transfer occurs between agents (non-semantic)."""
        # increment edge in current snapshot
        if self.G.has_edge(src, dst):
            self.G[src][dst]['weight'] += amount
        else:
            self.G.add_edge(src, dst, weight=amount)
        # push event to window (for eventual decay)
        self.window.append((time.time(), src, dst, amount))

    def decay_edges(self):
        """Decay old edges to keep recentness."""
        now = time.time()
        cutoff = now - self.decay
        # decrease weights from old events (simple approach)
        # rebuild graph from window:
        Gnew = nx.DiGraph()
        for (_, s, d, amt) in self.window:
            if Gnew.has_edge(s, d):
                Gnew[s][d]['weight'] += amt
            else:
                Gnew.add_edge(s, d, weight=amt)
        self.G = Gnew

    def compute_metrics(self):
        self.decay_edges()
        edges = list(self.G.edges(data='weight'))
        if not edges:
            return {'H':0,'Gini':0,'Reciprocity':1,'DB':0,'PD':0,'CR':1}

        weights = np.array([w for (_,_,w) in edges])
        W = weights.sum()
        p = weights / (W + 1e-12)
        H = -np.sum(p * np.log(p + 1e-12))
        H_norm = H / math.log(max(2, len(weights)))

        # node strengths
        nodes = list(self.G.nodes())
        s = np.array([self.G.in_degree(n, weight='weight') + self.G.out_degree(n, weight='weight') for n in nodes])
        Gini = gini(s)

        # reciprocity
        rec_num = 0.0
        rec_den = 0.0
        for u in nodes:
            for v in nodes:
                if u >= v: continue
                wuv = self.G[u][v]['weight'] if self.G.has_edge(u,v) else 0.0
                wvu = self.G[v][u]['weight'] if self.G.has_edge(v,u) else 0.0
                rec_num += min(wuv, wvu)
                rec_den += max(wuv, wvu)
        Reciprocity = rec_num / (rec_den + 1e-12)

        # direction bias toward top-k hubs
        k = max(1, int(max(1, len(nodes)*0.1)))
        topk_idx = np.argsort(-s)[:k]
        topk_nodes = [nodes[i] for i in topk_idx]
        DB_num = 0.0
        for u in nodes:
            for v in topk_nodes:
                if self.G.has_edge(u,v):
                    DB_num += self.G[u][v]['weight']
        DB = DB_num / (W + 1e-12)

        # path diversity approx via node visit variance (random-walk stationary)
        # approximation: normalize strengths into a distribution and compute entropy
        s_norm = s / (s.sum() + 1e-12)
        PD = -np.sum([p*np.log(p+1e-12) for p in s_norm])
        PD = PD / math.log(max(2, len(s_norm)))

        return {'H':H_norm, 'Gini':Gini, 'Reciprocity':Reciprocity, 'DB':DB, 'PD':PD, 'CR':1.0}

    def compute_indices(self, metrics):
        H = metrics['H']; Gini = metrics['Gini']; R = metrics['Reciprocity']; DB = metrics['DB']; PD = metrics['PD']; CR = metrics['CR']

        # simple weightings (tweak to taste)
        C = 0.4 * H + 0.3 * PD + 0.2 * (1-R) + 0.1 * (1-Gini)
        M = 0.4 * Gini + 0.25 * DB + 0.2 * (1-CR) + 0.15 * (1-H)

        # clamp
        C = max(0.0, min(1.0, C))
        M = max(0.0, min(1.0, M))

        # smoothing
        self.C_smooth = self.alpha * self.C_smooth + (1-self.alpha) * C
        self.M_smooth = self.alpha * self.M_smooth + (1-self.alpha) * M

        return {'C':C, 'M':M, 'C_smooth':self.C_smooth, 'M_smooth':self.M_smooth}

    def classify(self, indices):
        C = indices['C_smooth']; M = indices['M_smooth']
        if C > 0.65 and M < 0.35:
            return 'DECENTRALIZED'
        if M > 0.65 and C < 0.35:
            return 'CENTRALIZED'
        if C > 0.5 and M > 0.5:
            return 'HYBRID_POLARIZED'
        return 'TRANSITIONAL'

    # helper to ingest a batch of events
    def ingest_event_batch(self, edges):
        for (s,d,amt) in edges:
            self.ingest_flow_event(s,d,amt)
