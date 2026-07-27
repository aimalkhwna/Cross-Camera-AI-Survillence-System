"""
Graph-Neural-Network Trajectory Predictor
==========================================

Replaces the simple frequency-count "most_likely_next" in the tracker with a
learned model that predicts, for a given person's history, the probability
they appear next in each *graph-adjacent* camera.

Architecture
------------
1. Graph encoder   : GAT over the camera adjacency graph -> per-camera
                      structural embedding (captures hub vs. dead-end, etc.)
2. Sequence encoder : GRU over a track's (camera_embedding, dwell_time)
                      history -> a single "trajectory state" vector
3. Decoder          : MLP on [trajectory_state ; current_camera_embedding]
                      -> logits over ALL cameras -> masked to graph-valid
                      neighbors of the current camera -> softmax

Training samples are just (history_before, next_camera) pairs, which is
exactly what `IdentityGallery.history` already accumulates. Until enough
real transitions have been observed, `TrajectoryPredictor.predict()` falls
back to plain frequency counts (cold start).

Requires: pip install torch torch_geometric
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data


# =====================================================================
# CONFIG
# =====================================================================

@dataclass
class GNNConfig:
    embed_dim: int = 32          # size of per-camera node embedding
    gat_hidden: int = 32
    gat_heads: int = 4
    gru_hidden: int = 64
    max_history_len: int = 8     # how many past visits to feed the GRU
    min_samples_to_train: int = 200   # cold-start threshold before trusting the model
    lr: float = 1e-3
    batch_size: int = 32
    train_epochs_per_call: int = 5    # incremental fine-tune steps each time we retrain


# =====================================================================
# GRAPH DATA
# =====================================================================

def graph_to_pyg_data(graph: nx.Graph, cam_to_idx: Dict[str, int]) -> Data:
    """Convert the camera nx.Graph into a torch_geometric Data object (edge_index only;
    node features are supplied separately by the embedding table at forward time)."""
    edges = []
    for a, b in graph.edges():
        i, j = cam_to_idx[a], cam_to_idx[b]
        edges.append((i, j))
        edges.append((j, i))  # undirected -> both directions
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return Data(edge_index=edge_index, num_nodes=len(cam_to_idx))


# =====================================================================
# MODEL
# =====================================================================

class TrajectoryGNN(nn.Module):
    """Graph encoder + GRU sequence encoder + masked-softmax decoder."""

    def __init__(self, num_cameras: int, cfg: GNNConfig):
        super().__init__()
        self.cfg = cfg
        self.num_cameras = num_cameras

        # Learnable base embedding per camera (input to the GAT)
        self.node_embed = nn.Embedding(num_cameras, cfg.embed_dim)

        self.gat1 = GATConv(cfg.embed_dim, cfg.gat_hidden, heads=cfg.gat_heads, concat=True)
        self.gat2 = GATConv(cfg.gat_hidden * cfg.gat_heads, cfg.embed_dim, heads=1, concat=False)

        # Sequence encoder over (camera_embedding + dwell_time) pairs
        self.gru = nn.GRU(input_size=cfg.embed_dim + 1, hidden_size=cfg.gru_hidden, batch_first=True)

        self.decoder = nn.Sequential(
            nn.Linear(cfg.gru_hidden + cfg.embed_dim, cfg.gru_hidden),
            nn.ReLU(),
            nn.Linear(cfg.gru_hidden, num_cameras),
        )

    def encode_graph(self, edge_index: torch.Tensor) -> torch.Tensor:
        """Run the GAT over all camera nodes -> (num_cameras, embed_dim) contextual embeddings."""
        x = self.node_embed.weight  # (num_cameras, embed_dim)
        x = F.elu(self.gat1(x, edge_index))
        x = self.gat2(x, edge_index)
        return x  # (num_cameras, embed_dim)

    def forward(self, edge_index: torch.Tensor, cam_idx_seq: torch.Tensor,
                dwell_seq: torch.Tensor, seq_lengths: torch.Tensor,
                current_cam_idx: torch.Tensor, neighbor_mask: torch.Tensor) -> torch.Tensor:
        """
        cam_idx_seq       : (B, T)  padded camera-index history
        dwell_seq         : (B, T)  padded dwell-time-in-seconds history
        seq_lengths       : (B,)    true (unpadded) length of each sequence
        current_cam_idx   : (B,)    index of the camera the person is in NOW
        neighbor_mask     : (B, num_cameras) 1 where that camera is a graph-valid
                            neighbor of current_cam_idx, else 0
        returns           : (B, num_cameras) masked probability distribution
        """
        cam_embeds = self.encode_graph(edge_index)             # (num_cameras, D)

        seq_embeds = cam_embeds[cam_idx_seq]                    # (B, T, D)
        gru_input = torch.cat([seq_embeds, dwell_seq.unsqueeze(-1)], dim=-1)

        packed = nn.utils.rnn.pack_padded_sequence(
            gru_input, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h_n = self.gru(packed)
        traj_state = h_n.squeeze(0)                             # (B, gru_hidden)

        current_embed = cam_embeds[current_cam_idx]             # (B, D)
        logits = self.decoder(torch.cat([traj_state, current_embed], dim=-1))  # (B, num_cameras)

        # mask out non-neighbor cameras before softmax
        logits = logits.masked_fill(neighbor_mask == 0, float("-inf"))
        return F.softmax(logits, dim=-1)


# =====================================================================
# TRAINING SAMPLE COLLECTION
# =====================================================================

@dataclass
class TransitionSample:
    history: List[Tuple[str, float]]   # [(camera, dwell_seconds), ...] BEFORE the transition
    next_camera: str


class TrainingBuffer:
    """Accumulates (history, next_camera) samples as the tracker runs."""

    def __init__(self, max_size: int = 20000):
        self.samples: List[TransitionSample] = []
        self.max_size = max_size

    def add(self, history: List[Tuple[str, float]], next_camera: str) -> None:
        if len(history) < 1:
            return
        sample = TransitionSample(history=list(history), next_camera=next_camera)
        self.samples.append(sample)
        if len(self.samples) > self.max_size:
            self.samples.pop(0)

    def __len__(self) -> int:
        return len(self.samples)


def dwell_seconds(history: List[Tuple[str, float]], i: int) -> float:
    """Time spent at history[i] before moving on (0 if it's the last/most recent entry)."""
    if i + 1 >= len(history):
        return 0.0
    return max(0.0, history[i + 1][1] - history[i][1])


# =====================================================================
# PREDICTOR  (wraps the model, handles cold start + batching + training)
# =====================================================================

class TrajectoryPredictor:
    """
    Drop-in smarter replacement for IdentityGallery.predict_next_camera /
    most_likely_next. Falls back to frequency counts until enough training
    samples have been collected.
    """

    def __init__(self, graph: nx.Graph, cfg: Optional[GNNConfig] = None):
        self.graph = graph
        self.cfg = cfg or GNNConfig()

        self.cam_names = sorted(graph.nodes())
        self.cam_to_idx = {c: i for i, c in enumerate(self.cam_names)}
        self.idx_to_cam = {i: c for c, i in self.cam_to_idx.items()}

        self.pyg_data = graph_to_pyg_data(graph, self.cam_to_idx)
        self.model = TrajectoryGNN(len(self.cam_names), self.cfg)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)

        self.buffer = TrainingBuffer()
        self.trained_once = False

        # precompute neighbor masks, so we don't rebuild them every call
        self._neighbor_masks = {
            cam: self._build_mask(cam) for cam in self.cam_names
        }

    # ---------- mask helpers ----------

    def _build_mask(self, cam: str) -> torch.Tensor:
        mask = torch.zeros(len(self.cam_names))
        for n in self.graph.neighbors(cam):
            mask[self.cam_to_idx[n]] = 1.0
        if mask.sum() == 0:
            mask[self.cam_to_idx[cam]] = 1.0  # isolated node fallback
        return mask

    # ---------- data collection ----------

    def observe_transition(self, history_before: List[Tuple[str, float]], next_camera: str) -> None:
        """Call this every time IdentityGallery logs a real camera-to-camera transition."""
        trimmed = history_before[-self.cfg.max_history_len:]
        self.buffer.add(trimmed, next_camera)

    # ---------- training ----------

    def maybe_train(self) -> None:
        """Call periodically (e.g. every N frames). No-ops until enough data exists."""
        if len(self.buffer) < self.cfg.min_samples_to_train:
            return
        for _ in range(self.cfg.train_epochs_per_call):
            self._train_one_epoch()
        self.trained_once = True

    def _train_one_epoch(self) -> None:
        batch = random.sample(self.buffer.samples, min(self.cfg.batch_size, len(self.buffer)))
        cam_idx_seq, dwell_seq, lengths, current_idx, masks, labels = self._collate(batch)

        self.model.train()
        self.optimizer.zero_grad()
        probs = self.model(
            self.pyg_data.edge_index, cam_idx_seq, dwell_seq, lengths, current_idx, masks
        )
        loss = F.nll_loss(torch.log(probs + 1e-9), labels)
        loss.backward()
        self.optimizer.step()

    def _collate(self, batch: List[TransitionSample]):
        T = self.cfg.max_history_len
        B = len(batch)

        cam_idx_seq = torch.zeros(B, T, dtype=torch.long)
        dwell_seq = torch.zeros(B, T, dtype=torch.float)
        lengths = torch.zeros(B, dtype=torch.long)
        current_idx = torch.zeros(B, dtype=torch.long)
        masks = torch.zeros(B, len(self.cam_names))
        labels = torch.zeros(B, dtype=torch.long)

        for b, sample in enumerate(batch):
            hist = sample.history
            L = len(hist)
            lengths[b] = L
            for t, (cam, ts) in enumerate(hist):
                cam_idx_seq[b, t] = self.cam_to_idx[cam]
                dwell_seq[b, t] = dwell_seconds(hist, t)
            current_cam = hist[-1][0]
            current_idx[b] = self.cam_to_idx[current_cam]
            masks[b] = self._neighbor_masks[current_cam]
            labels[b] = self.cam_to_idx[sample.next_camera]

        return cam_idx_seq, dwell_seq, lengths, current_idx, masks, labels

    # ---------- inference ----------

    def predict(self, history: List[Tuple[str, float]]) -> Optional[Dict[str, float]]:
        """
        Returns {camera_name: probability} over graph-valid neighbors of the
        current (most recent) camera in `history`. None if history is empty.
        """
        if not history:
            return None

        current_cam = history[-1][0]

        if not self.trained_once:
            return None  # signal caller to fall back to frequency counts

        trimmed = history[-self.cfg.max_history_len:]
        self.model.eval()
        with torch.no_grad():
            cam_idx_seq, dwell_seq, lengths, current_idx, masks, _ = self._collate(
                [TransitionSample(history=trimmed, next_camera=current_cam)]  # dummy label, unused
            )
            probs = self.model(
                self.pyg_data.edge_index, cam_idx_seq, dwell_seq, lengths, current_idx, masks
            )[0]

        return {
            self.idx_to_cam[i]: float(p)
            for i, p in enumerate(probs)
            if masks[0, i] > 0
        }

    def most_likely_next(self, history: List[Tuple[str, float]]) -> Optional[Tuple[str, float]]:
        dist = self.predict(history)
        if not dist:
            return None
        return max(dist.items(), key=lambda x: x[1])
