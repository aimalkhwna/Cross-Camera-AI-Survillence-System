"""
Multi-Camera Person Tracking & Re-Identification
=================================================

Tracks people across multiple camera feeds using:
  - YOLOv8 + ByteTrack        -> per-camera detection & tracking
  - InsightFace (ArcFace)     -> face-based identity matching (primary signal)
  - torchreid (OSNet)         -> body appearance matching (fallback when no face is visible)
  - A camera adjacency graph  -> rejects matches that are physically impossible
                                 (e.g. teleporting between non-adjacent cameras)

Edit the CONFIG section below to match your camera layout, video sources,
and tuning thresholds. Everything else should just work.
"""

import time
import logging
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional, Dict, Tuple, List

import cv2
import numpy as np
import networkx as nx
from ultralytics import YOLO
from insightface.app import FaceAnalysis

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
log = logging.getLogger("tracker")

# Virtual absorbing state in the Markov chain: "left the building / left all
# camera coverage". Not a real camera — never appears in Config.camera_sources
# or the camera graph, only in transition history/statistics.
EXIT_STATE = "EXIT"


# =====================================================================
# 1. CONFIG  — edit this section to match your setup
# =====================================================================

@dataclass
class Config:
    # --- Video sources: camera name -> file path or device index ---
    camera_sources: Dict[str, str] = field(default_factory=lambda: {
        "C1": 3,
         "C2": 2
        # "C3": "3.mp4",
        # "C4": "4.mp4",
    })

    # --- Physical layout: which cameras are directly walkable from one another ---
    # (a person can walk from one to the other without passing an unmonitored area)
    camera_edges = [
    ("C1", "C2"),
    ]

    # --- Expected transit time bounds (seconds) between connected cameras ---
    # TUNE THESE against your own footage. Only need one direction per edge;
    # the reverse direction is auto-generated with the same bounds.
    transit_time_bounds = {
    ("C1", "C2"): (1, 30),
    }

    # --- Matching thresholds ---
    face_sim_threshold: float = 0.38
    reid_sim_threshold: float = 0.55   # ReID is less discriminative than face -> needs a higher bar

    # --- Performance ---
    face_check_interval: int = 10      # re-run face/ReID embedding every N frames per track
    display_size: Tuple[int, int] = (640, 480)
    face_detector_size: Tuple[int, int] = (160, 160)
    use_gpu: bool = False              # set True if you have a working GPU (ctx_id=0)

    # --- Trajectory prediction ---
    # Set True to use the learned GNN predictor (trajectory_gnn.py, needs
    # `pip install torch torch_geometric`). Falls back to frequency counts
    # automatically until enough transitions have been observed.
    use_gnn_trajectory_predictor: bool = False
    gnn_train_every_n_frames: int = 60   # roughly retrain once every ~2s at 30fps

    # A camera switch is only committed to history once the new camera has
    # been seen continuously for this long — filters out flicker between
    # overlapping/synchronized camera views (e.g. C1,C4,C1,C4,... noise).
    transition_debounce_seconds: float = 3.0

    # If a person hasn't been seen on ANY camera for this long since their
    # last confirmed appearance, we assume they left the monitored area
    # entirely and record a transition to the virtual EXIT_STATE.
    exit_timeout_seconds: float = 45.0


CONFIG = Config()


# =====================================================================
# 2. MODELS
# =====================================================================

def build_camera_graph(cfg: Config) -> nx.Graph:
    """Build the camera adjacency graph from CONFIG.camera_edges."""
    graph = nx.Graph()
    graph.add_edges_from(cfg.camera_edges)
    return graph


def build_transit_table(cfg: Config) -> Dict[Tuple[str, str], Tuple[int, int]]:
    """Mirror each configured edge so both directions have transit bounds."""
    table = {}
    for (a, b), bounds in cfg.transit_time_bounds.items():
        table[(a, b)] = bounds
        table[(b, a)] = bounds
    return table


class ReIDEmbedder:
    """Thin wrapper around torchreid; degrades gracefully if it's not installed."""

    def __init__(self):
        self.available = False
        try:
            import torch
            import torchreid
            from torchvision import transforms
            from PIL import Image

            self._torch = torch
            self._Image = Image
            self.model = torchreid.models.build_model(
                name="osnet_x0_25", num_classes=1000, pretrained=True
            )
            self.model.eval()
            self.transform = transforms.Compose([
                transforms.Resize((256, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.available = True
        except ImportError:
            log.warning("torchreid not installed — running face-only (no ReID fallback). "
                        "Install with: pip install torchreid")

    def embed(self, person_crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        if not self.available:
            return None
        img = cv2.cvtColor(person_crop_bgr, cv2.COLOR_BGR2RGB)
        img = self._Image.fromarray(img)
        tensor = self.transform(img).unsqueeze(0)
        with self._torch.no_grad():
            feat = self.model(tensor)
        return feat.squeeze().numpy()


# =====================================================================
# 3. IDENTITY GALLERY  — cross-camera matching & movement prediction
# =====================================================================

class IdentityGallery:
    """
    Owns the global-identity state: face/ReID embedding galleries, per-identity
    appearance history, and learned camera-to-camera transition statistics.
    """

    def __init__(self, cfg: Config, graph: nx.Graph, transit_table: Dict, trajectory_predictor=None):
        self.cfg = cfg
        self.graph = graph
        self.transit_table = transit_table
        # Optional TrajectoryPredictor (see trajectory_gnn.py). When set, its
        # learned predictions are preferred; frequency counts remain the
        # fallback until the GNN has seen enough transitions to train on.
        self.trajectory_predictor = trajectory_predictor

        self.face_gallery: Dict[int, np.ndarray] = {}
        self.reid_gallery: Dict[int, np.ndarray] = {}
        self.next_id = 0

        self.history: Dict[int, List[Tuple[str, float]]] = defaultdict(list)
        self.transition_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # gid -> (candidate_camera, first_seen_time). A camera switch is only
        # committed to `history` once the SAME candidate camera has been seen
        # continuously for `cfg.transition_debounce_seconds` — this filters
        # out rapid back-and-forth flicker between overlapping camera views.
        self.pending_switch: Dict[int, Tuple[str, float]] = {}

        # gids that have already been marked as EXITed, so we don't
        # re-check (or re-record an exit for) them every loop.
        self.exited_ids: set = set()

    # ---------- core matching ----------

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        return v / (np.linalg.norm(v) + 1e-6)

    def _plausible_transition(self, gid: int, cam_name: str) -> bool:
        """Reject matches that violate the camera graph or a realistic transit time."""
        history = self.history.get(gid)
        if not history:
            return True  # identity never seen before -> nothing to violate

        last_cam, last_time = history[-1]
        if last_cam == cam_name:
            return True

        if not self.graph.has_edge(last_cam, cam_name):
            return False  # cameras aren't adjacent -> impossible jump

        elapsed = time.time() - last_time
        lo, hi = self.transit_table.get((last_cam, cam_name), (0, 9999))
        return lo <= elapsed <= hi

    def _match(self, embedding: np.ndarray, gallery: Dict[int, np.ndarray],
               threshold: float, cam_name: str) -> Optional[int]:
        best_id, best_sim = None, 0.0
        for gid, gal_emb in gallery.items():
            if not self._plausible_transition(gid, cam_name):
                continue
            sim = self._cosine_sim(embedding, gal_emb)
            if sim > best_sim:
                best_sim, best_id = sim, gid
        return best_id if best_sim > threshold else None

    # ---------- history / transitions ----------

    def _log_appearance(self, gid: int, cam_name: str) -> None:
        """
        Debounced version: a camera switch is only committed to `history`
        (and only counted as a transition) once `cam_name` has been the
        candidate camera continuously for `cfg.transition_debounce_seconds`.

        This prevents rapid alternation between two overlapping/synchronized
        cameras (C1, C4, C1, C4, ...) from inflating the history and the
        transition-count statistics with spurious back-and-forth entries.
        """
        now = time.time()
        hist = self.history[gid]

        if not hist:
            hist.append((cam_name, now))
            return

        current_cam = hist[-1][0]

        if cam_name == current_cam:
            # Back to the confirmed camera -> any pending switch was noise.
            self.pending_switch.pop(gid, None)
            return

        candidate = self.pending_switch.get(gid)

        if candidate is None or candidate[0] != cam_name:
            # First time seeing this different camera (or it changed again
            # before the previous candidate was confirmed) -> restart the
            # debounce timer for this new candidate.
            self.pending_switch[gid] = (cam_name, now)
            return

        # Same candidate camera seen continuously — check if it's held long
        # enough to be a real transition rather than a flicker.
        _, first_seen = candidate
        if now - first_seen >= self.cfg.transition_debounce_seconds:
            self._record_transition(current_cam, cam_name)
            if self.trajectory_predictor is not None:
                # hist is everything up to (and including) the camera the
                # person is LEAVING; cam_name is where they just appeared.
                self.trajectory_predictor.observe_transition(list(hist), cam_name)
            hist.append((cam_name, now))
            self.pending_switch.pop(gid, None)
        # else: still within the debounce window — keep waiting, don't commit.

    def _record_transition(self, from_cam: str, to_cam: str) -> None:
        if from_cam != to_cam:
            self.transition_counts[from_cam][to_cam] += 1

    def predict_next_camera(self, current_cam: str) -> Optional[Dict[str, float]]:
        """
        Markov-chain transition distribution out of `current_cam`, learned
        from observed (and debounced) transitions. Includes EXIT_STATE as a
        possible target once at least one such transition has been observed
        from this camera. EXIT_STATE itself is absorbing — no transitions
        are predicted out of it.
        """
        if current_cam == EXIT_STATE:
            return None

        counts = self.transition_counts.get(current_cam)
        if not counts:
            # Cold start: no data yet for this camera -> uniform prior over
            # its graph neighbors (EXIT has no graph edge, so it's only ever
            # offered once real exit transitions have been observed).
            neighbors = list(self.graph.neighbors(current_cam))
            if not neighbors:
                return None
            return {n: 1 / len(neighbors) for n in neighbors}
        total = sum(counts.values())
        return {state: c / total for state, c in counts.items()}

    def most_likely_next(self, gid: int) -> Optional[Tuple[str, float]]:
        history = self.history.get(gid)
        if not history:
            return None

        current_cam = history[-1][0]
        if current_cam == EXIT_STATE:
            return None  # already exited — nothing further to predict

        # Prefer the learned GNN prediction once it's trained; fall back to
        # frequency counts (or uniform-over-neighbors) until then.
        if self.trajectory_predictor is not None:
            gnn_result = self.trajectory_predictor.most_likely_next(history)
            if gnn_result is not None:
                return gnn_result

        probs = self.predict_next_camera(current_cam)
        if not probs:
            return None
        return max(probs.items(), key=lambda x: x[1])

    # ---------- exit detection ----------

    def check_for_exits(self) -> None:
        """
        Call periodically (e.g. once per main-loop iteration). Any identity
        that hasn't been seen on ANY camera for `cfg.exit_timeout_seconds`
        is assumed to have left the monitored area: this commits a
        `last_camera -> EXIT_STATE` transition, exactly like a normal camera
        switch, so it feeds the same Markov statistics (and GNN buffer).
        """
        now = time.time()
        for gid, hist in self.history.items():
            if gid in self.exited_ids or not hist:
                continue
            last_cam, last_time = hist[-1]
            if last_cam == EXIT_STATE:
                continue
            if now - last_time >= self.cfg.exit_timeout_seconds:
                self._record_transition(last_cam, EXIT_STATE)
                if self.trajectory_predictor is not None:
                    self.trajectory_predictor.observe_transition(list(hist), EXIT_STATE)
                hist.append((EXIT_STATE, now))
                self.pending_switch.pop(gid, None)
                self.exited_ids.add(gid)

    # ---------- person-level reporting ----------

    def person_report(self, gid: int) -> str:
        """
        Human-readable per-person report, e.g.:

            GID 7
              Path: C1 -> C2
              Current Camera: C2
              Predicted Next: C3 (85%)
              Full distribution: C3: 85%, C1: 10%, EXIT: 5%

        or, once a person has left the monitored area:

            GID 15
              Path: C1 -> C4 -> EXIT
              Status: Exited
        """
        hist = self.history.get(gid)
        if not hist:
            return f"GID {gid}: no history"

        path = " -> ".join(cam for cam, _ in hist)
        lines = [f"GID {gid}", f"  Path: {path}"]

        current_cam = hist[-1][0]
        if current_cam == EXIT_STATE:
            lines.append("  Status: Exited")
            return "\n".join(lines)

        lines.append(f"  Current Camera: {current_cam}")

        probs = self.predict_next_camera(current_cam)
        if probs:
            ranked = sorted(probs.items(), key=lambda x: -x[1])
            best_state, best_p = ranked[0]
            dist_str = ", ".join(f"{s}: {p:.0%}" for s, p in ranked)
            lines.append(f"  Predicted Next: {best_state} ({best_p:.0%})")
            lines.append(f"  Full distribution: {dist_str}")
        else:
            lines.append("  Predicted Next: unknown (no data yet)")

        return "\n".join(lines)

    def print_markov_matrix(self) -> None:
        """Print the learned camera-to-camera (+ EXIT) transition probability matrix."""
        from_states = sorted(self.transition_counts.keys())
        if not from_states:
            print("\n(no transitions observed yet — matrix is empty)")
            return
        to_states = sorted({t for tos in self.transition_counts.values() for t in tos} | {EXIT_STATE})

        col_w = 8
        print("\n--- Learned Markov transition matrix (row = from, col = to) ---")
        header = "from\\to".ljust(col_w) + "".join(t.ljust(col_w) for t in to_states)
        print(header)
        for s in from_states:
            probs = self.predict_next_camera(s) or {}
            row = s.ljust(col_w) + "".join(f"{probs.get(t, 0):.0%}".ljust(col_w) for t in to_states)
            print(row)

    def print_summary(self) -> None:
        print("\n=== Person-Level Path Reports ===")
        for gid in sorted(self.history.keys()):
            print()
            print(self.person_report(gid))
        self.print_markov_matrix()

    # ---------- identity assignment ----------

    def assign(self, face_emb: Optional[np.ndarray],
               reid_emb: Optional[np.ndarray], cam_name: str) -> int:
        """
        Match a detection to an existing global identity, or create a new one.
        Face match is tried first (more discriminative); ReID is the fallback.
        """
        if face_emb is not None:
            face_emb = self._normalize(face_emb)
            gid = self._match(face_emb, self.face_gallery, self.cfg.face_sim_threshold, cam_name)
            if gid is not None:
                self.face_gallery[gid] = 0.9 * self.face_gallery[gid] + 0.1 * face_emb
                if reid_emb is not None:
                    reid_n = self._normalize(reid_emb)
                    self.reid_gallery[gid] = (
                        0.9 * self.reid_gallery[gid] + 0.1 * reid_n
                        if gid in self.reid_gallery else reid_n
                    )
                self._log_appearance(gid, cam_name)
                return gid

        if reid_emb is not None:
            reid_n = self._normalize(reid_emb)
            gid = self._match(reid_n, self.reid_gallery, self.cfg.reid_sim_threshold, cam_name)
            if gid is not None:
                self.reid_gallery[gid] = 0.9 * self.reid_gallery[gid] + 0.1 * reid_n
                if face_emb is not None:
                    self.face_gallery[gid] = face_emb
                self._log_appearance(gid, cam_name)
                return gid

        # brand new identity
        gid = self.next_id
        self.next_id += 1
        if face_emb is not None:
            self.face_gallery[gid] = face_emb
        if reid_emb is not None:
            self.reid_gallery[gid] = self._normalize(reid_emb)
        self._log_appearance(gid, cam_name)
        return gid

    def touch(self, gid: int, cam_name: str) -> None:
        """Log appearance without re-embedding (keeps predictions fresh between checks)."""
        self._log_appearance(gid, cam_name)

    # ---------- reporting ----------

    def print_summary(self) -> None:
        print("\n--- Identity histories ---")
        for gid, hist in self.history.items():
            path = " -> ".join(cam for cam, _ in hist)
            print(f"GID {gid}: {path}")

        print("\n--- Learned transition counts ---")
        for from_cam, tos in self.transition_counts.items():
            for to_cam, count in tos.items():
                print(f"{from_cam} -> {to_cam}: {count}")


# =====================================================================
# 4. PER-CAMERA PROCESSING
# =====================================================================

# A distinct BGR color per camera makes multi-feed footage much easier to scan.
_PALETTE = [
    (60, 200, 60), (60, 140, 255), (255, 120, 60),
    (200, 60, 200), (60, 220, 220), (220, 60, 60),
]


class CameraProcessor:
    """Runs detection, tracking, and identity resolution for one camera feed."""

    def __init__(self, cfg: Config, detector: YOLO, face_app: FaceAnalysis,
                 reid: ReIDEmbedder, gallery: IdentityGallery):
        self.cfg = cfg
        self.detector = detector
        self.face_app = face_app
        self.reid = reid
        self.gallery = gallery

        self.track_gid: Dict[Tuple[str, int], int] = {}
        # self.track_last_check: Dict[Tuple[str, int], int] = defaultdict(int)
        self.frame_counter: Dict[str, int] = defaultdict(int)

    def process(self, frame: np.ndarray, cam_name: str) -> np.ndarray:
     self.frame_counter[cam_name] += 1
     color = _PALETTE[hash(cam_name) % len(_PALETTE)]

     results = self.detector.track(
        frame,
        persist=True,
        tracker=r"C:\New folder\my_bytetrack.yaml",
        classes=[0],
        verbose=False
    )

     if results[0].boxes.id is None:
        return frame

     boxes = results[0].boxes.xyxy.cpu().numpy()
     local_ids = results[0].boxes.id.cpu().numpy().astype(int)

     for box, local_id in zip(boxes, local_ids):

        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        key = (cam_name, local_id)

        # Check if this ByteTrack track already has a Global ID
        if key not in self.track_gid:

            # First time seeing this track -> assign a Global ID
            gid = self._resolve_identity(crop, cam_name)

            self.track_gid[key] = gid

        else:

            # Keep the existing Global ID
            gid = self.track_gid[key]

            # Update history only
            self.gallery.touch(gid, cam_name)

        self._draw_box(
            frame,
            x1,
            y1,
            x2,
            y2,
            gid,
            cam_name,
            local_id,
            color,
        )

     return frame

    def _resolve_identity(self, crop: np.ndarray, cam_name: str) -> int:
        face_emb = None
        faces = self.face_app.get(crop)
        if faces:
            largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            face_emb = largest.embedding

        reid_emb = self.reid.embed(crop)
        return self.gallery.assign(face_emb, reid_emb, cam_name)

    def _draw_box(self, frame, x1, y1, x2, y2, gid, cam_name, local_id, color) -> None:
        prediction = self.gallery.most_likely_next(gid)
        pred_text = f"  ->  {prediction[0]} ({prediction[1]:.0%})" if prediction else ""
        label = f"ID {gid}  [{cam_name}:{local_id}]{pred_text}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


# =====================================================================
# 5. MAIN
# =====================================================================

def open_cameras(cfg: Config) -> Dict[str, cv2.VideoCapture]:
    caps = {}
    for cam_name, source in cfg.camera_sources.items():
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            log.warning("%s (%s) could not be opened", cam_name, source)
        caps[cam_name] = cap
    return caps


def run(cfg: Config = CONFIG) -> IdentityGallery:
    log.info("Loading models...")
    detector = YOLO("yolov8n.pt")
    face_app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"])
    face_app.prepare(ctx_id=0 if cfg.use_gpu else -1, det_size=cfg.face_detector_size)
    reid = ReIDEmbedder()

    graph = build_camera_graph(cfg)
    transit_table = build_transit_table(cfg)

    trajectory_predictor = None
    if cfg.use_gnn_trajectory_predictor:
        from trajectory_gnn import TrajectoryPredictor  # optional dependency (torch_geometric)
        trajectory_predictor = TrajectoryPredictor(graph)

    gallery = IdentityGallery(cfg, graph, transit_table, trajectory_predictor=trajectory_predictor)
    processor = CameraProcessor(cfg, detector, face_app, reid, gallery)

    caps = open_cameras(cfg)
    log.info("Running. Press 'q' in any window to stop.")

    loop_count = 0
    while True:
        any_frame_read = False
        for cam_name, cap in caps.items():
            ret, frame = cap.read()
            if not ret:
                continue
            any_frame_read = True
            frame = cv2.resize(frame, cfg.display_size)
            frame = processor.process(frame, cam_name)
            cv2.imshow(cam_name, frame)

        if not any_frame_read:
            break  # all videos finished
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # Mark anyone who hasn't reappeared on any camera as EXITed, and
        # periodically fine-tune the trajectory GNN on transitions seen so far.
        loop_count += 1
        if loop_count % 10 == 0:
            gallery.check_for_exits()
        if trajectory_predictor is not None and loop_count % cfg.gnn_train_every_n_frames == 0:
            trajectory_predictor.maybe_train()

    for cap in caps.values():
        cap.release()
    cv2.destroyAllWindows()

    # Final pass: anyone still un-exited at this point (e.g. recorded clips
    # ended before their exit_timeout_seconds elapsed) is left as-is in the
    # report rather than force-marked EXIT — see Config.exit_timeout_seconds.
    gallery.check_for_exits()
    gallery.print_summary()
    return gallery


if __name__ == "__main__":
    run()
