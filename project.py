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

---------------------------------------------------------------------------
STABILITY PATCH (see chat) — what changed vs. the original and why:

1. New tracks are no longer identified from their very first (often blurry
   / partial / face-away) frame. CameraProcessor now buffers a track's
   crops for `identity_resolve_frames` frames and resolves identity from
   the BEST crop in that window (largest detected face, else largest
   body box). This is the #1 fix for "ID changes when the frame changes":
   most of that churn was bad first-frame embeddings failing to match
   the gallery and silently minting a new global ID.

2. Matching now uses a stricter threshold for candidates last seen on a
   DIFFERENT camera than for candidates last seen on the SAME camera
   (`cross_cam_face_sim_threshold` / `cross_cam_reid_sim_threshold` vs.
   the original `face_sim_threshold` / `reid_sim_threshold`, which are
   now the same-camera / recovery thresholds). This is the fix for
   "the ID is the same in all the cameras" for people who are NOT
   actually the same person — it makes it harder for two different
   people on two different cameras to falsely merge into one global ID,
   while keeping recovery of the SAME person on the SAME camera easy.

3. `face_check_interval` (previously defined but unused) now actually
   does something: every N frames, an already-resolved track's crop is
   re-embedded and folded into the gallery average, so the stored
   embedding drifts less and stays representative over a long track.
---------------------------------------------------------------------------
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger("tracker")

# Virtual absorbing state in the Markov chain: "left the building / left all
# camera coverage". Not a real camera — never appears in Config.camera_sources
# or the camera graph, only in transition history/statistics.
EXIT_STATE = "EXIT"


# =====================================================================
# 1. CONFIG — edit this section to match your setup
# =====================================================================

@dataclass
class Config:
    # --- Video sources: camera name -> file path or device index ---
    camera_sources: Dict[str, str] = field(default_factory=lambda: {
        "C1": "1.mp4",
        "C2": "2.mp4",
        "C3": "3.mp4",
        "C4": "4.mp4",
    })

    # --- Physical layout: which cameras are directly walkable from one another ---
    camera_edges = [
        ("C1", "C2"),
        ("C2", "C3"),
        ("C3", "C4"),
        ("C4", "C1"),
    ]

    # --- Expected transit time bounds (seconds) between connected cameras ---
    # Use zero as the lower bound when cameras overlap or their video
    # streams are synchronized: the same person can legitimately be seen
    # by two adjacent cameras at the same time.
    transit_time_bounds = {
        ("C1", "C2"): (0, 30),
        ("C2", "C1"): (0, 30),
        ("C2", "C3"): (0, 30),
        ("C3", "C2"): (0, 30),
        ("C3", "C4"): (0, 30),
        ("C4", "C3"): (0, 30),
        ("C4", "C1"): (0, 30),
        ("C1", "C4"): (0, 30),
    }

    # --- Matching thresholds: SAME-camera track recovery ---
    # (a track that briefly lost its ByteTrack ID and reappeared on the
    # SAME camera should be forgiven a slightly weaker match — it's very
    # likely the same person)
    face_sim_threshold: float = 0.38
    reid_sim_threshold: float = 0.55

    # --- Matching thresholds: CROSS-camera matching ---
    # deliberately stricter than the same-camera thresholds above, since a
    # false match here merges two DIFFERENT people into one global ID.
    cross_cam_face_sim_threshold: float = 0.55
    cross_cam_reid_sim_threshold: float = 0.70

    # --- "Sole candidate" handoff threshold ---
    # When the camera graph + transit-time window narrows candidates down to
    # EXACTLY ONE gid that could plausibly be walking from the adjacent
    # camera into this one right now, that positional/temporal narrowing is
    # itself strong evidence — so a much weaker appearance match is enough
    # to confirm it, rather than requiring the full cross_cam threshold.
    # This is what stops "new camera -> instant new ID" when there's really
    # only one person it could be. Set to a negative number to disable this
    # behavior entirely and fall back to the strict thresholds above only.
    handoff_sim_threshold: float = 0.20

    # --- Same-camera recovery window ---
    # If a track disappears (ByteTrack loses it) and a brand-new local track
    # appears on the SAME camera within this many seconds, and no other
    # currently-active track on that camera is already using that person's
    # global ID, treat the new track as a recovery of the person who just
    # vanished rather than requiring the strict face_sim/reid_sim thresholds
    # to be cleared. This is what stops rapid ID-splitting on a single
    # camera from momentary occlusion / detection flicker.
    recent_loss_window_seconds: float = 5.0

    # --- New-track identity resolution ---
    # Instead of identifying a track from its first frame, buffer this many
    # frames and resolve identity from the best crop seen in that window.
    identity_resolve_frames: int = 5

    # --- Performance ---
    face_check_interval: int = 10  # re-embed & refresh gallery every N frames per resolved track
    display_size: Tuple[int, int] = (640, 480)
    face_detector_size: Tuple[int, int] = (160, 160)
    use_gpu: bool = False  # set True if you have a working GPU (ctx_id=0)

    # --- Trajectory prediction ---
    use_gnn_trajectory_predictor: bool = False
    gnn_train_every_n_frames: int = 60

    # A camera switch is only committed to history once the new camera has
    # been seen continuously for this long — filters out flicker between
    # overlapping/synchronized camera views (e.g. C1,C4,C1,C4,... noise).
    transition_debounce_seconds: float = 3.0

    # If a person hasn't been seen on ANY camera for this long since their
    # last confirmed appearance, we assume they left the monitored area
    # entirely and record a transition to the virtual EXIT_STATE.
    exit_timeout_seconds: float = 45.0

    # --- Debugging ---
    # When True, every match attempt in IdentityGallery._match logs the gid
    # it compared against, the similarity score, the threshold it needed to
    # clear, and whether it passed — plus a log line whenever
    # _plausible_transition rejects a gid before scoring even happens.
    # Turn this on first when chasing an ID-splitting/merging problem —
    # don't guess at threshold values, read what the real scores are.
    debug_matching: bool = True


CONFIG = Config()


# =====================================================================
# 2. MODELS
# =====================================================================

def build_camera_graph(cfg: Config) -> nx.Graph:
    graph = nx.Graph()
    graph.add_edges_from(cfg.camera_edges)
    return graph


def build_transit_table(cfg: Config) -> Dict[Tuple[str, str], Tuple[int, int]]:
    table = {}
    for (a, b), bounds in cfg.transit_time_bounds.items():
        table[(a, b)] = bounds
        table[(b, a)] = bounds
    return table


class ReIDEmbedder:
    """Thin wrapper around torchreid; degrades gracefully if it's not installed.

    IMPORTANT: if this prints the "torchreid not installed" warning below,
    you are running FACE-ONLY. Anyone whose face isn't visible in their
    identity-resolution window (see Config.identity_resolve_frames) cannot
    be matched or recovered at all and WILL get a fresh ID every time their
    ByteTrack local track breaks. Installing torchreid (`pip install
    torchreid`) fixes this — it's very likely the single biggest lever for
    ID stability if you're seeing this warning.
    """

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
        except Exception as e:
            log.warning("ReID unavailable — running face-only (no ReID fallback). "
                        "Root cause: %s: %s", type(e).__name__, e)

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
# 3. IDENTITY GALLERY — cross-camera matching & movement prediction
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
        self.trajectory_predictor = trajectory_predictor

        self.face_gallery: Dict[int, np.ndarray] = {}
        self.reid_gallery: Dict[int, np.ndarray] = {}
        self.next_id = 0
        self.history: Dict[int, List[Tuple[str, float]]] = defaultdict(list)
        self.transition_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.pending_switch: Dict[int, Tuple[str, float]] = {}
        self.exited_ids: set = set()
        # Undebounced "where were they last actually seen" — updated on
        # EVERY appearance, unlike `history`, which only commits a camera
        # switch after `transition_debounce_seconds` of continuous
        # confirmation. Plausibility/adjacency checks need the raw, current
        # position; only the reported path/Markov stats need the debounced
        # version. Conflating the two was causing stale-camera rejections.
        self.raw_last_seen: Dict[int, Tuple[str, float]] = {}

    # ---------- core matching ----------

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        return v / (np.linalg.norm(v) + 1e-6)

    def _plausible_transition(self, gid: int, cam_name: str) -> bool:
        """Reject matches that violate the camera graph or a realistic transit time.

        Uses `raw_last_seen` (always current) rather than the debounced
        `history`, since a person can legitimately move again before their
        previous switch has been debounce-confirmed into history.
        """
        last = self.raw_last_seen.get(gid)
        if last is None:
            return True
        last_cam, last_time = last
        if last_cam == cam_name:
            return True
        if not self.graph.has_edge(last_cam, cam_name):
            if self.cfg.debug_matching:
                log.info("[match] gid=%s REJECTED: %s -> %s not adjacent in camera_edges",
                         gid, last_cam, cam_name)
            return False
        elapsed = time.time() - last_time
        lo, hi = self.transit_table.get((last_cam, cam_name), (0, 9999))
        ok = lo <= elapsed <= hi
        if self.cfg.debug_matching and not ok:
            log.info("[match] gid=%s REJECTED: %s -> %s took %.1fs, outside bounds (%s, %s)",
                     gid, last_cam, cam_name, elapsed, lo, hi)
        return ok

    def _last_camera(self, gid: int) -> Optional[str]:
        last = self.raw_last_seen.get(gid)
        return last[0] if last else None

    def _sole_handoff_candidate(self, cam_name: str, active_gids: Optional[set] = None) -> Optional[int]:
        """
        Find a single existing gid that plausibly accounts for a brand-new
        track appearing at cam_name right now, considering TWO situations:

          1. Cross-camera handoff: gid was last seen on a DIFFERENT camera,
             and the graph + transit-time window allows this arrival.
          2. Same-camera recovery: gid was last seen on THIS SAME camera
             very recently (within recent_loss_window_seconds) — i.e. their
             track almost certainly just blipped (occlusion / detection
             miss) rather than them actually having left.

        `active_gids` (gids already tied to a currently-live track on this
        camera) are always excluded — this is what prevents two people who
        are simultaneously on screen from ever being collapsed into one ID.

        Returns None if there are zero or more-than-one candidates (ambiguous
        -> don't guess).
        """
        active_gids = active_gids or set()
        now = time.time()
        candidates = []
        for gid, last in self.raw_last_seen.items():
            if gid in self.exited_ids or gid in active_gids:
                continue
            last_cam, last_time = last
            if last_cam == EXIT_STATE:
                continue
            if last_cam == cam_name:
                if now - last_time <= self.cfg.recent_loss_window_seconds:
                    candidates.append(gid)
                continue
            if self._plausible_transition(gid, cam_name):
                candidates.append(gid)
        return candidates[0] if len(candidates) == 1 else None

    def _match(self, embedding: np.ndarray, gallery: Dict[int, np.ndarray],
               same_cam_threshold: float, cross_cam_threshold: float,
               cam_name: str) -> Optional[int]:
        """
        Same-camera candidates (track recovery) use `same_cam_threshold`.
        Different-camera candidates use the stricter `cross_cam_threshold`,
        so two different people on two different cameras are much less
        likely to be falsely merged into one global ID.
        """
        best_id, best_sim = None, None
        for gid, gal_emb in gallery.items():
            if not self._plausible_transition(gid, cam_name):
                continue
            sim = self._cosine_sim(embedding, gal_emb)
            last_cam = self._last_camera(gid)
            same_cam = last_cam == cam_name
            threshold = same_cam_threshold if same_cam else cross_cam_threshold
            passed = sim > threshold
            if self.cfg.debug_matching:
                log.info("[match] gid=%s last_cam=%s -> %s (%s) sim=%.3f threshold=%.3f %s",
                         gid, last_cam, cam_name, "same-cam" if same_cam else "cross-cam",
                         sim, threshold, "PASS" if passed else "fail")
            if not passed:
                continue
            if best_sim is None or sim > best_sim:
                best_sim, best_id = sim, gid
        return best_id

    # ---------- history / transitions ----------

    def _log_appearance(self, gid: int, cam_name: str) -> None:
        now = time.time()
        # Always update the raw position immediately, regardless of debounce.
        self.raw_last_seen[gid] = (cam_name, now)

        hist = self.history[gid]
        if not hist:
            hist.append((cam_name, now))
            return

        current_cam = hist[-1][0]
        if cam_name == current_cam:
            self.pending_switch.pop(gid, None)
            return

        candidate = self.pending_switch.get(gid)
        if candidate is None or candidate[0] != cam_name:
            self.pending_switch[gid] = (cam_name, now)
            return

        _, first_seen = candidate
        if now - first_seen >= self.cfg.transition_debounce_seconds:
            self._record_transition(current_cam, cam_name)
            if self.trajectory_predictor is not None:
                self.trajectory_predictor.observe_transition(list(hist), cam_name)
            hist.append((cam_name, now))
            self.pending_switch.pop(gid, None)

    def _record_transition(self, from_cam: str, to_cam: str) -> None:
        if from_cam != to_cam:
            self.transition_counts[from_cam][to_cam] += 1

    def predict_next_camera(self, current_cam: str) -> Optional[Dict[str, float]]:
        if current_cam == EXIT_STATE:
            return None
        counts = self.transition_counts.get(current_cam)
        if not counts:
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
            return None
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
                self.raw_last_seen[gid] = (EXIT_STATE, now)
                self.pending_switch.pop(gid, None)
                self.exited_ids.add(gid)

    # ---------- person-level reporting ----------

    def person_report(self, gid: int) -> str:
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
        # Every camera in the graph gets a row, even ones with zero recorded
        # departures yet — predict_next_camera() already has a cold-start
        # fallback (uniform prior over graph neighbors) for those, so they
        # still show a meaningful distribution instead of being omitted.
        from_states = sorted(set(self.graph.nodes()) | set(self.transition_counts.keys()))
        if not from_states:
            print("\n(no cameras in the graph yet)")
            return
        to_states = sorted(
            {t for tos in self.transition_counts.values() for t in tos}
            | set(self.graph.nodes())
            | {EXIT_STATE}
        )
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
               reid_emb: Optional[np.ndarray], cam_name: str,
               active_gids: Optional[set] = None) -> int:
        """
        Match a detection to an existing global identity, or create a new one.
        Face match is tried first (more discriminative); ReID is the fallback.
        Same-camera candidates use the looser recovery threshold; different-
        camera candidates use the stricter cross-camera threshold.

        `active_gids`: gids currently tied to OTHER live tracks on this same
        camera right now — passed through to the sole-candidate handoff so
        it can never assign an ID that's already in use by someone else
        simultaneously on screen.
        """
        if face_emb is not None:
            face_emb = self._normalize(face_emb)
            gid = self._match(
                face_emb, self.face_gallery,
                self.cfg.face_sim_threshold, self.cfg.cross_cam_face_sim_threshold,
                cam_name,
            )
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
            gid = self._match(
                reid_n, self.reid_gallery,
                self.cfg.reid_sim_threshold, self.cfg.cross_cam_reid_sim_threshold,
                cam_name,
            )
            if gid is not None:
                self.reid_gallery[gid] = 0.9 * self.reid_gallery[gid] + 0.1 * reid_n
                if face_emb is not None:
                    self.face_gallery[gid] = face_emb
                self._log_appearance(gid, cam_name)
                return gid

        # Last resort before minting a new identity: if the camera graph +
        # transit-time window allows exactly ONE existing person to be
        # arriving at cam_name right now, treat this as that person's
        # handoff even if appearance similarity alone didn't clear the
        # strict cross-camera threshold. Positional/temporal narrowing to a
        # single candidate is itself strong evidence for a 2-3 camera graph.
        if self.cfg.handoff_sim_threshold >= 0:
            handoff_gid = self._sole_handoff_candidate(cam_name, active_gids)
            if handoff_gid is not None:
                accept = self.cfg.handoff_sim_threshold <= 0
                sim_for_log = None
                if not accept:
                    # if we have an embedding to compare, use it as a sanity
                    # check (very low bar); if we have NO embedding at all
                    # for this gid in either modality, accept on position
                    # alone rather than refuse a match we have no way to score.
                    sims = []
                    if face_emb is not None and handoff_gid in self.face_gallery:
                        sims.append(self._cosine_sim(face_emb, self.face_gallery[handoff_gid]))
                    if reid_emb is not None and handoff_gid in self.reid_gallery:
                        reid_n = self._normalize(reid_emb)
                        sims.append(self._cosine_sim(reid_n, self.reid_gallery[handoff_gid]))
                    if not sims:
                        accept = True
                    else:
                        sim_for_log = max(sims)
                        accept = sim_for_log >= self.cfg.handoff_sim_threshold
                if self.cfg.debug_matching:
                    log.info("[match] cam=%s sole handoff candidate gid=%s sim=%s %s",
                             cam_name, handoff_gid, sim_for_log,
                             "ACCEPTED" if accept else "rejected (too dissimilar)")
                if accept:
                    if face_emb is not None:
                        self.face_gallery[handoff_gid] = (
                            0.9 * self.face_gallery[handoff_gid] + 0.1 * face_emb
                            if handoff_gid in self.face_gallery else face_emb
                        )
                    if reid_emb is not None:
                        reid_n = self._normalize(reid_emb)
                        self.reid_gallery[handoff_gid] = (
                            0.9 * self.reid_gallery[handoff_gid] + 0.1 * reid_n
                            if handoff_gid in self.reid_gallery else reid_n
                        )
                    self._log_appearance(handoff_gid, cam_name)
                    return handoff_gid

        # brand new identity
        gid = self.next_id
        self.next_id += 1
        if self.cfg.debug_matching:
            log.info("[match] cam=%s NEW gid=%s created (face_emb=%s, reid_emb=%s)",
                     cam_name, gid, face_emb is not None, reid_emb is not None)
        if face_emb is not None:
            self.face_gallery[gid] = face_emb
        if reid_emb is not None:
            self.reid_gallery[gid] = self._normalize(reid_emb)
        self._log_appearance(gid, cam_name)
        return gid

    def touch(self, gid: int, cam_name: str) -> None:
        """Log appearance without re-embedding (keeps predictions fresh between checks)."""
        self._log_appearance(gid, cam_name)

    def refresh_embedding(self, gid: int, face_emb: Optional[np.ndarray],
                           reid_emb: Optional[np.ndarray]) -> None:
        """Fold a fresh embedding into the gallery average for an already-
        resolved track (called periodically — see Config.face_check_interval).
        Keeps the stored embedding representative over a long-running track
        instead of being frozen at whatever the resolution-window crop was."""
        if face_emb is not None and gid in self.face_gallery:
            face_n = self._normalize(face_emb)
            self.face_gallery[gid] = 0.9 * self.face_gallery[gid] + 0.1 * face_n
        elif face_emb is not None:
            self.face_gallery[gid] = self._normalize(face_emb)

        if reid_emb is not None and gid in self.reid_gallery:
            reid_n = self._normalize(reid_emb)
            self.reid_gallery[gid] = 0.9 * self.reid_gallery[gid] + 0.1 * reid_n
        elif reid_emb is not None:
            self.reid_gallery[gid] = self._normalize(reid_emb)


# =====================================================================
# 4. PER-CAMERA PROCESSING
# =====================================================================

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
        # New tracks are held here until `identity_resolve_frames` crops have
        # been collected, so we resolve identity from the BEST crop instead
        # of whatever the first (possibly blurry / face-away) frame gave us.
        # key -> list of (score, crop, face_emb_or_None)
        self.pending_tracks: Dict[Tuple[str, int], list] = defaultdict(list)
        self.frame_counter: Dict[str, int] = defaultdict(int)
        # frames since last periodic re-embed, per resolved track
        self.since_refresh: Dict[Tuple[str, int], int] = defaultdict(int)

    def process(self, frame: np.ndarray, cam_name: str) -> np.ndarray:
        self.frame_counter[cam_name] += 1
        color = _PALETTE[hash(cam_name) % len(_PALETTE)]

        results = self.detector.track(
            frame,
            persist=True,
            tracker=r"C:\internship Project\my_bytetrack.yaml",
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

            if key in self.track_gid:
                gid = self.track_gid[key]
                self.gallery.touch(gid, cam_name)

                # Periodically re-embed to keep the gallery entry fresh
                # (this is what Config.face_check_interval now drives).
                self.since_refresh[key] += 1
                if self.since_refresh[key] >= self.cfg.face_check_interval:
                    self.since_refresh[key] = 0
                    face_emb, _ = self._detect_face(crop)
                    reid_emb = self.reid.embed(crop)
                    self.gallery.refresh_embedding(gid, face_emb, reid_emb)
            else:
                active_gids = {
                    g for (c, _), g in self.track_gid.items() if c == cam_name
                }
                gid = self._accumulate_and_maybe_resolve(key, crop, cam_name, active_gids)
                if gid is None:
                    # still buffering — don't draw a box yet for this track
                    continue

            self._draw_box(frame, x1, y1, x2, y2, gid, cam_name, local_id, color)

        return frame

    def _detect_face(self, crop: np.ndarray):
        """Returns (face_embedding_or_None, face_area_or_0)."""
        faces = self.face_app.get(crop)
        if not faces:
            return None, 0
        largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        area = (largest.bbox[2] - largest.bbox[0]) * (largest.bbox[3] - largest.bbox[1])
        return largest.embedding, area

    def _accumulate_and_maybe_resolve(self, key, crop, cam_name, active_gids: set) -> Optional[int]:
        """
        Buffer crops for a brand-new local track. Once `identity_resolve_frames`
        crops have been collected (or the track vanished sooner than that —
        handled implicitly since ByteTrack just won't send more boxes for it),
        pick the crop with the best face score (falls back to largest body
        box if no face was ever found) and resolve identity from THAT crop.
        `active_gids` are gids already tied to other currently-live tracks
        on this camera — passed through so a simultaneous second person can
        never be mis-recovered as an already-on-screen person's ID.
        """
        face_emb, face_area = self._detect_face(crop)
        body_area = crop.shape[0] * crop.shape[1]
        # face area matters far more than body size when present
        score = face_area * 1000 + body_area if face_area > 0 else body_area

        self.pending_tracks[key].append((score, crop, face_emb))

        if len(self.pending_tracks[key]) < self.cfg.identity_resolve_frames:
            return None  # keep buffering

        best_score, best_crop, best_face_emb = max(self.pending_tracks[key], key=lambda t: t[0])
        del self.pending_tracks[key]

        reid_emb = self.reid.embed(best_crop)
        gid = self.gallery.assign(best_face_emb, reid_emb, cam_name, active_gids=active_gids)
        self.track_gid[key] = gid
        return gid

    def _draw_box(self, frame, x1, y1, x2, y2, gid, cam_name, local_id, color) -> None:
        prediction = self.gallery.most_likely_next(gid)
        pred_text = f" -> {prediction[0]} ({prediction[1]:.0%})" if prediction else ""
        label = f"ID {gid} [{cam_name}:{local_id}]{pred_text}"
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
        from trajectory_gnn import TrajectoryPredictor
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
            break

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        loop_count += 1
        if loop_count % 10 == 0:
            gallery.check_for_exits()
        if trajectory_predictor is not None and loop_count % cfg.gnn_train_every_n_frames == 0:
            trajectory_predictor.maybe_train()

    for cap in caps.values():
        cap.release()
    cv2.destroyAllWindows()

    gallery.check_for_exits()
    gallery.print_summary()
    return gallery


if __name__ == "__main__":
    run()