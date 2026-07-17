from dataclasses import dataclass, field
from collections import defaultdict, Counter
from typing import List, Dict, Optional, Tuple, Callable, Any, Sequence
from pathlib import Path
import atexit
import json
import os
import bisect
import threading
import numpy as np
import math
import cv2
import gtsam # first
import gtsam_quadrics #second
import time
from gtsam import Pose3, Rot3, Point3
from gtsam import NonlinearFactorGraph, Values, Symbol
from gtsam.symbol_shorthand import X # X(i) for pose i
from gtsam import ISAM2, ISAM2Params      # <-- added
# ROS visualization messages
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None

# Add near the top imports (above class definitions)
GRAPH: NonlinearFactorGraph = NonlinearFactorGraph()
INITIAL: Values = Values()
RESULT: Values = Values()
# Existing global quadric store
QUADRICS: Dict[int, gtsam_quadrics.ConstrainedDualQuadric] = {}
# ---- Quadric observation counts ----
quadric_observation_count: Dict[int, int] = defaultdict(int)
# ---- Label statistics ----
# For MSG-Loc, TargetLabelLikelihood[qkey][label] = [norm_conf, sum_conf, freq_norm]
target_label_likelihood: Dict[int, Dict[str, List[float]]] = defaultdict(dict)
# For GOReloc & SH, BaselineLabel[qkey][label] = [norm_by_score, sum_score, total_score_sum, color]
baseline_label: Dict[int, Dict[str, List[float]]] = defaultdict(dict)
# BaselineLabelCount was likely a separate map<int, map<string, int>>
baseline_label_count: Dict[int, Counter] = defaultdict(Counter)
# Final fixed color per quadric
quadric_fixed_colors: Dict[int, np.ndarray] = {}
EPSILON = 1e-6

def _key_to_int(key: Any) -> int:
    """Normalize GTSAM key/symbol/integer to an int key."""
    if isinstance(key, int):
        return int(key)
    if hasattr(key, "key"):
        try:
            return int(key.key())
        except Exception:
            pass
    return int(key)

def _symbol_char_of_key(key: Any) -> str:
    """Extract the symbol namespace character (x/q, etc.) from an integer key."""
    k = _key_to_int(key)
    ch = gtsam.Symbol(k).chr()
    if isinstance(ch, int):
        ch = chr(ch)
    return str(ch)

def _symbol_str_of_key(key: Any) -> str:
    """Return a debug symbol string (e.g., x123, q17)."""
    k = _key_to_int(key)
    try:
        return str(gtsam.Symbol(k).string())
    except Exception:
        return str(k)

def _ensure_key_namespace(key: Any, expected: str, allow_raw_index: bool = True) -> int:
    """
    Enforce key namespace.
    - Return as-is if already in the expected namespace
    - Promote to expected when it looks like a raw index (namespace-free integer)
    - Raise immediately if x/q namespaces are swapped
    """
    k = _key_to_int(key)
    ch = _symbol_char_of_key(k)

    if ch == expected:
        return k

    if ch in ("x", "q"):
        raise ValueError(
            f"Key namespace mismatch: expected '{expected}', got '{ch}' ({_symbol_str_of_key(k)})"
        )

    if allow_raw_index and k >= 0:
        return int(gtsam.symbol(expected, int(k)))

    raise ValueError(
        f"Invalid key for namespace '{expected}': {k} ({_symbol_str_of_key(k)})"
    )

def ensure_pose_key(key: Any) -> int:
    return _ensure_key_namespace(key, "x", allow_raw_index=True)

def ensure_quadric_key(key: Any) -> int:
    return _ensure_key_namespace(key, "q", allow_raw_index=True)

class Detection:
    def __init__(self, bounds, colors, labels, scores):
        self.bounds = bounds
        self.colors = colors
        self.labels = labels
        self.scores = scores
        self.median_depth = None
        self.source_key = None
        self.quadric_key = None
        self.ob_radii = None
        self.ob_pose3 = None

@dataclass
class AssociationResult:
    associated: List[Detection]
    unassociated: List[Detection]

@dataclass
class QuadricNode:
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    orientation: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0]))
    radii: np.ndarray       = field(default_factory=lambda: np.zeros(3))
    neighbor_node: List[int] = field(default_factory=list)
    bound: Optional[Any]     = None
    sequence: int            = 0
    caption: Optional[str]           = None
    labels: List[str]                = field(default_factory=list)
    scores: List[float]              = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'translation': self.translation.tolist(),
            'orientation': self.orientation.tolist(),
            'radii': self.radii.tolist(),
            'neighbor_node': self.neighbor_node,
            'bound': self.bound,
            'sequence': self.sequence,
            'caption': self.caption,
            'labels': self.labels,
            'scores': self.scores
        }

    @staticmethod
    def from_dict(data: dict) -> 'QuadricNode':
        node = QuadricNode()
        node.translation  = np.array(data['translation'])
        node.orientation  = np.array(data['orientation'])
        node.radii        = np.array(data['radii'])
        node.neighbor_node = data.get('neighbor_node', [])
        node.bound        = data.get('bound')
        node.sequence     = data.get('sequence', 0)
        node.caption      = data.get('caption')
        node.labels       = data.get('labels', [])
        node.scores       = data.get('scores', [])
        return node

@dataclass
class QuadricGraph:
    nodes: Dict[int, QuadricNode] = field(default_factory=dict)
    edges: Dict[Tuple[int, int], float] = field(default_factory=dict)

    def add_node(self, key: int, node: QuadricNode):
        self.nodes[key] = node

    def has_edge(self, key1: int, key2: int) -> bool:
        a, b = sorted((key1, key2))
        return (a, b) in self.edges

    def add_edge(self, key1: int, key2: int, distance: float):
        a, b = sorted((key1, key2))
        if not self.has_edge(a, b):
            self.edges[(a, b)] = distance
            self.nodes[a].neighbor_node.append(b)
            self.nodes[b].neighbor_node.append(a)

    def to_dict(self) -> Dict:
        return {
            'nodes': {k: v.to_dict() for k, v in self.nodes.items()},
            'edges': {f"{k[0]}_{k[1]}": d for k, d in self.edges.items()}
        }

    @staticmethod
    def from_dict(data: Dict) -> 'QuadricGraph':
        graph = QuadricGraph()
        for key_str, node_data in data['nodes'].items():
            key = int(key_str)
            graph.nodes[key] = QuadricNode.from_dict(node_data)
        for k_str, dist in data['edges'].items():
            a, b = map(int, k_str.split('_'))
            graph.edges[(a, b)] = dist
            graph.nodes[a].neighbor_node.append(b)
            graph.nodes[b].neighbor_node.append(a)
        return graph
# -----------------------------
# Frame -> source graph construction
# -----------------------------
def _pixel_to_point( u: float, v: float, depth_m: float, K: gtsam.Cal3_S2) -> Tuple[float, float, float]:
    # pinhole: z=depth, x=(u-cx)*z/fx, y=(v-cy)*z/fy
    x = (u - K.px()) * depth_m / K.fx()
    y = (v - K.py()) * depth_m / K.fy()
    z = depth_m
    return (x, y, z)

def build_source_graph_from_frame( frame: dict, est_for_assoc, calib_rgb) -> Tuple[QuadricGraph, float, Optional[str]]:
    """
    frame: {"stamp": float, "detections": [...], (opt) "rgb_path","depth_path"}
    Returns: (source_graph, stamp, depth_path)
    """
    source_graph = QuadricGraph()
    stamp = frame.stamp

    # Parse detections
    dets = frame.detections
    for i, det in enumerate(dets):
        # bbox_xywh: [xc, yc, w, h]
        xc, yc, w, h = det["bbox_xywh"]
        xmin = float(xc - 0.5 * w); xmax = float(xc + 0.5 * w)
        ymin = float(yc - 0.5 * h); ymax = float(yc + 0.5 * h)
        qbox = gtsam_quadrics.AlignedBox2(xmin, ymin, xmax, ymax)
     

        # 1.3 median depth
        md = det.get("median_depth", None)
  
        
        if md is None or md < 0.01 or md > 15.0:
            continue

        # 2) bbox 4 midpoints -> camera-frame 3D -> world-frame 3D
        midpix = [
            ((qbox.xmin() + qbox.xmax()) * 0.5, qbox.ymin()),  # top_mid
            ((qbox.xmin() + qbox.xmax()) * 0.5, qbox.ymax()),  # bottom_mid
            (qbox.xmin(), (qbox.ymin() + qbox.ymax()) * 0.5),  # left_mid
            (qbox.xmax(), (qbox.ymin() + qbox.ymax()) * 0.5),  # right_mid
        ]
        pts_world = []
        for (u, v) in midpix:
            p_cam = _pixel_to_point(u, v, md, calib_rgb)  # (x,y,z) in camera
            p_w = est_for_assoc.transformFrom(gtsam.Point3(*p_cam))
            pts_world.append(p_w)

        # 3) Initialize center, radii, and pose
        pts_np = np.array([
            (p if isinstance(p, np.ndarray) else [p.x(), p.y(), p.z()])
            for p in pts_world
        ], dtype=float)

        cx, cy, cz = pts_np.mean(axis=0)
        center = np.array([cx, cy, cz], dtype=float)
        new_center = center


        node = QuadricNode()
        node.translation = new_center
        node.bound = qbox
        node.labels   = list(det.get("concept", []))
        node.scores   = list(det.get("score", []))
        
        # Range filtering (optional)
        if np.linalg.norm(node.translation) <= 15.0 and np.linalg.norm(node.translation) >= 0.01:
            source_graph.add_node(i, node)

    # Neighbor edges (nearest K)
    for k, n in source_graph.nodes.items():
        dists = []
        for o, m in source_graph.nodes.items():
            if o == k:
                continue
            d = float(np.linalg.norm(n.translation - m.translation))
            dists.append((d, o))
        dists.sort()
        for _, o in dists[:min(5, len(dists))]:
            d = float(np.linalg.norm(source_graph.nodes[k].translation - source_graph.nodes[o].translation))
            source_graph.add_edge(k, o, d)

    return source_graph, stamp
    
def PQBsFromValues(values: gtsam.Values) -> Dict[int, gtsam_quadrics.ConstrainedDualQuadric]:
    global QUADRICS, quadric_fixed_colors

    quadric_map: Dict[int, gtsam_quadrics.ConstrainedDualQuadric] = {}
    fixed_color_map: Dict[int, np.ndarray] = {}

    for key in values.keys():
        ch = gtsam.Symbol(key).chr()
        if isinstance(ch, int):
            ch = chr(ch)
        if ch != 'q':
            continue

        try:
            q = gtsam_quadrics.ConstrainedDualQuadric.getFromValues(values, key)
        except Exception:
            # Skip if it cannot be extracted as a quadric
            continue

        quadric_map[key] = q

        # Build fixed quadric color from baseline_label color (if available)
        BL = baseline_label.get(key)
        if BL:
            _best_lab, stats = max(BL.items(), key=lambda kv: kv[1][0] if len(kv[1]) > 0 else -1.0)
            if len(stats) > 3 and stats[3] is not None:
                col = np.asarray(stats[3], dtype=float)
                if col.size >= 3:
                    fixed_color_map[key] = col[:3]

    # Update globals
    QUADRICS = quadric_map
    quadric_fixed_colors = fixed_color_map

    return QUADRICS

def accumulate_source_to_global_graph(
    global_graph: QuadricGraph,
    initial_values: gtsam.Values,
    source_graph: QuadricGraph,
    associated: List,
    unassociated: List,
    global_knn_k: int = 3,
) -> QuadricGraph:
    """
    Accumulate the source graph into the global quadric graph using association results.
    """
    # ========== 1) Build Source key -> Global key mapping ==========
    source_to_global = {}
    
    for aq in associated:
        source_to_global[aq.source_key] = aq.quadric_key
    
    for uaq in unassociated:
        source_to_global[uaq.source_key] = uaq.quadric_key

    # ========== 2) Synchronize all quadric nodes from initial_values ==========
    for key in initial_values.keys():
        sym = gtsam.Symbol(key)
        ch = sym.chr()
        if isinstance(ch, int):
            ch = chr(ch)
        if ch != 'q':
            continue
        
        global_key = key  # ★ use gtsam key as-is
        
        try:
            quadric = gtsam_quadrics.ConstrainedDualQuadric.getFromValues(
                initial_values, global_key
            )
        except RuntimeError:
            continue
        
        pose = quadric.pose()
        t = pose.translation()
        q = pose.rotation().toQuaternion()
        radii = quadric.radii()

        if hasattr(t, "x"):
            translation = [t.x(), t.y(), t.z()]
        else:
            translation = [float(t[0]), float(t[1]), float(t[2])]

        if hasattr(q, "x"):
            orientation = [q.w(), q.x(), q.y(), q.z()]
        else:
            orientation = [float(q[0]), float(q[1]), float(q[2]), float(q[3])]

        if hasattr(radii, "x"):
            radii_list = [radii.x(), radii.y(), radii.z()]
        else:
            radii_list = [float(radii[0]), float(radii[1]), float(radii[2])]

        # Keep existing neighbors
        existing_neighbors = global_graph.nodes[global_key].neighbor_node if global_key in global_graph.nodes else []

        new_node = QuadricNode(
            translation=translation,
            orientation=orientation,
            radii=radii_list,
            neighbor_node=existing_neighbors
        )
        global_graph.add_node(global_key, new_node)  # ★ use global_key

    # ========== 3) Convert and add edges (from current-frame source graph) ==========
    for (src_key1, src_key2), distance in source_graph.edges.items():
        if src_key1 not in source_to_global or src_key2 not in source_to_global:
            continue
        
        global_key1 = source_to_global[src_key1]  # ★ keep original gtsam key
        global_key2 = source_to_global[src_key2]
        
        if global_key1 == global_key2:
            continue
        
        global_graph.add_edge(global_key1, global_key2, distance)

    # ========== 4) Add k-NN edges in the global graph ==========
    knn_k = max(0, int(global_knn_k))
    if knn_k > 0 and len(global_graph.nodes) > 1:
        global_positions: Dict[int, np.ndarray] = {}
        for gk, node in global_graph.nodes.items():
            try:
                p = np.asarray(node.translation, dtype=float).reshape(-1)
            except Exception:
                continue
            if p.size < 3:
                continue
            p3 = p[:3]
            if not np.all(np.isfinite(p3)):
                continue
            global_positions[gk] = p3

        for key_i, pos_i in global_positions.items():
            dists: List[Tuple[float, int]] = []
            for key_j, pos_j in global_positions.items():
                if key_i == key_j:
                    continue
                d = float(np.linalg.norm(pos_i - pos_j))
                if not np.isfinite(d):
                    continue
                dists.append((d, key_j))

            dists.sort(key=lambda x: x[0])
            for d, key_j in dists[:min(knn_k, len(dists))]:
                global_graph.add_edge(key_i, key_j, d)

    return global_graph



# =========================
# Keyframe (single class)
# =========================
@dataclass
class Keyframe:
    index: int
    stamp: float
    rgb_path: str
    depth_path: str
    detections: List[Dict]
    vo_t: Optional[Tuple[float, float, float]] = None          # (tx, ty, tz)
    vo_q: Optional[Tuple[float, float, float, float]] = None   # (qx, qy, qz, qw)

    def as_dict(self) -> Dict:
        d = {
            "index": self.index, "stamp": self.stamp,
            "rgb_path": self.rgb_path, "depth_path": self.depth_path,
            "detections": self.detections,
        }
        if self.vo_t is not None and self.vo_q is not None:
            d["vo_pose"] = {"t": self.vo_t, "q": self.vo_q}
        return d

    # --- Utilities: VO parsing/nearest search/interval ---
    @staticmethod
    def build_vo(lines: List[str]) -> Dict[str, List]:
        stamps, trans, quats = [], [], []
        for ln in lines:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            vals = ln.split()
            if len(vals) != 8:
                continue
            s, tx, ty, tz, qx, qy, qz, qw = map(float, vals)
            stamps.append(s)
            trans.append((tx, ty, tz))
            quats.append((qx, qy, qz, qw))
        order = sorted(range(len(stamps)), key=lambda i: stamps[i])
        return {
            "stamps": [stamps[i] for i in order],
            "t":      [trans[i]  for i in order],
            "q":      [quats[i]  for i in order],
        }

    @staticmethod
    def find_nearest_vo(stamp: float, vo: Dict[str, List], tol: float = 0.02):
        if not vo or not vo.get("stamps"):
            return None, None
        stamps = vo["stamps"]
        i = bisect.bisect_left(stamps, stamp)
        cands = [j for j in (i, i-1) if 0 <= j < len(stamps)]
        if not cands:
            return None, None
        best = min(cands, key=lambda j: abs(stamps[j] - stamp))
        if abs(stamps[best] - stamp) <= tol:
            return vo["t"][best], vo["q"][best]
        return None, None

    @staticmethod
    def should_be_keyframe(last_kf_stamp: Optional[float], new_stamp: float, dt: float) -> bool:
        return (last_kf_stamp is None) or (new_stamp - last_kf_stamp >= dt)

    @classmethod
    def from_sequence(cls, seq: Dict, interval_sec: float = 0.1,
                      vo: Optional[Dict[str, List]] = None, vo_tolerance_sec: float = 0.02,
                      base_path: Optional[str] = None) -> List["Keyframe"]:
        frames: List[Dict] = seq.get("frames", [])
        if not frames:
            return []
        frames_sorted = sorted(enumerate(frames), key=lambda kv: float(kv[1]["stamp"]))
        keyframes, last_kf_time = [], None

        def _resolve_frame_path(p: str) -> str:
            if not p:
                return ""
            if os.path.isabs(p):
                return p
            if base_path:
                return str(Path(base_path) / p)
            return p

        for idx, fr in frames_sorted:
            t = float(fr["stamp"])
            if cls.should_be_keyframe(last_kf_time, t, interval_sec):
                vo_t = vo_q = None
                if vo is not None:
                    vo_t, vo_q = cls.find_nearest_vo(t, vo, tol=vo_tolerance_sec)
                rgb_path = _resolve_frame_path(str(fr.get("rgb_path", "")))
                depth_path = _resolve_frame_path(str(fr.get("depth_path", "")))
                keyframes.append(cls(
                    index=idx, stamp=t,
                    rgb_path=rgb_path, depth_path=depth_path,
                    detections=fr.get("detections", []),
                    vo_t=vo_t, vo_q=vo_q
                ))
                last_kf_time = t
        return keyframes


# =========================
# KeyframeBuilder
# =========================
class KeyframeBuilder:
    """
    Accept file paths only, process internally, and create self.keyframe (List[Keyframe]).
    """
    def __init__(self,
                 detection_json: str,
                 odom_txt: Optional[str] = None,
                 interval_sec: float = 0.1,
                 vo_tolerance_sec: float = 0.02,
                 base_path: Optional[str] = None) -> None:
        self.detection_json_path = Path(detection_json)
        self.odom_txt_path = Path(odom_txt) if odom_txt else None
        self.interval_sec = float(interval_sec)
        self.vo_tolerance_sec = float(vo_tolerance_sec)
        self.base_path = str(Path(base_path).expanduser()) if base_path else ""
        self.keyframe: List[Keyframe] = []

    # -------- Internal I/O --------
    def _load_detections(self) -> Dict:
        p = self.detection_json_path
        if not p.exists():
            raise FileNotFoundError(f"Detection JSON not found: {p}")
        with p.open("r") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "sequences" not in data:
            raise ValueError("Detection JSON must contain a top-level 'sequences' list.")
        return data

    def _load_vo(self) -> Optional[Dict[str, List]]:
        if self.odom_txt_path is None:
            return None
        if not self.odom_txt_path.exists():
            raise FileNotFoundError(f"Odometry file not found: {self.odom_txt_path}")
        lines = self.odom_txt_path.read_text().splitlines()
        return Keyframe.build_vo(lines)

    # -------- Core build --------
    def build(self) -> List[Keyframe]:
        self.keyframe.clear()
        data = self._load_detections()
        vo = self._load_vo()

        sequences = data.get("sequences", [])
        for seq in sequences:
            kfs = Keyframe.from_sequence(
                seq,
                interval_sec=self.interval_sec,
                vo=vo,
                vo_tolerance_sec=self.vo_tolerance_sec,
                base_path=self.base_path,
            )
            self.keyframe.extend(kfs)
        return self.keyframe

    # -------- Serialization helper --------
    def to_dicts(self) -> List[Dict]:
        return [kf.as_dict() for kf in self.keyframe]


# =========================
# Associate
# =========================
class Association:
    """
    Simple stateless association:
    - Perform gating/association without an external map (treat all as new at early stage)
    - For each detection, build an initial 3D quadric (center/radii/pose) using median_depth
    - If detection dict has 'quadric_key', return it as associated (optional)
    """
    def __init__(
        self,
        start_qid: int = 0,            # Start index for new quadric keys
        radius_threshold: float = 15.0,  # Max camera-to-quadric center distance gate (m)
        inter_min_ratio: float = 0.1,  # Minimum projected bbox ratio remaining inside image
        
        was_min: float = 0.3,          # Minimum Wasserstein similarity (drop candidate if below)
        iou_min: float = 0.1,          # Minimum 2D bbox IoU (geometric overlap gate)
        assoc_score_min: float = 0.7,  # Minimum final association score (was + iou + label_score)
        det_topk: int = 3,             # Number of top quadric candidates kept per detection
        q_topk: int = 3,               # Number of top detection candidates kept per quadric
        label_match_topk: int = 5,     # Number of top detection labels used in label scoring
        label_match_min: int = 1,      # Minimum number of label matches (reject if below)
        
        new_object_score_min: float = 0.05,     # Minimum confidence to treat as a new-object candidate
        new_object_area_min: float = 5.0 * 5.0,  # Minimum bbox area for a new-object candidate (px^2)
        consecutive_init_frames: int = 3,       # Frames of consecutive observation before confirming new quadric
        pending_iou_min: float = 0.1,           # Minimum IoU for pending-track matching
        pending_label_must_match: bool = False,  # Force top-1 label match for pending matching
    ):
        self._next_qid = start_qid
        # State used to delay new-quadric initialization
        self._pending_tracks: Dict[int, Dict[str, Any]] = {}
        self._next_pending_id: int = 0
        self._assoc_frame_seq: int = 0

        # Parameter groups by feature
        self._projection_param: Dict[str, float] = {
            "radius_threshold": float(radius_threshold),
            "inter_min_ratio": float(inter_min_ratio),
        }
        self._association_param: Dict[str, Any] = {
            "was_min": float(was_min),
            "iou_min": float(iou_min),
            "assoc_score_min": float(assoc_score_min),
            "det_topk": max(1, int(det_topk)),
            "q_topk": max(1, int(q_topk)),
            "label_match_topk": max(1, int(label_match_topk)),
            "label_match_min": max(1, int(label_match_min)),
        }
        self._new_object_param: Dict[str, Any] = {
            "new_object_score_min": float(new_object_score_min),
            "new_object_area_min": float(new_object_area_min),
            "consecutive_init_frames": max(1, int(consecutive_init_frames)),
            "pending_iou_min": float(pending_iou_min),
            "pending_label_must_match": bool(pending_label_must_match),
        }
        
    def _calib_from_dict(self, camera_param: Dict[str, float]) -> gtsam.Cal3_S2:
        fx = float(camera_param["fx"]); fy = float(camera_param["fy"])
        cx = float(camera_param["cx"]); cy = float(camera_param["cy"])
        skew = float(camera_param.get("skew", 0.0))  # updated
        return gtsam.Cal3_S2(fx, fy, skew, cx, cy)
    

    def _new_qkey(self) -> int:
        """
        Return a GTSAM Key directly. In Python, gtsam.symbol(ord('q'), i) is common.
        It is used directly as a key in the main loop, so return Key instead of int.
        """
        qid = self._next_qid
        self._next_qid += 1
        return gtsam.symbol('q', qid)

    def _as_key(self, k: Any) -> int:
        if isinstance(k, int):
            return k
        if hasattr(k, "key"):                 # gtsam.Symbol
            return int(k.key())
        try:
            return int(k)                     # string like "123"
        except Exception:
            # Last resort: treat as index and create a 'q' namespace key
            return gtsam.symbol('q', int(k))
        
        
    # --- Internal helper (near top of run(), before use) ---
    def _clip_box_to_image(self,b: gtsam_quadrics.AlignedBox2, W: float, H: float):
        axmin = max(0.0, min(float(b.xmin()), W))
        aymin = max(0.0, min(float(b.ymin()), H))
        axmax = max(0.0, min(float(b.xmax()), W))
        aymax = max(0.0, min(float(b.ymax()), H))
        return axmin, aymin, axmax, aymax
    
    def iou_aligned_box2(self,
        bb1: gtsam_quadrics.AlignedBox2,
        bb2: gtsam_quadrics.AlignedBox2,
    ) -> float:
        """
        Compute IoU (Intersection-over-Union) for two AlignedBox2 boxes (bb1, bb2).
        Return 0.0 if there is no valid area or if the union is too small.
        """
        x1_min = float(bb1.xmin())
        y1_min = float(bb1.ymin())
        x1_max = float(bb1.xmax())
        y1_max = float(bb1.ymax())

        x2_min = float(bb2.xmin())
        y2_min = float(bb2.ymin())
        x2_max = float(bb2.xmax())
        y2_max = float(bb2.ymax())

        # Area of each box
        w1 = max(0.0, x1_max - x1_min)
        h1 = max(0.0, y1_max - y1_min)
        w2 = max(0.0, x2_max - x2_min)
        h2 = max(0.0, y2_max - y2_min)

        area1 = w1 * h1
        area2 = w2 * h2

        if area1 <= 0.0 or area2 <= 0.0:
            return 0.0

        # Intersection area
        ix_min = max(x1_min, x2_min)
        iy_min = max(y1_min, y2_min)
        ix_max = min(x1_max, x2_max)
        iy_max = min(y1_max, y2_max)

        iw = max(0.0, ix_max - ix_min)
        ih = max(0.0, iy_max - iy_min)
        inter = iw * ih

        # union = area1 + area2 - inter
        union = area1 + area2 - inter
        if union <= 1e-12:
            return 0.0

        return inter / union

    def _wasserstein_box(self,a: gtsam_quadrics.AlignedBox2,
                        b: gtsam_quadrics.AlignedBox2,
                        W: float, H: float) -> float:
        # Clip to image bounds
        axmin, aymin, axmax, aymax = self._clip_box_to_image(a, W, H)
        bxmin, bymin, bxmax, bymax = self._clip_box_to_image(b, W, H)

        # Center
        acx = 0.5 * (axmin + axmax)
        acy = 0.5 * (aymin + aymax)
        bcx = 0.5 * (bxmin + bxmax)
        bcy = 0.5 * (bymin + bymax)

        # Width/height
        aw = max(0.0, axmax - axmin)
        ah = max(0.0, aymax - aymin)
        bw = max(0.0, bxmax - bxmin)
        bh = max(0.0, bymax - bymin)

        # Variance (use /4.0 to match code convention)
        avx = (aw * aw) / 4.0
        avy = (ah * ah) / 4.0
        bvx = (bw * bw) / 4.0
        bvy = (bh * bh) / 4.0

        # 2-Wasserstein^2 (between diagonal-covariance Gaussians): ||m1-m2||^2 + Σ(v1+v2 - 2*sqrt(v1*v2))
        dcx = acx - bcx
        dcy = acy - bcy
        mean_term = dcx * dcx + dcy * dcy
        cov_term  = (avx + bvx - 2.0 * np.sqrt(max(0.0, avx * bvx))) \
                + (avy + bvy - 2.0 * np.sqrt(max(0.0, avy * bvy)))
        return mean_term + cov_term    

    def label_totalscore_from_kf(
        self,
        kf: "Keyframe",
        det_index: int,
        qkey: int,
        max_matches: int = 5,
    ) -> Tuple[float, int]:
        """
        Python version of totalscore() from map_read.cpp.
        - source: concept/score from kf.detections[det_index]
        - target: target_label_likelihood[qkey]
        Returns: (TotalScore, matchedLabelsCount)
        """
        if det_index < 0 or det_index >= len(kf.detections):
            return 0.0, 0

        det = kf.detections[det_index]
        source_labels = det.get("concept", [])
        source_scores_raw = det.get("score", [])

        if not isinstance(source_labels, (list, tuple, np.ndarray)) or len(source_labels) == 0:
            return 0.0, 0

        if isinstance(source_scores_raw, (int, float)):
            source_scores = [float(source_scores_raw)] * len(source_labels)
        elif isinstance(source_scores_raw, (list, tuple, np.ndarray)):
            source_scores = [float(s) for s in source_scores_raw]
        else:
            return 0.0, 0

        n = min(len(source_labels), len(source_scores))
        if n <= 0:
            return 0.0, 0

        # Normalize source scores (used similarly to normalsourceLabelcandidate)
        source_map: Dict[str, float] = {}
        for lab, sc in zip(source_labels[:n], source_scores[:n]):
            try:
                lab_key = str(lab)
                score = max(0.0, float(sc))
            except Exception:
                continue
            source_map[lab_key] = source_map.get(lab_key, 0.0) + score

        if not source_map:
            return 0.0, 0

        score_sum = float(sum(source_map.values()))
        if score_sum > 1e-12:
            source_map = {lab: sc / score_sum for lab, sc in source_map.items()}
        else:
            uniform = 1.0 / float(len(source_map))
            source_map = {lab: uniform for lab in source_map.keys()}

        target_map = target_label_likelihood.get(qkey, {})
        if not target_map:
            return 0.0, 0

        total_score = 0.0
        matched_count = 0

        for source_label, source_score in sorted(source_map.items(), key=lambda kv: kv[1], reverse=True):
            target_data = target_map.get(source_label)
            if target_data is None:
                continue

            if isinstance(target_data, (list, tuple, np.ndarray)):
                if len(target_data) == 0:
                    continue
                # Backward compatibility: use index 2 for [norm_conf, sum_conf, freq_norm], otherwise index 0
                target_likelihood = float(target_data[2] if len(target_data) > 2 else target_data[0])
            else:
                try:
                    target_likelihood = float(target_data)
                except Exception:
                    continue

            total_score += source_score * target_likelihood
            matched_count += 1
            if matched_count >= max_matches:
                break

        return total_score, matched_count

    def _top_label_from_det(self, det: "Detection") -> Optional[str]:
        if not det.labels:
            return None
        if det.scores and len(det.scores) == len(det.labels):
            try:
                idx = int(np.argmax(np.asarray(det.scores, dtype=float)))
                return str(det.labels[idx])
            except Exception:
                pass
        return str(det.labels[0])

    def _advance_frame_and_prune_pending(self) -> int:
        """Advance frame index for consecutive observations and keep only previous-frame pending tracks."""
        current_frame_seq = self._assoc_frame_seq
        self._assoc_frame_seq += 1
        self._pending_tracks = {
            tid: tr for tid, tr in self._pending_tracks.items()
            if int(tr.get("last_frame", -999999)) == (current_frame_seq - 1)
        }
        return current_frame_seq

    def _solve_assignment(
        self,
        score_matrix: np.ndarray,
        valid_mask: np.ndarray,
    ) -> Tuple[List[Tuple[int, int]], str]:
        """
        Global 1:1 assignment constrained by valid_mask.
        Returns: ([(row, col), ...], assign_mode)
        """
        if score_matrix.size == 0 or valid_mask.size == 0 or not bool(np.any(valid_mask)):
            return [], "none"

        if linear_sum_assignment is not None:
            invalid_score = -1e9
            masked_score_matrix = np.where(valid_mask, score_matrix, invalid_score)
            row_ind, col_ind = linear_sum_assignment(-masked_score_matrix)
            pairs = [
                (int(r), int(c))
                for r, c in zip(row_ind, col_ind)
                if bool(valid_mask[r, c])
            ]
            return pairs, "hungarian"

        # Fallback for environments without scipy: score-based greedy 1:1
        rr, cc = np.where(valid_mask)
        edges = [(float(score_matrix[r, c]), int(r), int(c)) for r, c in zip(rr, cc)]
        edges.sort(key=lambda x: x[0], reverse=True)
        used_rows: set = set()
        used_cols: set = set()
        pairs: List[Tuple[int, int]] = []
        for _score, r, c in edges:
            if r in used_rows or c in used_cols:
                continue
            used_rows.add(r)
            used_cols.add(c)
            pairs.append((r, c))
        return pairs, "greedy_fallback"

    def _project_visible_quadrics(
        self,
        est_for_assoc: gtsam.Pose3,
        calib_rgb: gtsam.Cal3_S2,
        width: int,
        height: int,
    ) -> Dict[int, Tuple[gtsam_quadrics.AlignedBox2, Any]]:
        """Project only in-view quadrics from camera pose and build bbox lookup."""
        radius_threshold = float(self._projection_param["radius_threshold"])
        inter_min_ratio = float(self._projection_param["inter_min_ratio"])
        img_xmin, img_ymin = 0.0, 0.0
        img_xmax, img_ymax = float(width), float(height)

        cam_t = est_for_assoc.translation()
        camera_position = (
            cam_t if isinstance(cam_t, np.ndarray)
            else np.array([cam_t.x(), cam_t.y(), cam_t.z()], dtype=float)
        )
        camera_direction = np.array(est_for_assoc.rotation().matrix())[:, 2].astype(float)

        proj_bboxes: Dict[int, Tuple[gtsam_quadrics.AlignedBox2, Any]] = {}
        for qkey, q in QUADRICS.items():
            qc = q.pose().translation()
            q_center = (
                qc if isinstance(qc, np.ndarray)
                else np.array([qc.x(), qc.y(), qc.z()], dtype=float)
            )

            v = q_center - camera_position
            dist = float(np.linalg.norm(v))
            if dist <= 1e-9:
                continue
            front = float(np.dot(camera_direction, v / dist))
            if dist > radius_threshold or front <= 0.0:
                continue

            try:
                proj = gtsam_quadrics.QuadricCamera.project(q, est_for_assoc, calib_rgb).bounds()
            except Exception:
                continue

            pw = max(0.0, float(proj.xmax() - proj.xmin()))
            ph = max(0.0, float(proj.ymax() - proj.ymin()))
            area_p = pw * ph
            if area_p <= 0.0:
                continue

            ix0 = max(float(proj.xmin()), img_xmin)
            iy0 = max(float(proj.ymin()), img_ymin)
            ix1 = min(float(proj.xmax()), img_xmax)
            iy1 = min(float(proj.ymax()), img_ymax)
            inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
            if inter < inter_min_ratio * area_p:
                continue

            proj_bboxes[qkey] = (gtsam_quadrics.AlignedBox2(ix0, iy0, ix1, iy1), qc)
        return proj_bboxes

    def _build_valid_detections(
        self,
        keyframe: "Keyframe",
        est_for_assoc: gtsam.Pose3,
        calib_rgb: gtsam.Cal3_S2,
    ) -> List[Tuple[int, "Detection", float, float]]:
        """Convert input detections to Association Detection objects and filter by quality."""
        valid_dets: List[Tuple[int, Detection, float, float]] = []
        for i, d in enumerate(keyframe.detections):
            bbox = d.get("bbox_xywh")
            if not bbox or len(bbox) != 4:
                continue

            cx, cy, w_box, h_box = map(float, bbox)
            xmin = cx - w_box / 2.0
            ymin = cy - h_box / 2.0
            bounds = gtsam_quadrics.AlignedBox2(xmin, ymin, xmin + w_box, ymin + h_box)

            det = Detection(
                bounds=bounds,
                colors=d.get("color_rgb", []),
                labels=d.get("concept", []),
                scores=d.get("score", []),
            )
            det.source_key = i

            md = d.get("median_depth")
            det.median_depth = float(md) if md is not None else None
            if det.median_depth is None or det.median_depth < 0.01 or det.median_depth > 10:
                continue

            midpix = [
                ((bounds.xmin() + bounds.xmax()) * 0.5, bounds.ymin()),
                ((bounds.xmin() + bounds.xmax()) * 0.5, bounds.ymax()),
                (bounds.xmin(), (bounds.ymin() + bounds.ymax()) * 0.5),
                (bounds.xmax(), (bounds.ymin() + bounds.ymax()) * 0.5),
            ]
            pts_world = []
            for (u, v) in midpix:
                p_cam = _pixel_to_point(u, v, det.median_depth, calib_rgb)
                pts_world.append(est_for_assoc.transformFrom(gtsam.Point3(*p_cam)))

            pts_np = np.array(
                [(p if isinstance(p, np.ndarray) else [p.x(), p.y(), p.z()]) for p in pts_world],
                dtype=float,
            )
            cx, cy, cz = pts_np.mean(axis=0)
            vx = pts_np[3] - pts_np[2]
            vy = pts_np[1] - pts_np[0]
            rx = np.linalg.norm(vx) / 2.0
            ry = np.linalg.norm(vy) / 2.0
            rz = (rx + ry) / 2.0

            det.ob_radii = np.array([rx, ry, rz])
            rot = est_for_assoc.rotation().compose(gtsam.Rot3(np.eye(3)))
            det.ob_pose3 = gtsam.Pose3(rot, gtsam.Point3(cx, cy, cz))

            det_area = max(0.0, w_box) * max(0.0, h_box)
            det_conf = 0.0
            if isinstance(det.scores, (list, tuple, np.ndarray)) and len(det.scores) > 0:
                try:
                    det_conf = float(max(det.scores))
                except Exception:
                    det_conf = 0.0
            valid_dets.append((i, det, det_area, det_conf))
        return valid_dets

    def _build_det_quadric_score_matrix(
        self,
        keyframe: "Keyframe",
        valid_dets: List[Tuple[int, "Detection", float, float]],
        proj_bboxes: Dict[int, Tuple[gtsam_quadrics.AlignedBox2, Any]],
        width: int,
        height: int,
    ) -> Tuple[
        List[int],
        List[int],
        Dict[int, "Detection"],
        Dict[int, Tuple[float, float]],
        np.ndarray,
        np.ndarray,
        int,
        int,
    ]:
        """Build the Det x Quadric score matrix and gate mask."""
        assoc_param = self._association_param
        was_min = float(assoc_param["was_min"])
        iou_min = float(assoc_param["iou_min"])
        assoc_score_min = float(assoc_param["assoc_score_min"])
        label_match_topk = int(assoc_param["label_match_topk"])
        label_match_min = int(assoc_param["label_match_min"])

        det_ids: List[int] = [det_idx for det_idx, _det, _area, _conf in valid_dets]
        q_ids: List[int] = sorted(proj_bboxes.keys())
        det_index_to_det = {det_idx: det for det_idx, det, _area, _conf in valid_dets}
        det_index_to_quality = {
            det_idx: (det_area, det_conf)
            for det_idx, _det, det_area, det_conf in valid_dets
        }

        score_matrix = np.zeros((len(det_ids), len(q_ids)), dtype=float)
        valid_gate_mask = np.zeros((len(det_ids), len(q_ids)), dtype=bool)
        num_candidate_edges = 0
        label_gate_rejects = 0

        for r, det_idx in enumerate(det_ids):
            det = det_index_to_det[det_idx]
            for c, qkey in enumerate(q_ids):
                qbox, _qcenter = proj_bboxes[qkey]
                d_wass = self._wasserstein_box(det.bounds, qbox, float(width), float(height))
                was = float(np.exp(-np.sqrt(max(0.0, d_wass)) / 100.0))
                if was < was_min:
                    continue

                iou = self.iou_aligned_box2(det.bounds, qbox)
                if iou < iou_min:
                    continue

                label_score, matched = self.label_totalscore_from_kf(
                    keyframe,
                    det_idx,
                    qkey,
                    max_matches=label_match_topk,
                )
                if matched < label_match_min:
                    label_gate_rejects += 1
                    continue

                score = was + iou + label_score
                if score < assoc_score_min:
                    continue

                score_matrix[r, c] = score
                valid_gate_mask[r, c] = True
                num_candidate_edges += 1

        return (
            det_ids,
            q_ids,
            det_index_to_det,
            det_index_to_quality,
            score_matrix,
            valid_gate_mask,
            num_candidate_edges,
            label_gate_rejects,
        )

    def _build_mutual_knn_mask(
        self,
        score_matrix: np.ndarray,
        valid_gate_mask: np.ndarray,
    ) -> np.ndarray:
        """Build row/column top-k intersection (mutual kNN) mask."""
        det_topk = int(self._association_param["det_topk"])
        q_topk = int(self._association_param["q_topk"])

        det_top_mask = np.zeros_like(valid_gate_mask, dtype=bool)
        q_top_mask = np.zeros_like(valid_gate_mask, dtype=bool)

        if score_matrix.size > 0:
            for r in range(score_matrix.shape[0]):
                valid_cols = np.where(valid_gate_mask[r])[0]
                if valid_cols.size == 0:
                    continue
                order = valid_cols[np.argsort(-score_matrix[r, valid_cols])]
                det_top_mask[r, order[:det_topk]] = True

            for c in range(score_matrix.shape[1]):
                valid_rows = np.where(valid_gate_mask[:, c])[0]
                if valid_rows.size == 0:
                    continue
                order = valid_rows[np.argsort(-score_matrix[valid_rows, c])]
                q_top_mask[order[:q_topk], c] = True

        return valid_gate_mask & det_top_mask & q_top_mask

    def _collect_pending_candidates(
        self,
        det_ids: List[int],
        assigned_det: set,
        det_index_to_det: Dict[int, "Detection"],
        det_index_to_quality: Dict[int, Tuple[float, float]],
    ) -> Tuple[List[Tuple[int, "Detection", Optional[str]]], int]:
        """Select only new-object candidates among association-failed detections."""
        min_score = float(self._new_object_param["new_object_score_min"])
        min_area = float(self._new_object_param["new_object_area_min"])

        pending_candidates: List[Tuple[int, Detection, Optional[str]]] = []
        dropped_low_quality = 0
        for det_idx in det_ids:
            if det_idx in assigned_det:
                continue
            det = det_index_to_det[det_idx]
            det_area, det_conf = det_index_to_quality[det_idx]
            if det_conf < min_score or det_area < min_area:
                dropped_low_quality += 1
                continue
            pending_candidates.append((det_idx, det, self._top_label_from_det(det)))
        return pending_candidates, dropped_low_quality

    def _update_pending_tracks_global(
        self,
        pending_candidates: List[Tuple[int, "Detection", Optional[str]]],
        current_frame_seq: int,
    ) -> Tuple[List["Detection"], int, int, str]:
        """
        Update pending tracks via global matching on a Det x Pending matrix.
        Promote to a new quadric when streak requirement is met.
        """
        pending_iou_min = float(self._new_object_param["pending_iou_min"])
        pending_label_must_match = bool(self._new_object_param["pending_label_must_match"])
        consecutive_init_frames = int(self._new_object_param["consecutive_init_frames"])

        promoted_dets: List[Detection] = []
        promoted_new = 0
        pending_wait = 0
        pending_assign_mode = "none"
        matched_rows: set = set()

        pending_ids = sorted(self._pending_tracks.keys())
        if pending_candidates and pending_ids:
            iou_matrix = np.zeros((len(pending_candidates), len(pending_ids)), dtype=float)
            valid_mask = np.zeros((len(pending_candidates), len(pending_ids)), dtype=bool)

            for r, (_det_idx, det, det_label) in enumerate(pending_candidates):
                for c, tid in enumerate(pending_ids):
                    tr = self._pending_tracks.get(tid, None)
                    if tr is None:
                        continue
                    tr_bbox = tr.get("bbox", None)
                    if tr_bbox is None:
                        continue

                    iou = self.iou_aligned_box2(det.bounds, tr_bbox)
                    if iou < pending_iou_min:
                        continue

                    tr_label = tr.get("label", None)
                    if (
                        pending_label_must_match
                        and det_label is not None
                        and tr_label is not None
                        and det_label != tr_label
                    ):
                        continue

                    iou_matrix[r, c] = iou
                    valid_mask[r, c] = True

            pair_indices, pending_assign_mode = self._solve_assignment(iou_matrix, valid_mask)
            for r, c in pair_indices:
                _det_idx, det, det_label = pending_candidates[r]
                tid = pending_ids[c]
                tr = self._pending_tracks.get(tid, None)
                if tr is None:
                    continue

                tr["bbox"] = det.bounds
                tr["label"] = det_label if det_label is not None else tr.get("label", None)
                tr["streak"] = int(tr.get("streak", 0)) + 1
                tr["last_frame"] = current_frame_seq
                matched_rows.add(r)

                if int(tr["streak"]) >= consecutive_init_frames:
                    det.quadric_key = self._new_qkey()
                    promoted_dets.append(det)
                    promoted_new += 1
                    self._pending_tracks.pop(tid, None)
                else:
                    pending_wait += 1

        # Start a new pending track for unmatched candidates
        for r, (_det_idx, det, det_label) in enumerate(pending_candidates):
            if r in matched_rows:
                continue
            new_tid = self._next_pending_id
            self._next_pending_id += 1
            self._pending_tracks[new_tid] = {
                "bbox": det.bounds,
                "label": det_label,
                "streak": 1,
                "last_frame": current_frame_seq,
            }
            pending_wait += 1

        return promoted_dets, promoted_new, pending_wait, pending_assign_mode
    
    # ---------- public ----------
    def run(
        self,
        keyframe: "Keyframe",
        est_for_assoc: gtsam.Pose3,
        camera_param: Dict[str, float],
        depth_image: Optional[np.ndarray] = None,
    ) -> Tuple[List["Detection"], List["Detection"]]:
        """
        Args:
            keyframe: keyframe containing detections (used: detections, depth_path)
            est_for_assoc: estimated camera pose at current frame (for gating/initialization) = one-step prediction
            camera_param: {"fx","fy","cx","cy", "depth_factor"(opt)}
            depth_image: if None, load from keyframe.depth_path
        Returns:
            (associated, unassociated)
        """
        # 0) Prepare inputs/state
        calib_rgb = self._calib_from_dict(camera_param)
        width  = int(camera_param.get("width", 640))
        height = int(camera_param.get("height", 480))
        associated: List[Detection] = []
        unassociated: List[Detection] = []

        current_frame_seq = self._advance_frame_and_prune_pending()

        # 1) Camera-pose-based projected quadrics + debug visualization
        proj_bboxes = self._project_visible_quadrics(est_for_assoc, calib_rgb, width, height)

        # 2) Build valid detections
        valid_dets = self._build_valid_detections(keyframe, est_for_assoc, calib_rgb)

        # 3) Det x Quadric score matrix -> mutual kNN -> global assignment
        (
            det_ids,
            q_ids,
            det_index_to_det,
            det_index_to_quality,
            score_matrix,
            valid_gate_mask,
            _num_candidate_edges,
            _label_gate_rejects,
        ) = self._build_det_quadric_score_matrix(
            keyframe=keyframe,
            valid_dets=valid_dets,
            proj_bboxes=proj_bboxes,
            width=width,
            height=height,
        )
        mutual_mask = self._build_mutual_knn_mask(score_matrix, valid_gate_mask)
        _mutual_edge_count = int(np.count_nonzero(mutual_mask))

        pair_indices, _assign_mode = self._solve_assignment(score_matrix, mutual_mask)
        assigned_det: set = set()
        for r, c in pair_indices:
            det_idx = det_ids[r]
            qkey = q_ids[c]
            det = det_index_to_det[det_idx]
            det.quadric_key = qkey
            associated.append(det)
            assigned_det.add(det_idx)

        # 4) Update streak for unassigned detections via Det x Pending global matching
        pending_candidates, _dropped_low_quality = self._collect_pending_candidates(
            det_ids=det_ids,
            assigned_det=assigned_det,
            det_index_to_det=det_index_to_det,
            det_index_to_quality=det_index_to_quality,
        )
        promoted_dets, _promoted_new, _pending_wait, _pending_assign_mode = self._update_pending_tracks_global(
            pending_candidates=pending_candidates,
            current_frame_seq=current_frame_seq,
        )
        unassociated.extend(promoted_dets)

        # Keep only pending tracks updated in the current frame (drop continuity on missed middle frames)
        self._pending_tracks = {
            tid: tr for tid, tr in self._pending_tracks.items()
            if int(tr.get("last_frame", -1)) == current_frame_seq
        }

        return associated, unassociated


# =========================
# Backend
# =========================
class Backend:
    """
    Backend ported from C++ pose_callback logic to Python/GTSAM.
    - Accumulate BetweenFactor from odom (VO)
    - Add PriorFactor on the first keyframe
    - Run LM optimization after factor validation (missing key / linearization failure checks)
    - Extract result Pose and provide callbacks


    Hooks (optional):
    on_graph_odom: Callable[[int, Pose3, float], None]
    on_keyframe_result: Callable[[int, Dict], None]
    """
    def __init__(self,
        prior_sigmas: Tuple[float, float, float, float, float, float] = (0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001),
        odom_sigmas: Tuple[float, float, float, float, float, float] = (0.001, 0.001, 0.001, 0.001, 0.001, 0.001),
        quadric_sigmas_4: Tuple[float, float, float, float] = (100, 100, 100, 100),
        center_sigmas: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        lm_params: Optional[Dict] = None,
        on_graph_odom: Optional[Callable[[int, Pose3, float], None]] = None,
        on_keyframe_result: Optional[Callable[[int, Dict], None]] = None):
        # ↓↓↓ Changed to reference global objects
        self.graph = GRAPH
        self.initial = INITIAL
        self.result = RESULT
        self.odom_id: int = -1
        self.pre_pose: Optional[Pose3] = None


        self.mtx = threading.Lock()


        # Noise models (Pose3 order: [rx, ry, rz, tx, ty, tz]; rotations in radians)
        self.priorNoise = gtsam.noiseModel.Diagonal.Sigmas(np.array(prior_sigmas, dtype=float))
        self.odomNoise = gtsam.noiseModel.Diagonal.Sigmas(np.array(odom_sigmas, dtype=float))
        self.quadricNoise = gtsam.noiseModel.Diagonal.Sigmas(np.array(quadric_sigmas_4, dtype=float))
        self.centerNoise = gtsam.noiseModel.Diagonal.Sigmas(np.array(center_sigmas, dtype=float))

        # LM params
        self.params = gtsam.LevenbergMarquardtParams()
        self.params.setMaxIterations(100)
        # self.params.setRelativeErrorTol(1e-5)
        # self.params.setAbsoluteErrorTol(1e-5)
        # self.params.setVerbosityLM("SUMMARY")
        self.params.setVerbosity("TERMINATION")
        # --- iSAM2 settings ---
        isam_params = gtsam.ISAM2Params()
        isam_params.setRelinearizeThreshold(0.01)
        isam_params.setRelinearizeSkip(2)
        isam_params.setFactorization("QR")
        self.isam = gtsam.ISAM2(isam_params)
        # Optional hooks
        self.on_graph_odom = on_graph_odom
        self.on_keyframe_result = on_keyframe_result
        
        self.odom_pub = rospy.Publisher(
            "/graph_odom",
            Odometry,
            queue_size=50
        )
        self.source_graph_marker_pub = rospy.Publisher(
            "/source_graph_markers", MarkerArray, queue_size=10, latch=True
        )
        self.graph_marker_array_pub = rospy.Publisher(
            "/quadric_graph_markers", MarkerArray, queue_size=10, latch=True
        )
        self.graph_edge_pub = rospy.Publisher(
            "/graph_edges", MarkerArray, queue_size=10, latch=True
        )
        self.rgb_pub = rospy.Publisher('/camera/rgb/image_raw', Image, queue_size=10)
        self.depth_pub = rospy.Publisher('/camera/depth/image_raw', Image, queue_size=10)
        self.quadric_proj_pub = rospy.Publisher('/quadric_projected_image', Image, queue_size=10)
        self.quadric_3d_pub = rospy.Publisher('/quadric_3d_image', Image, queue_size=10)
        self.bridge = CvBridge()
        # Runtime stats
        self.run_count = 0
        self.run_sum = 0.0
        
    def odomToPose3(self, pose: Pose3) -> Pose3:
        """Return odom pose after rotation correction to camera frame (Yaw=-pi/2, Pitch=0, Roll=-pi/2)."""
        rot = pose.rotation()
        trans = pose.translation()
        cam_rot = Rot3.Yaw(-math.pi/2) * Rot3.Pitch(0.0) * Rot3.Roll(-math.pi/2)
        final_rot = rot.compose(cam_rot)
        return Pose3(final_rot, trans)    
    
    def ps_and_qs_from_values(self,values: gtsam.Values):
        # TODO there's got to be a better way to access the typed values...
        return ({
            k: values.atPose3(k)
            for k in values.keys()
            if gtsam.Symbol(k).string()[0] == 'x'
        }, {
            k: gtsam_quadrics.ConstrainedDualQuadric.getFromValues(values, k)
            for k in values.keys()
            if gtsam.Symbol(k).string()[0] == 'q'
        })
        
    def new_factors(self,current: gtsam.NonlinearFactorGraph,
                previous: gtsam.NonlinearFactorGraph):
        # Figure out the new factors
        fs = (set([current.at(i) for i in range(0, current.size())]) -
            set([previous.at(i) for i in range(0, previous.size())]))

        # Return a NEW graph with the factors
        out = gtsam.NonlinearFactorGraph()
        for f in fs:
            out.add(f)
        return out


    def new_values(self, current: gtsam.Values, previous: gtsam.Values):
        # Figure out new values
        cps, cqs = self.ps_and_qs_from_values(current)
        pps, pqs = self.ps_and_qs_from_values(previous)
        vs = {
            **{k: cps[k] for k in list(set(cps.keys()) - set(pps.keys()))},
            **{k: cqs[k] for k in list(set(cqs.keys()) - set(pqs.keys()))}
        }

        # Return NEW values with each of our estimates
        out = gtsam.Values()
        for k, v in vs.items():
            if type(v) == gtsam_quadrics.ConstrainedDualQuadric:
                v.addToValues(out, k)
            else:
                out.insert(k, v)
        return out

    def _factor_keys_as_int(self, factor: Any) -> List[int]:
        """Extract key list from a GTSAM factor as an int list."""
        try:
            keyvec = factor.keys()
        except Exception:
            return []

        out: List[int] = []
        if hasattr(keyvec, "size") and hasattr(keyvec, "at"):
            try:
                n = int(keyvec.size())
                for i in range(n):
                    out.append(int(keyvec.at(i)))
                return out
            except Exception:
                out = []

        try:
            for k in keyvec:
                out.append(int(k))
        except Exception:
            return []
        return out

    def validate_graph_keys_in_initial(
        self,
    ) -> Tuple[bool, List[int]]:
        """
        Pre-check before optimize:
        - Verify all keys referenced by graph factors exist in initial.
        """
        missing_keys: set = set()
        for fi in range(int(self.graph.size())):
            f = self.graph.at(fi)
            for k in self._factor_keys_as_int(f):
                if not self.initial.exists(k):
                    missing_keys.add(int(k))
        if missing_keys:
            return False, sorted(missing_keys)
        return True, []
    
    def optimize(self) -> None:
        ok, _missing_keys = self.validate_graph_keys_in_initial()
        if not ok:
            return

        optimizer = gtsam.LevenbergMarquardtOptimizer(
            self.graph, self.initial, self.params
        )
        try:
            self.initial = optimizer.optimize()
        except Exception:
            # Safe fallback to avoid interrupting the real-time pipeline
            try:
                safe_optimizer = gtsam.LevenbergMarquardtOptimizer(
                    self.graph, self.initial, self.params
                )
                self.initial = safe_optimizer.optimizeSafely()
            except Exception:
                return
        self.run_count += 1
        
        
    def optimize_isam(self) -> None:
        """
        iSAM2-based incremental optimization.
        - self.graph / self.initial hold all accumulated factors/variables so far,
          and new_factors/new_values select only what should be sent to ISAM this step.
        """
        # Extract and filter new factors / new values
        new_fg = self.new_factors(self.graph, self.isam.getFactorsUnsafe())
        new_vals = self.new_values(self.initial, self.isam.getLinearizationPoint())

        # Skip when there is nothing new to add
        if new_fg.size() == 0 and new_vals.size() == 0:
            return

        # Apply this step's factors/initial values to iSAM2
        self.isam.update(new_fg, new_vals)
    
        # Compute latest estimate (for the full graph)
        self.initial = self.isam.calculateEstimate()

        self.run_count += 1
    def init_quadric_stats(
        self,
        qkey: int,
        labels: Sequence[str],
        scores: Sequence[float],
        colors: Optional[Sequence[Sequence[float]]],
    ) -> None:
        """
        When a quadric is first created:
        - Initialize TargetLabelLikelihood
        - Initialize BaselineLabel
        """
        quadric_observation_count[qkey] = 1  # Start at 1 per quadric (object detection count)

        # ---- Build label confidence / color dict ----
        if labels is None or scores is None:
            label_conf = {}
        else:
            label_conf = {str(lab): float(s) for lab, s in zip(labels, scores)}

        if labels is None or colors is None:
            label_color = {}
        else:
            # label -> [R,G,B] (float)
            label_color = {}
            for lab, col in zip(labels, colors):
                if col is None:
                    continue
                try:
                    label_color[str(lab)] = [float(c) for c in col][:3]
                except Exception:
                    continue


        # ---- Initialize TargetLabelLikelihood ----
        # target_label_likelihood[qkey][label] = [freq, count]
        # freq = count / quadric_observation_count[qkey]
        if label_conf:
            tgt = {}
            total_frames = float(quadric_observation_count[qkey])  # currently 1

            for lab in label_conf.keys():
                count = 1.0
                freq  = count / total_frames  # => 1.0 / 1.0 = 1.0
                tgt[lab] = [freq, count]

            target_label_likelihood[qkey] = tgt

        # ---- Initialize BaselineLabel (top-1 label based) ----
        # baseline_label[qkey][label] = [norm_by_score, sum_score, total_score_sum, color]
        if label_conf:
            best_lab, best_raw = max(label_conf.items(), key=lambda kv: kv[1])
            BL = baseline_label[qkey]
            v = BL.get(best_lab, [0.0, 0.0, 0.0, []])
            if len(v) < 4:
                v = list(v) + [[] for _ in range(4 - len(v))]
            v[1] += float(best_raw)  # sum_score
            if best_lab in label_color:
                v[3] = label_color[best_lab]

            BL[best_lab] = v

            # Score-based normalization
            denom = sum(d[1] for d in BL.values())
            for d in BL.values():
                d[0] = (d[1] / denom) if denom > 0.0 else 0.0  # norm_by_score
                d[2] = denom                                   # total_score_sum

                
                
    def update_quadric_stats(
        self,
        qkey: int,
        labels: Sequence[str],
        scores: Sequence[float],
        colors: Optional[Sequence[Sequence[float]]],
    ) -> None:
        """
        When an existing quadric is associated with a new detection:
        - Update TargetLabelLikelihood
        - Update BaselineLabel
        """
        n_prev = quadric_observation_count[qkey]
        n = n_prev + 1
        quadric_observation_count[qkey] = n  # Total object detections = frame count

        # ---- Build label confidence / color dict ----
        if labels is None or scores is None:
            label_conf = {}
        else:
            label_conf = {str(lab): float(s) for lab, s in zip(labels, scores)}

        if labels is None or colors is None:
            label_color = {}
        else:
            label_color = {}
            for lab, col in zip(labels, colors):
                if col is None:
                    continue
                try:
                    label_color[str(lab)] = [float(c) for c in col][:3]
                except Exception:
                    continue

        # ---- Update TargetLabelLikelihood ----
        # target_label_likelihood[qkey][label] = [freq, count]
        # freq = count / quadric_observation_count[qkey]
        if label_conf:
            tgt = target_label_likelihood.setdefault(qkey, {})

            # For labels appearing in this frame, increment count by 1
            for lab in label_conf.keys():
                if lab not in tgt:
                    tgt[lab] = [0.0, 0.0]  # [freq, count]
                tgt[lab][1] += 1.0         # accumulate count

            # For all labels, freq = count / total_frames
            total_frames = float(quadric_observation_count[qkey])
            for data in tgt.values():
                count = data[1]
                data[0] = count / total_frames if total_frames > 0.0 else 0.0

        # ---- Update BaselineLabel (keep existing logic) ----
        if label_conf:
            best_lab, best_raw = max(label_conf.items(), key=lambda kv: kv[1])

            BL = baseline_label[qkey]
            v  = BL.get(best_lab, [0.0, 0.0, 0.0, []])
            if len(v) < 4:
                v = list(v) + [[] for _ in range(4 - len(v))]
            v[1] += float(best_raw)       # sum_score
            if best_lab in label_color:
                v[3] = label_color[best_lab]
            BL[best_lab] = v

            denom = sum(d[1] for d in BL.values())
            for d in BL.values():
                d[0] = (d[1] / denom) if denom > 0.0 else 0.0  # norm_by_score
                d[2] = denom


    def save_map_to_json(
        self,
        filepath: str,
        quadric_graph: "QuadricGraph",
    ) -> None:
        """
        Save QuadricGraph to a JSON file.
        
        Args:
            filepath: path to save JSON file
            quadric_graph: QuadricGraph object to save
        
        JSON format:
            {
                "edges": [{"from": int, "to": int, "distance": float}, ...],
                "nodes": [{
                    "id": int,
                    "translation": [x, y, z],
                    "orientation": [qx, qy, qz, qw],
                    "radii": [rx, ry, rz],
                    "neighbor_node": [...]
                }, ...]
            }
        """
        # ---- Helper: convert numpy array or list to list ----
        def to_list(val):
            if hasattr(val, 'tolist'):
                return val.tolist()
            return list(val) if val is not None else []

        # ---- Helper: compress label stats into {"label": [value]} ----
        def to_single_value_label_map(label_stats: Dict[str, Any]) -> Dict[str, List[float]]:
            compact: Dict[str, List[float]] = {}
            for label, stats in label_stats.items():
                first_value = None
                if isinstance(stats, (list, tuple, np.ndarray)) and len(stats) > 0:
                    first_value = stats[0]
                else:
                    first_value = stats

                try:
                    compact[str(label)] = [float(first_value)]
                except (TypeError, ValueError):
                    continue
            return compact

        # ---- Helper: convert format for baseline_label export ----
        # [norm_by_score, sum_score, total_score_sum, color] -> [norm_by_score, color]
        def to_baseline_export_map(label_stats: Dict[str, Any]) -> Dict[str, List[Any]]:
            compact: Dict[str, List[Any]] = {}
            for label, stats in label_stats.items():
                if isinstance(stats, (list, tuple, np.ndarray)) and len(stats) > 0:
                    try:
                        norm = float(stats[0])
                    except (TypeError, ValueError):
                        continue

                    # If color exists: [norm, [r,g,b]], else: [norm]
                    if len(stats) > 3 and isinstance(stats[3], (list, tuple, np.ndarray)):
                        try:
                            color = [float(c) for c in list(stats[3])[:3]]
                            if len(color) == 3:
                                compact[str(label)] = [norm, color]
                            else:
                                compact[str(label)] = [norm]
                        except (TypeError, ValueError):
                            compact[str(label)] = [norm]
                    else:
                        compact[str(label)] = [norm]
                else:
                    try:
                        compact[str(label)] = [float(stats)]
                    except (TypeError, ValueError):
                        continue
            return compact
        
        # ---- Build nodes ----
        nodes_list = []
        for qkey, node in quadric_graph.nodes.items():
            node_dict = {
                "id": qkey,
                "translation": to_list(node.translation),
                "orientation": to_list(node.orientation),
                "radii": to_list(node.radii),
                "neighbor_node": node.neighbor_node if node.neighbor_node else [],
            }
            
            # ---- Add label stats from global variables ----
            # Save target_label_likelihood with key name msgloc_label in map.json
            if qkey in target_label_likelihood:
                node_dict["msgloc_label"] = to_single_value_label_map(target_label_likelihood[qkey])

            # Keep original key name for baseline_label
            if qkey in baseline_label:
                node_dict["baseline_label"] = to_baseline_export_map(baseline_label[qkey])
            
            nodes_list.append(node_dict)
        
        # ---- Build edges ----
        edges_list = []
        for (from_key, to_key), distance in quadric_graph.edges.items():
            edge_dict = {
                "from": from_key,
                "to": to_key,
                "distance": distance,
            }
            edges_list.append(edge_dict)
        
        # ---- Save JSON ----
        map_data = {
            "nodes": nodes_list,
            "edges": edges_list,
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(map_data, f, indent=2, ensure_ascii=False)

    def publish_image_by_stamp(
            self,
            rgb_dir: str,
            depth_dir: str,          # added
            stamp,
            initial=None,
            pose=None,
            associated=None,
            unassociated=None,
            tolerance: float = 0.05
        ):
        """
        Find and publish the RGB image by timestamp.
        If initial and pose are provided, also publish quadric projected bbox image.
        If associated/unassociated are provided, also draw detection bboxes.
        """
        global quadric_fixed_colors
        
        # Camera calibration
        FX, FY = 525.0, 525.0
        CX, CY = 319.5, 239.5
        # FX, FY = 605.784, 604.493
        # CX, CY = 323.726, 242.095
        calibration = gtsam.Cal3_S2(FX, FY, 0.0, CX, CY)
        
        # ─────────────────────────────────────────
        # 0) Normalize stamp
        # ─────────────────────────────────────────
        if hasattr(stamp, "to_sec"):
            ros_stamp = stamp
            stamp_sec = stamp.to_sec()
        else:
            stamp_sec = float(stamp)
            ros_stamp = rospy.Time.from_sec(stamp_sec)
            
        def ellipsoid_wireframe_vertices(axes, R, center, rings=12, segments=18):
            us = np.linspace(0, 2*np.pi, segments, endpoint=False)
            vs = np.linspace(-np.pi/2, np.pi/2, rings+1)
            verts, lines = [], []
            
            # Latitude rings
            for vi, v in enumerate(vs):
                z, r = np.sin(v), np.cos(v)
                for u in us:
                    verts.append([r*np.cos(u), r*np.sin(u), z])
                base = vi * segments
                lines += [[base+i, base+(i+1)%segments] for i in range(segments)]
            
            # Longitude rings
            offset = len(verts)
            for ui, u in enumerate(us):
                for v in vs:
                    verts.append([np.cos(v)*np.cos(u), np.cos(v)*np.sin(u), np.sin(v)])
                lines += [[offset+ui*(rings+1)+i, offset+ui*(rings+1)+i+1] for i in range(rings)]
            
            verts = (R @ (np.asarray(verts) * np.abs(axes)).T).T + center
            return verts, np.asarray(lines, int)
        
        def project_pts(pts_c):
            x, y, z = pts_c.T
            z = np.where(z == 0, 1e-6, z)
            u = FX * x / z + CX
            v = FY * y / z + CY
            return np.vstack([u, v]).T
        # ─────────────────────────────────────────
        # 1) Find RGB image
        # ─────────────────────────────────────────
        rgb_path = self._find_image_by_stamp(rgb_dir, stamp_sec, tolerance)
        if not rgb_path:
            return
        
        img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if img is None:
            return
        
        # ─────────────────────────────────────────
        # 1-1) Find and publish depth image
        # ─────────────────────────────────────────
        depth_path = self._find_image_by_stamp(depth_dir, stamp_sec, tolerance)
        if depth_path:
            # Depth is usually 16-bit PNG (in mm units)
            depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth_img is not None:
                # 16-bit depth
                if depth_img.dtype == np.uint16:
                    depth_msg = self.bridge.cv2_to_imgmsg(depth_img, encoding="16UC1")
                # 32-bit float depth
                elif depth_img.dtype == np.float32:
                    depth_msg = self.bridge.cv2_to_imgmsg(depth_img, encoding="32FC1")
                # Others (e.g., 8-bit)
                else:
                    depth_msg = self.bridge.cv2_to_imgmsg(depth_img, encoding="mono8")
                
                depth_msg.header.stamp = ros_stamp
                depth_msg.header.frame_id = "camera_depth"
                self.depth_pub.publish(depth_msg)
        
        # ─────────────────────────────────────────
        # 2) Publish original RGB image
        # ─────────────────────────────────────────
        msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        msg.header.stamp = ros_stamp
        msg.header.frame_id = "camera_rgb"
        self.rgb_pub.publish(msg)
        # ─────────────────────────────────────────
        # 3-1) Publish projected quadric image
        # ─────────────────────────────────────────
        if initial is not None and pose is not None:
            p3d_img = img.copy()
            h, w = p3d_img.shape[:2]
            
            # pose -> T_c_w conversion
            if hasattr(pose, 'matrix'):
                T_w_c = pose.matrix()
            else:
                T_w_c = pose
            T_c_w = np.linalg.inv(T_w_c)
            
            # Iterate all quadrics
            for key in initial.keys():
                # 1) Check symbol char: skip if not 'q'
                try:
                    sym = Symbol(key)
                except Exception:
                    continue

                # Safely handle chr() return type differences by implementation
                ch = sym.chr()
                if isinstance(ch, int):
                    ch = chr(ch)

                if ch != 'q':   # assume quadrics are created with 'q'
                    continue

                # 2) Get actual ConstrainedDualQuadric
                try:
                    quadric = gtsam_quadrics.ConstrainedDualQuadric.getFromValues(initial, key)
                except Exception:
                    # Skip if quadric does not exist for this key or type is mismatched
                    continue
                
                # Extract quadric attributes
                q_pose = quadric.pose()
                R = q_pose.rotation().matrix()
                ctr = np.array([q_pose.x(), q_pose.y(), q_pose.z()])
                radii = np.array(quadric.radii())
                
                # Get color from quadric_fixed_colors (white if missing)
                col = quadric_fixed_colors.get(key, [255, 255, 255])
                
                # Generate wireframe
                verts, lines = ellipsoid_wireframe_vertices(radii, R, ctr)
                
                # World -> camera coordinate transform
                vc = (T_c_w[:3, :3] @ verts.T + T_c_w[:3, 3:4]).T
                
                # Project only points in front (z > 0)
                front = vc[:, 2] > 0
                if not np.any(front):
                    continue
                
                pts2d = project_pts(vc[front]).astype(int)
                uv = {i: p for i, p in zip(np.where(front)[0], pts2d)}
                
                # Draw wireframe lines
                for i0, i1 in lines:
                    if i0 in uv and i1 in uv:
                        u0, v0 = uv[i0]
                        u1, v1 = uv[i1]
                        if 0 <= u0 < w and 0 <= v0 < h and 0 <= u1 < w and 0 <= v1 < h:
                            cv2.line(p3d_img, (u0, v0), (u1, v1), col, 1)
                
                # Draw center point
                ctr_cam = T_c_w[:3, :3] @ ctr + T_c_w[:3, 3]
                if ctr_cam[2] > 0:
                    ctr2d = project_pts(ctr_cam.reshape(1, 3))[0]
                    ui, vi = int(round(ctr2d[0])), int(round(ctr2d[1]))
                    if 0 <= ui < w and 0 <= vi < h:
                        cv2.circle(p3d_img, (ui, vi), 5, col, -1)
            
            # Publish projected image
            p3d_msg = self.bridge.cv2_to_imgmsg(p3d_img, encoding="bgr8")
            p3d_msg.header.stamp = ros_stamp
            p3d_msg.header.frame_id = "camera_rgb"
            self.quadric_3d_pub.publish(p3d_msg)  # /quadric_projected_image
        # ─────────────────────────────────────────
        # 3) Publish quadric projected bbox image
        # ─────────────────────────────────────────
        if initial is not None and pose is not None:
            proj_img = img.copy()
            h, w = proj_img.shape[:2]
            
            # Convert pose to Pose3 if it is a numpy array
            if not hasattr(pose, 'matrix'):
                pose = gtsam.Pose3(pose)
            
            # Iterate all quadrics
            for key in initial.keys():
                try:
                    sym = Symbol(key)
                except Exception:
                    continue

                ch = sym.chr()
                if isinstance(ch, int):
                    ch = chr(ch)

                if ch != 'q':
                    continue

                try:
                    quadric = gtsam_quadrics.ConstrainedDualQuadric.getFromValues(initial, key)
                except Exception:
                    continue
                
                # ★ Project bbox with QuadricCamera
                try:
                    aligned_box = gtsam_quadrics.QuadricCamera.project(
                        quadric, pose, calibration
                    ).bounds()
                except Exception:
                    continue
                
                # Extract values from AlignedBox2
                xmin = aligned_box.xmin()
                ymin = aligned_box.ymin()
                xmax = aligned_box.xmax()
                ymax = aligned_box.ymax()
                
                # Filter NaN / inf
                vals = [xmin, ymin, xmax, ymax]
                if any((math.isnan(v) or math.isinf(v)) for v in vals):
                    # For debug logs, use print / rospy.logwarn here
                    # rospy.logwarn(f"[PROJ] invalid bbox for q{Symbol(key).index()}: {vals}")
                    continue

                # Only then convert to int + clip
                xmin, ymin = max(0, int(xmin)), max(0, int(ymin))
                xmax, ymax = min(w-1, int(xmax)), min(h-1, int(ymax))

                if xmin >= xmax or ymin >= ymax:
                    continue
                
                # Image bounds check
                xmin, ymin = max(0, int(xmin)), max(0, int(ymin))
                xmax, ymax = min(w-1, int(xmax)), min(h-1, int(ymax))
                
                if xmin >= xmax or ymin >= ymax:
                    continue
                
                # Get color (green if missing)
                col = quadric_fixed_colors.get(key, [0, 255, 0])
                
                # Draw bbox (quadric projection: solid line)
                cv2.rectangle(proj_img, (xmin, ymin), (xmax, ymax), col, 2)
                
                # Show quadric index
                q_idx = sym.index()
                cv2.putText(proj_img, f"q{q_idx}", (xmin, ymin - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
            
            # ─────────────────────────────────────────
            # 4) Draw associated detection bbox (green, dashed style)
            # ─────────────────────────────────────────
            if associated is not None:
                for det in associated:
                    xmin = max(0, int(det.bounds.xmin()))
                    ymin = max(0, int(det.bounds.ymin()))
                    xmax = min(w-1, int(det.bounds.xmax()))
                    ymax = min(h-1, int(det.bounds.ymax()))
                    
                    if xmin >= xmax or ymin >= ymax:
                        continue
                    
                    # Green, thickness 1 (to distinguish from quadric)
                    cv2.rectangle(proj_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
                    
                    # Show associated quadric key
                    if hasattr(det, 'quadric_key') and det.quadric_key is not None:
                        try:
                            q_idx = Symbol(det.quadric_key).index()
                            label = f"det->q{q_idx}"
                        except:
                            label = "assoc"
                    else:
                        label = "assoc"
                    cv2.putText(proj_img, label, (xmin, ymax + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # ─────────────────────────────────────────
            # 5) Draw unassociated detection bbox (red)
            # ─────────────────────────────────────────
            if unassociated is not None:
                for det in unassociated:
                    xmin = max(0, int(det.bounds.xmin()))
                    ymin = max(0, int(det.bounds.ymin()))
                    xmax = min(w-1, int(det.bounds.xmax()))
                    ymax = min(h-1, int(det.bounds.ymax()))
                    
                    if xmin >= xmax or ymin >= ymax:
                        continue
                    
                    # Red
                    cv2.rectangle(proj_img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
                    cv2.putText(proj_img, "new", (xmin, ymax + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            # Publish projected image
            proj_msg = self.bridge.cv2_to_imgmsg(proj_img, encoding="bgr8")
            proj_msg.header.stamp = ros_stamp
            proj_msg.header.frame_id = "camera_rgb"
            self.quadric_proj_pub.publish(proj_msg)
            
    def _find_image_by_stamp(self, directory: str, stamp: float, tolerance: float = 0.05) -> Optional[str]:
        """
        Find the image file in a directory closest to the given timestamp.
        
        Args:
            directory: image directory
            stamp: reference timestamp (seconds, float)
            tolerance: allowed tolerance (seconds)
        """
        if not os.path.isdir(directory):
            return None
        
        best_path = None
        best_diff = float('inf')
        
        for fname in os.listdir(directory):
            if not fname.endswith('.png'):
                continue
            
            base = os.path.splitext(fname)[0]
            try:
                # Example: 1234567890.123.png
                if '.' in base:
                    file_stamp = float(base)
                # Example: 1234567890123456789.png (nanosecond integer)
                else:
                    file_stamp = int(base) / 1e9
            except ValueError:
                continue
            
            diff = abs(file_stamp - stamp)
            if diff < best_diff and diff <= tolerance:
                best_diff = diff
                best_path = os.path.join(directory, fname)
        
        return best_path

    def visualise_source_graph(self, graph):
        """
        Visualize QuadricGraph.
        
        Args:
            graph: QuadricGraph object (with nodes, edges attributes)
        """
        marker_array = MarkerArray()

        # ========== Node visualization ==========
        for key, node in graph.nodes.items():
            marker = Marker()
            marker.header.frame_id = "test"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "source_nodes"
            marker.id = Symbol(key).index()
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            # Set position
            marker.pose.position.x = node.translation[0]
            marker.pose.position.y = node.translation[1]
            marker.pose.position.z = node.translation[2]

            # Set orientation (quaternion: w, x, y, z -> x, y, z, w)
            marker.pose.orientation.x = node.orientation[1]
            marker.pose.orientation.y = node.orientation[2]
            marker.pose.orientation.z = node.orientation[3]
            marker.pose.orientation.w = node.orientation[0]

            # Set scale
            marker.scale.x = 0.02
            marker.scale.y = 0.02
            marker.scale.z = 0.02

            # Set color (gray)
            marker.color.r = 0.5
            marker.color.g = 0.5
            marker.color.b = 0.5
            marker.color.a = 0.8

            marker.lifetime = rospy.Duration(0.4)
            marker_array.markers.append(marker)

        # ========== Edge visualization ==========
        edge_id = 0
        for (key1, key2), edge_data in graph.edges.items():
            node1 = graph.nodes.get(key1)
            node2 = graph.nodes.get(key2)

            if node1 is None or node2 is None:
                continue

            line_marker = Marker()
            line_marker.header.frame_id = "test"
            line_marker.header.stamp = rospy.Time.now()
            line_marker.ns = "source_edges"
            line_marker.id = edge_id
            edge_id += 1
            line_marker.type = Marker.LINE_LIST
            line_marker.action = Marker.ADD

            # Line thickness
            line_marker.scale.x = 0.005

            # Color (blue)
            line_marker.color.r = 0.0
            line_marker.color.g = 0.0
            line_marker.color.b = 1.0
            line_marker.color.a = 1.0

            # Add two points
            p1 = Point()
            p1.x = node1.translation[0]
            p1.y = node1.translation[1]
            p1.z = node1.translation[2]
            line_marker.points.append(p1)

            p2 = Point()
            p2.x = node2.translation[0]
            p2.y = node2.translation[1]
            p2.z = node2.translation[2]
            line_marker.points.append(p2)

            line_marker.lifetime = rospy.Duration(0.4)
            marker_array.markers.append(line_marker)

        # Publish
        self.source_graph_marker_pub.publish(marker_array)    
        

    def visualise_quadric_graph(self, graph_or_values):
        """Visualize the quadric graph."""
        marker_array = MarkerArray()
        edge_marker_array = MarkerArray()
        
        nodes_data = {}
        edges_data = {}

        if isinstance(graph_or_values, gtsam.Values):
            for key in graph_or_values.keys():
                try:
                    quadric = gtsam_quadrics.ConstrainedDualQuadric.getFromValues(
                        graph_or_values, key
                    )
                except RuntimeError:
                    continue

                sym = gtsam.Symbol(key)
                idx = sym.index()  # ★ always use index()

                pose = quadric.pose()
                t = pose.translation()
                q = pose.rotation().toQuaternion()
                radii = quadric.radii()

                if hasattr(t, "x"):
                    tx, ty, tz = t.x(), t.y(), t.z()
                else:
                    tx, ty, tz = float(t[0]), float(t[1]), float(t[2])

                if hasattr(q, "x"):
                    qx, qy, qz, qw = q.x(), q.y(), q.z(), q.w()
                else:
                    qw, qx, qy, qz = map(float, q)

                if hasattr(radii, "x"):
                    rx, ry, rz = radii.x(), radii.y(), radii.z()
                else:
                    rx, ry, rz = float(radii[0]), float(radii[1]), float(radii[2])

                nodes_data[key] = {
                    'index': idx,
                    'translation': (tx, ty, tz),
                    'orientation': (qx, qy, qz, qw),
                    'radii': (rx, ry, rz)
                }

        elif isinstance(graph_or_values, QuadricGraph):
            for key, node in graph_or_values.nodes.items():
                # ★ Always try Symbol.index()
                try:
                    idx = gtsam.Symbol(key).index()
                except:
                    idx = int(key) % 100000  # fallback
                
                tx, ty, tz = node.translation[0], node.translation[1], node.translation[2]
                qw, qx, qy, qz = node.orientation[0], node.orientation[1], node.orientation[2], node.orientation[3]
                rx, ry, rz = node.radii[0], node.radii[1], node.radii[2]

                nodes_data[key] = {
                    'index': idx,
                    'translation': (tx, ty, tz),
                    'orientation': (qx, qy, qz, qw),
                    'radii': (rx, ry, rz)
                }
            
            edges_data = graph_or_values.edges
        else:
            rospy.logerr(f"Unsupported type: {type(graph_or_values)}")
            return

        # ========== Node visualization ==========
        for key, node_info in nodes_data.items():
            tx, ty, tz = node_info['translation']
            qx, qy, qz, qw = node_info['orientation']
            rx, ry, rz = node_info['radii']
            idx = node_info['index']

            marker = Marker()
            marker.header.frame_id = "test"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "quadrics"
            marker.id = idx  # ★ idx is now a small integer
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            marker.pose.position.x = tx
            marker.pose.position.y = ty
            marker.pose.position.z = tz
            marker.pose.orientation.x = qx
            marker.pose.orientation.y = qy
            marker.pose.orientation.z = qz
            marker.pose.orientation.w = qw

            marker.scale.x = max(0.1, rx * 2.0)  # ★ guarantee minimum size
            marker.scale.y = max(0.1, ry * 2.0)
            marker.scale.z = max(0.1, rz * 2.0)

            color = quadric_fixed_colors.get(key, [128, 128, 128])
            if isinstance(color, (list, np.ndarray)) and len(color) >= 3:
                marker.color.r = float(color[0]) / 255.0
                marker.color.g = float(color[1]) / 255.0
                marker.color.b = float(color[2]) / 255.0
            else:
                marker.color.r = 0.5
                marker.color.g = 0.5
                marker.color.b = 0.5
            marker.color.a = 0.9

            marker.lifetime = rospy.Duration(0)  # ★ 0 = permanent
            marker_array.markers.append(marker)

            # Text marker
            text_marker = Marker()
            text_marker.header.frame_id = "test"
            text_marker.header.stamp = rospy.Time.now()
            text_marker.ns = "quadric_ids"
            text_marker.id = idx + 10000
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD

            text_marker.pose.position.x = tx
            text_marker.pose.position.y = ty
            text_marker.pose.position.z = tz + 0.5

            text_marker.scale.z = 0.3
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0

            text_marker.text = str(idx)
            text_marker.lifetime = rospy.Duration(0)
            marker_array.markers.append(text_marker)

        # ========== Edge visualization ==========
        edge_id = 0
        for (key1, key2), distance in edges_data.items():
            if key1 not in nodes_data or key2 not in nodes_data:
                continue

            line_marker = Marker()
            line_marker.header.frame_id = "test"
            line_marker.header.stamp = rospy.Time.now()
            line_marker.ns = "quadric_edges"
            line_marker.id = edge_id
            edge_id += 1
            line_marker.type = Marker.LINE_STRIP
            line_marker.action = Marker.ADD

            line_marker.scale.x = 0.005
            line_marker.color.r = 0.0
            line_marker.color.g = 1.0
            line_marker.color.b = 0.0
            line_marker.color.a = 0.8

            p1 = Point()
            p1.x, p1.y, p1.z = nodes_data[key1]['translation']
            p2 = Point()
            p2.x, p2.y, p2.z = nodes_data[key2]['translation']

            line_marker.points.append(p1)
            line_marker.points.append(p2)
            line_marker.lifetime = rospy.Duration(0)

            edge_marker_array.markers.append(line_marker)

        # ========== Publish ==========
        if marker_array.markers:
            self.graph_marker_array_pub.publish(marker_array)
        
        if edge_marker_array.markers:
            self.graph_edge_pub.publish(edge_marker_array)
        
    def publish_pose_as_odom(
        self,
        pose: Pose3,
        stamp: rospy.Time,
        frame_id: str = "odom",        # world reference frame
        child_frame_id: str = "base_link"  # robot/camera link name
    ) -> None:
        msg = Odometry()
        if hasattr(stamp, "to_sec"):     # when stamp is rospy.Time
            ros_stamp = stamp
            stamp_sec = stamp.to_sec()
        else:                            # when stamp is float/int
            stamp_sec = float(stamp)
            ros_stamp = rospy.Time.from_sec(stamp_sec)
        msg.header.stamp = ros_stamp
        msg.header.frame_id = frame_id
        msg.child_frame_id = child_frame_id

        # --- translation ---
        t = pose.translation()
        if hasattr(t, "x"):  # gtsam.Point3
            tx, ty, tz = t.x(), t.y(), t.z()
        else:                # handle numpy-array case
            tx, ty, tz = float(t[0]), float(t[1]), float(t[2])

        # --- rotation(quat) ---
        q = pose.rotation().toQuaternion()
        if hasattr(q, "x"):  # gtsam quaternion
            qx, qy, qz, qw = q.x(), q.y(), q.z(), q.w()
        else:                # assume array [w,x,y,z]
            qw, qx, qy, qz = map(float, q)

        msg.pose.pose.position.x = tx
        msg.pose.pose.position.y = ty
        msg.pose.pose.position.z = tz

        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw

        # No velocity available now, set to 0
        msg.twist.twist.linear.x = 0.0
        msg.twist.twist.linear.y = 0.0
        msg.twist.twist.linear.z = 0.0
        msg.twist.twist.angular.x = 0.0
        msg.twist.twist.angular.y = 0.0
        msg.twist.twist.angular.z = 0.0

        self.odom_pub.publish(msg)
# =========================
# ROS Param & main
# =========================
def load_camera_param_from_yaml(cam_info_path: str) -> Dict[str, float]:
    """Read camera parameters from an OpenCV YAML file."""
    fs = cv2.FileStorage(cam_info_path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"Cannot open cam_info file: {cam_info_path}")

    def read_real(key: str) -> float:
        node = fs.getNode(key)
        if node.empty():
            raise KeyError(f"Missing '{key}' entry in cam_info: {cam_info_path}")
        return float(node.real())

    try:
        return {
            "fx": read_real("Camera.fx"),
            "fy": read_real("Camera.fy"),
            "cx": read_real("Camera.cx"),
            "cy": read_real("Camera.cy"),
            "skew": 0.0,
            "width": int(read_real("Camera.width")),
            "height": int(read_real("Camera.height")),
            "depth_factor": read_real("DepthMapFactor"),
        }
    finally:
        fs.release()


def build_config_from_ros_params() -> Tuple[Dict[str, Any], str, str]:
    detection_path = str(rospy.get_param("~detection_path", "")).strip()
    odom_path = str(rospy.get_param("~odom_path", "")).strip()
    cam_info_path = str(rospy.get_param("~cam_info", "")).strip()
    base_path = str(rospy.get_param("~base_path", "")).strip()
    output_dir = str(rospy.get_param("~output_dir", "")).strip()
    save_name = str(rospy.get_param("~save_name", "map.json")).strip()

    if not detection_path:
        raise ValueError("Missing required parameter: ~detection_path")
    if not odom_path:
        raise ValueError("Missing required parameter: ~odom_path")
    if not cam_info_path:
        raise ValueError("Missing required parameter: ~cam_info")
    if not output_dir:
        raise ValueError("Missing required parameter: ~output_dir")
    if not save_name:
        raise ValueError("Invalid parameter: ~save_name must not be empty")
    if not save_name.lower().endswith(".json"):
        save_name += ".json"

    if not os.path.isfile(detection_path):
        raise FileNotFoundError(f"Detection file not found: {detection_path}")
    if not os.path.isfile(odom_path):
        raise FileNotFoundError(f"Odometry file not found: {odom_path}")
    if not os.path.isfile(cam_info_path):
        raise FileNotFoundError(f"cam_info file not found: {cam_info_path}")

    cfg: Dict[str, Any] = {
        "detection_json": detection_path,
        "odom_txt": odom_path,
        "base_path": base_path,
        "interval_sec": float(rospy.get_param("~interval_sec", 0.01)),
        "vo_tolerance_sec": float(rospy.get_param("~vo_tolerance_sec", 0.1)),
        "camera_param": load_camera_param_from_yaml(cam_info_path),
    }
    return cfg, output_dir, save_name


def main(cfg: Dict, output_dir: str, save_name: str = "map.json") -> None:
    builder = KeyframeBuilder(
        detection_json=cfg["detection_json"],
        odom_txt=cfg.get("odom_txt"),
        interval_sec=cfg.get("interval_sec", 0.1),
        vo_tolerance_sec=cfg.get("vo_tolerance_sec", 0.02),
        base_path=cfg.get("base_path"),
    )
    keyframes = builder.build()
    print(
        f"[INFO] Built {len(keyframes)} keyframes "
        f"(interval={cfg.get('interval_sec', 0.1)}s)"
    )
    
    camera_param = cfg.get("camera_param")
    calib_rgb = gtsam.Cal3_S2(
        camera_param["fx"],
        camera_param["fy"],
        camera_param["skew"],
        camera_param["cx"],
        camera_param["cy"],
    )
    
    backend = Backend()
    asso = Association(**cfg.get("association_param", {}))

    prev_vo_pose: Optional[Pose3] = None   # Raw absolute VO pose (for measurement)
    prev_id: Optional[int] = None          # Previous accepted graph node id
    node_id = 0                            # Graph node id increments only for accepted frames
    
    
    # Initialize global graph (once)
    global_quadric_graph = QuadricGraph()

    def save_current_map(reason: str = "normal exit") -> None:
        output_dir_path = Path(output_dir).expanduser()
        output_dir_path.mkdir(parents=True, exist_ok=True)
        map_json_path = str(output_dir_path / save_name)
        backend.save_map_to_json(map_json_path, global_quadric_graph)
        print(f"[INFO] map save complete ({reason}): {map_json_path}")

    def save_map_at_exit() -> None:
        try:
            save_current_map("process exit")
        except Exception as exc:
            print(f"[ERROR] failed to save map during process exit: {exc}")

    atexit.register(save_map_at_exit)

    for k, kf in enumerate(keyframes):
        if rospy.is_shutdown():
            print("[INFO] ROS shutdown requested; saving current map before exit.")
            break

        loop_start = time.time()
        
        if kf.vo_t is None or kf.vo_q is None:
            print(f"[WARN] skip idx={kf.index} (no VO)")
            continue
      
        # VO absolute pose (odom frame) -- used for measurement only
        qx, qy, qz, qw = kf.vo_q    
        R = Rot3.Quaternion(qw, qx, qy, qz)
        t = Point3(*kf.vo_t)
        T_vo_abs = Pose3(R, t)
        T_vo_abs = backend.odomToPose3(T_vo_abs)
        i_curr = node_id
        x_curr = ensure_pose_key(X(i_curr))

        if prev_vo_pose is None:
            if not backend.initial.exists(x_curr):
                backend.initial.insert(x_curr, T_vo_abs)
            backend.graph.add(gtsam.PriorFactorPose3(x_curr, T_vo_abs, backend.priorNoise))
            est_for_assoc = T_vo_abs
        else:
            i_prev = prev_id
            x_prev = ensure_pose_key(X(i_prev))
            if not backend.initial.exists(x_prev):
                backend.initial.insert(x_prev, prev_vo_pose)

            z = prev_vo_pose.between(T_vo_abs)
            backend.graph.add(gtsam.BetweenFactorPose3(x_prev, x_curr, z, backend.odomNoise))

            if backend.initial.exists(x_prev):
                est_prev = backend.initial.atPose3(x_prev)
            else:
                est_prev = T_vo_abs
            est_pred = est_prev.compose(z)

     
            backend.initial.insert(x_curr, est_pred)

            est_for_assoc = est_pred
            
        source_graph, stamp = build_source_graph_from_frame(kf, est_for_assoc, calib_rgb)

        # =========================
        # 1) Association
        # =========================
        t_asso_start = time.time()
        associated, unassociated = asso.run(kf, est_for_assoc, camera_param)
        t_asso = time.time() - t_asso_start
        print(f"[ASSOC] idx={kf.index} X({i_curr}) assoc={len(associated)} unassoc={len(unassociated)}")

        # =========================
        # Factor/Stats update
        # =========================
        t_update_start = time.time()
        xkey = x_curr

        # ---------- unassociated ----------
        for uaq in unassociated:
            try:
                qkey = ensure_quadric_key(uaq.quadric_key)
                uaq.quadric_key = qkey
            except Exception:
                continue

            if not backend.initial.exists(qkey):
                gtsam_quadrics.ConstrainedDualQuadric(
                    uaq.ob_pose3,
                    uaq.ob_radii
                ).addToValues(backend.initial, qkey)
            backend.graph.add(
                gtsam_quadrics.BoundingBoxFactor(
                    uaq.bounds, calib_rgb, xkey, qkey, backend.quadricNoise
                )
            )

        # ---------- associated ----------
        for aq in associated:
            try:
                qkey = ensure_quadric_key(aq.quadric_key)
                aq.quadric_key = qkey
            except Exception:
                continue
            
            # Add quadric if missing in initial (safety guard)
            if not backend.initial.exists(qkey):
                gtsam_quadrics.ConstrainedDualQuadric(
                    aq.ob_pose3,
                    aq.ob_radii
                ).addToValues(backend.initial, qkey)
            
            # Add factor
            backend.graph.add(
                gtsam_quadrics.BoundingBoxFactor(
                    aq.bounds, calib_rgb, xkey, qkey, backend.quadricNoise
                )
            )
        t_update = time.time() - t_update_start
        print(f"[UPDATE] idx={kf.index} X({i_curr}) assoc={len(associated)} unassoc={len(unassociated)} time={t_update*1000:.1f}ms")

        # =========================
        # 2) Optimization
        # =========================
        t_opt_start = time.time()
        if node_id % 1 == 0:
            backend.optimize()
        t_opt = time.time() - t_opt_start
        
        PQBsFromValues(backend.initial)
        est = T_vo_abs
        if backend.initial.exists(x_curr):
            est = backend.initial.atPose3(x_curr)
        print(f"[OPT] X({i_curr}) = {est.translation()}")

        # =========================
        # 3) Label statistics update
        # =========================
        t_stats_start = time.time()
        # unassociated: initialize new quadrics
        for uaq in unassociated:
            try:
                qkey = ensure_quadric_key(uaq.quadric_key)
                uaq.quadric_key = qkey
            except Exception:
                continue
            if qkey not in quadric_observation_count:  # only if not initialized yet
                backend.init_quadric_stats(
                    qkey, uaq.labels, uaq.scores, uaq.colors,
                )
        
        # associated: update existing quadrics
        
        for aq in associated:
            try:
                qkey = ensure_quadric_key(aq.quadric_key)
                aq.quadric_key = qkey
            except Exception:
                continue

            if qkey in quadric_observation_count:  # update when already initialized
                backend.update_quadric_stats(
                    qkey, aq.labels, aq.scores, aq.colors,
                )
            else:  # initialize on first occurrence
                backend.init_quadric_stats(
                    qkey, aq.labels, aq.scores, aq.colors,
                )
        t_stats = time.time() - t_stats_start
        print(f"[STATS] idx={kf.index} time={t_stats*1000:.1f}ms")

        # =========================
        # 3) Publish + visualization
        # =========================
        t_pub_start = time.time()
        stamp = rospy.Time.from_sec(kf.stamp)

        # backend.publish_image_by_stamp(
        #     rgb_dir='/root/workspace/dataset_root/rgbd_dataset_freiburg2_desk/rgb',
        #     depth_dir='/root/workspace/dataset_root/rgbd_dataset_freiburg2_desk/depth',
        #     stamp=stamp,
        #     initial=backend.initial,
        #     pose=est, associated=associated, unassociated=unassociated
        # )
        

        backend.publish_pose_as_odom(
            est, stamp, frame_id="test", child_frame_id="odom"
        )

        prev_map_node_count = len(global_quadric_graph.nodes)
        prev_map_edge_count = len(global_quadric_graph.edges)

        global_quadric_graph = accumulate_source_to_global_graph(
            global_quadric_graph, backend.initial, source_graph, associated, unassociated
        )
        curr_map_node_count = len(global_quadric_graph.nodes)
        curr_map_edge_count = len(global_quadric_graph.edges)
        print(
            f"[MAP] before nodes={prev_map_node_count} edges={prev_map_edge_count} "
            f"| after nodes={curr_map_node_count} edges={curr_map_edge_count}"
        )

        backend.visualise_source_graph(source_graph)
        backend.visualise_quadric_graph(global_quadric_graph)
        t_pub = time.time() - t_pub_start

        # Prepare for next loop
        prev_vo_pose = T_vo_abs
        prev_id = i_curr
        node_id += 1

        # =========================
        # ★ Match loop period (0.2s)
        # =========================
        loop_elapsed = time.time() - loop_start
     
        print(
            f"[TIME] idx={kf.index} asso={t_asso*1000:.1f}ms "
            f"opt={t_opt*1000:.1f}ms pub={t_pub*1000:.1f}ms "
            f"total={loop_elapsed*1000:.1f}ms "
        )

    # Save map after loop ends
    save_current_map("normal exit")
    try:
        atexit.unregister(save_map_at_exit)
    except Exception:
        pass
    print("[INFO] done.")
if __name__ == "__main__":
    rospy.init_node("quadric_slam_backend", anonymous=True)
    config, output_dir, save_name = build_config_from_ros_params()
    try:
        main(config, output_dir, save_name)
    except (KeyboardInterrupt, rospy.ROSInterruptException):
        print("[INFO] interrupted by user.")
