import math
from typing import Dict, List, Tuple

import numpy as np


class FeatureExtractor:
    """Extract geometric hand features from landmarks."""

    @staticmethod
    def _joint_angle(a: Tuple[int, int], b: Tuple[int, int], c: Tuple[int, int]) -> float:
        ba = (a[0] - b[0], a[1] - b[1])
        bc = (c[0] - b[0], c[1] - b[1])

        mag_ba = math.hypot(ba[0], ba[1])
        mag_bc = math.hypot(bc[0], bc[1])
        if mag_ba == 0.0 or mag_bc == 0.0:
            return 0.0

        dot = ba[0] * bc[0] + ba[1] * bc[1]
        cos_theta = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
        return math.degrees(math.acos(cos_theta))

    @classmethod
    def get_curl_percentage(cls, landmarks: List[Tuple[int, int]]) -> Dict[str, float]:
        if not landmarks or len(landmarks) < 21:
            return {"thumb": 100.0, "index": 100.0, "middle": 100.0, "ring": 100.0, "pinky": 100.0}

        finger_joints = {
            "thumb": (1, 2, 3, 4),
            "index": (5, 6, 7, 8),
            "middle": (9, 10, 11, 12),
            "ring": (13, 14, 15, 16),
            "pinky": (17, 18, 19, 20),
        }

        curls: Dict[str, float] = {}
        for name, (mcp_idx, pip_idx, dip_idx, tip_idx) in finger_joints.items():
            mcp = landmarks[mcp_idx]
            pip = landmarks[pip_idx]
            dip = landmarks[dip_idx]
            tip = landmarks[tip_idx]

            pip_angle = cls._joint_angle(mcp, pip, dip)
            dip_angle = cls._joint_angle(pip, dip, tip)
            straightness = (pip_angle + dip_angle) / 2.0

            curl = ((180.0 - straightness) / 120.0) * 100.0
            curls[name] = float(max(0.0, min(100.0, curl)))

        return curls

    @staticmethod
    def get_palm_width(landmarks: List[Tuple[int, int]]) -> float:
        if not landmarks or len(landmarks) < 21:
            return 0.0
        return float(np.linalg.norm(np.array(landmarks[17]) - np.array(landmarks[5])))

    @staticmethod
    def get_finger_separation_ratio(landmarks: List[Tuple[int, int]]) -> Dict[str, float]:
        if not landmarks or len(landmarks) < 21:
            return {"index_middle": 1.0}

        palm_width = max(1.0, FeatureExtractor.get_palm_width(landmarks))
        idx_tip = np.array(landmarks[8])
        mid_tip = np.array(landmarks[12])
        separation = float(np.linalg.norm(idx_tip - mid_tip) / palm_width)
        return {"index_middle": separation}
