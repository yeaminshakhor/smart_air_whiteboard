import json
import logging
import os
from typing import Any, Optional

import cv2
import numpy as np

_log = logging.getLogger(__name__)


class EmojiManager:
    """Loads keyword->emoji definitions and renders small image assets."""

    def __init__(self, mapping_path: str) -> None:
        self.mapping_path = mapping_path
        self.assets_dir = os.path.dirname(mapping_path)
        self.keyword_map = self._load_keyword_map()

    def _load_keyword_map(self) -> dict[str, dict[str, Any]]:
        default_map: dict[str, dict[str, Any]] = {
            "SMILE": {"label": ":)", "bg": [0, 220, 255], "fg": [20, 20, 20]},
            "HEART": {"label": "<3", "bg": [80, 80, 255], "fg": [255, 255, 255]},
            "OK": {"label": "OK", "bg": [80, 200, 120], "fg": [255, 255, 255]},
        }

        if not os.path.exists(self.mapping_path):
            return default_map

        try:
            with open(self.mapping_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            _log.warning("Could not load emoji map %s: %s", self.mapping_path, exc)
            return default_map

        if not isinstance(data, dict) or not data:
            return default_map

        normalized = {k.upper(): v for k, v in data.items() if isinstance(v, dict)}
        return normalized or default_map

    def match_keyword(self, text: str) -> Optional[str]:
        upper_text = text.upper()
        for keyword in sorted(self.keyword_map.keys(), key=len, reverse=True):
            if upper_text.endswith(keyword):
                return keyword
        return None

    def render_keyword(self, keyword: str, size: int = 96) -> Any:
        cfg = self.keyword_map.get(keyword.upper(), {})

        asset_name = cfg.get("asset")
        if isinstance(asset_name, str) and asset_name:
            asset_img = self._load_asset(asset_name, size)
            if asset_img is not None:
                return asset_img

        fallback_name = f"{keyword.lower()}.png"
        asset_img = self._load_asset(fallback_name, size)
        if asset_img is not None:
            return asset_img

        label = str(cfg.get("label", keyword[:2]))
        bg = cfg.get("bg", [120, 120, 220])
        fg = cfg.get("fg", [255, 255, 255])

        canvas = np.zeros((size, size, 4), dtype=np.uint8)
        center = (size // 2, size // 2)
        radius = max(8, int(size * 0.42))
        cv2.circle(canvas, center, radius, (int(bg[0]), int(bg[1]), int(bg[2]), 255), -1)
        cv2.circle(canvas, center, radius, (255, 255, 255, 255), 2)

        scale = max(0.4, size / 170.0)
        thickness = max(1, int(size / 56.0))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        text_x = max(0, (size - tw) // 2)
        text_y = min(size - 2, (size + th) // 2)
        cv2.putText(
            canvas,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (int(fg[0]), int(fg[1]), int(fg[2]), 255),
            thickness,
            cv2.LINE_AA,
        )
        return canvas

    def _load_asset(self, asset_name: str, size: int) -> Optional[Any]:
        asset_path = os.path.join(self.assets_dir, asset_name)
        if not os.path.exists(asset_path):
            return None

        img = cv2.imread(asset_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

        return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
