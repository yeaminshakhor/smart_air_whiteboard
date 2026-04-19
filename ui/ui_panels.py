"""
ui_panels.py - Obsidian HUD drawn with OpenCV.

Fixes vs original:
  - Color hit-boxes now return ("color", bgr_tuple) - matching _apply_ui_action.
    Previously the inner tuple was stored inconsistently; htype=="color" returned
    hval which was already ("color", bgr), giving a double-wrapped result.
  - Mode hit-boxes return ("mode", mode_key) - unchanged, already correct.
  - handle_click() uses explicit isinstance + tag checks; no len() discrimination.
"""
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

from managers.clipboard_manager import ClipboardManager

PRIMARY = (165, 202, 93)  # #5DCAA5 BGR
SURFACE = (14, 14, 14)
ON_SURFACE = (229, 226, 225)
ON_SURFACE_VARIANT = (194, 202, 189)
OUTLINE_VARIANT = (68, 73, 61)


def draw_glass_panel(frame, x, y, w, h, alpha=0.6, tint=SURFACE):
    if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0] or w <= 0 or h <= 0:
        return
    overlay = frame[y:y + h, x:x + w]
    tint_rect = np.full(overlay.shape, tint, dtype=np.uint8)
    glass = cv2.addWeighted(overlay, 1 - alpha, tint_rect, alpha, 0)
    frame[y:y + h, x:x + w] = glass


def _rounded_rect(img, x, y, w, h, r, color, thickness=-1):
    r = max(0, min(int(r), min(w, h) // 2))
    if w <= 0 or h <= 0:
        return
    if r == 0:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
        return
    if thickness < 0:
        cv2.rectangle(img, (x + r, y), (x + w - r, y + h), color, -1)
        cv2.rectangle(img, (x, y + r), (x + w, y + h - r), color, -1)
        cv2.circle(img, (x + r, y + r), r, color, -1)
        cv2.circle(img, (x + w - r, y + r), r, color, -1)
        cv2.circle(img, (x + r, y + h - r), r, color, -1)
        cv2.circle(img, (x + w - r, y + h - r), r, color, -1)
    else:
        _rounded_rect(img, x, y, w, h, r, color, -1)
        _rounded_rect(img, x + thickness, y + thickness, w - 2 * thickness, h - 2 * thickness, max(0, r - thickness), (0, 0, 0), -1)


class UIManager:
    def __init__(self, w: int, h: int, clipboard: ClipboardManager):
        self.width = w
        self.height = h
        self.clipboard = clipboard

        self.rail_w = 64
        self.rail_h = 280
        self.rail_x = 24
        self.rail_y = self.height // 2 - self.rail_h // 2

        self.hud_w = 280
        self.hud_h = 420
        self.hud_x = self.width - self.hud_w - 24
        self.hud_y = 100

        self.colors: List[Tuple[str, Tuple[int, int, int]]] = [
            ("white", (255, 255, 255)),
            ("red", (0, 0, 255)),
            ("yellow", (0, 255, 255)),
            ("blue", (255, 0, 0)),
            ("green", (93, 202, 165)),
        ]

        self.clipboard_items: List[Dict] = []
        self.hit_boxes: List[Tuple] = []
        self._last_clipboard_version = -1
        self.clipboard_visible = False
        self.palette_visible = True

    def update(self) -> None:
        if self._last_clipboard_version == self.clipboard.version:
            return
        self.clipboard_items = []
        for i, item in enumerate(self.clipboard.get_all_items()):
            if i >= 3:
                break
            if item["type"] == "image" and item["content"] is not None:
                img = item["content"]
                h, img_w = img.shape[:2]
                aspect = img_w / h if h != 0 else 1
                new_h = 60
                new_w = int(new_h * aspect)
                new_w = max(1, min(new_w, self.hud_w - 40))
                item_x = self.hud_x + 20
                item_y = self.hud_y + 160 + i * 80
                self.clipboard_items.append(
                    {
                        "name": f"item_{i}",
                        "value": item,
                        "rect": (item_x, item_y, new_w, new_h),
                    }
                )
        self._last_clipboard_version = self.clipboard.version

    def draw_panels(
        self,
        frame: np.ndarray,
        current_mode: str = "drawing",
        page_text: str = "",
        brush_size: int = 8,
        opacity: float = 1.0,
    ) -> None:
        self.hit_boxes.clear()

        draw_glass_panel(frame, 0, 0, self.width, 56, alpha=0.6, tint=SURFACE)
        cv2.putText(frame, "GestBoard", (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ON_SURFACE, 2)

        nav_x, nav_y, pill_h = 170, 14, 28
        for mode_key, label in [("drawing", "Draw"), ("erasing", "Erase"), ("writing", "Write")]:
            pill_w = 88
            active = current_mode == mode_key
            if active:
                _rounded_rect(frame, nav_x, nav_y, pill_w, pill_h, r=14, color=PRIMARY)
                cv2.circle(frame, (nav_x + 14, nav_y + pill_h // 2), 4, (41, 56, 0), -1)
                cv2.putText(frame, label.upper(), (nav_x + 26, nav_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (41, 56, 0), 1)
            else:
                cv2.putText(frame, label.upper(), (nav_x + 14, nav_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, ON_SURFACE_VARIANT, 1)
            self.hit_boxes.append((nav_x, nav_y, pill_w, pill_h, "mode", mode_key))
            nav_x += pill_w + 10

        cv2.putText(frame, page_text, (nav_x + 20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ON_SURFACE_VARIANT, 1)
        status_w = 170
        status_x = self.width - status_w - 16
        _rounded_rect(frame, status_x, 14, status_w, 28, r=14, color=(10, 10, 10))
        cv2.circle(frame, (status_x + 16, 28), 4, PRIMARY, -1)
        cv2.putText(frame, "HAND DETECTED", (status_x + 28, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, ON_SURFACE_VARIANT, 1)

        draw_glass_panel(frame, self.rail_x, self.rail_y, self.rail_w, self.rail_h, alpha=0.6, tint=SURFACE)

        if self.palette_visible:
            cy = self.rail_y + 30
            for _name, bgr in self.colors:
                cx = self.rail_x + self.rail_w // 2
                cv2.circle(frame, (cx, cy), 11, bgr, -1)
                self.hit_boxes.append((self.rail_x, cy - 14, self.rail_w, 28, "color", bgr))
                cy += 38

        draw_glass_panel(frame, self.hud_x, self.hud_y, self.hud_w, self.hud_h, alpha=0.6, tint=SURFACE)

        cv2.putText(frame, "GESTURE LEGEND", (self.hud_x + 18, self.hud_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, PRIMARY, 1)

        legend = [
            ("DRAW", "Index finger"),
            ("ERASE", "Open palm"),
            ("WRITE", "4 fingers (no thumb)"),
            ("NAV", "Thumbs up/down"),
        ]
        ly = self.hud_y + 58
        for title, sub in legend:
            cv2.putText(frame, title, (self.hud_x + 18, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.48, ON_SURFACE, 1)
            cv2.putText(frame, sub, (self.hud_x + 100, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.37, ON_SURFACE_VARIANT, 1)
            ly += 26

        cv2.putText(frame, "PAGE INDEX", (self.hud_x + 18, self.hud_y + 175), cv2.FONT_HERSHEY_SIMPLEX, 0.45, ON_SURFACE_VARIANT, 1)
        for i in range(4):
            cv2.circle(frame, (self.hud_x + 150 + i * 16, self.hud_y + 171), 4, PRIMARY if i == 1 else (60, 60, 60), -1)

        cv2.putText(frame, "CLIPBOARD", (self.hud_x + 18, self.hud_y + 215), cv2.FONT_HERSHEY_SIMPLEX, 0.45, ON_SURFACE_VARIANT, 1)
        cell, gap = 74, 10
        grid_y = self.hud_y + 230
        slots = [(self.hud_x + 18 + i * (cell + gap), grid_y) for i in range(3)]
        for sx, sy in slots:
            _rounded_rect(frame, sx, sy, cell, cell, r=10, color=(30, 30, 30))

        if self.clipboard_visible:
            for i, item in enumerate(self.clipboard_items[:3]):
                sx, sy = slots[i]
                img = item["value"]["content"]
                if img is None:
                    continue
                thumb = cv2.resize(img, (cell, cell), interpolation=cv2.INTER_AREA)
                roi = frame[sy:sy + cell, sx:sx + cell]
                if roi.shape == thumb.shape:
                    roi[:] = thumb
                self.hit_boxes.append((sx, sy, cell, cell, "clipboard_item", item["value"]))
        else:
            cv2.putText(frame, "Press X to open", (self.hud_x + 22, self.hud_y + 340), cv2.FONT_HERSHEY_SIMPLEX, 0.40, ON_SURFACE_VARIANT, 1)

        paste_y = self.hud_y + self.hud_h - 44
        _rounded_rect(frame, self.hud_x + 18, paste_y, self.hud_w - 36, 28, r=14, color=(20, 20, 20))
        cv2.putText(frame, "PASTE IMAGE", (self.hud_x + 76, paste_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, ON_SURFACE, 1)
        self.hit_boxes.append((self.hud_x + 18, paste_y, self.hud_w - 36, 28, "action", "paste"))

    def is_point_over_ui(self, x: int, y: int) -> bool:
        if y < 56:
            return True
        if self.rail_x <= x <= self.rail_x + self.rail_w and self.rail_y <= y <= self.rail_y + self.rail_h:
            return True
        if self.hud_x <= x <= self.hud_x + self.hud_w and self.hud_y <= y <= self.hud_y + self.hud_h:
            return True
        return False

    def is_over_clipboard(self, x: int, y: int) -> bool:
        if not self.clipboard_visible:
            return False
        return self.hud_x <= x <= self.hud_x + self.hud_w and self.hud_y <= y <= self.hud_y + self.hud_h

    def get_selected_item(self, x: int, y: int) -> Optional[Any]:
        for item in self.clipboard_items:
            ix, iy, iw, ih = item["rect"]
            if ix <= x <= ix + iw and iy <= y <= iy + ih:
                return item["value"]
        return None

    def handle_click(self, pos: Tuple[int, int], select_enabled: bool = True) -> Optional[Any]:
        """
        Returns one of:
          ("mode",  mode_str)          - mode switch
          ("color", (B, G, R))         - color select
          ("size",  int)               - brush size delta
          ("opacity", float)           - opacity delta
          ("action", str)              - named action e.g. "paste"
          dict with type=="image"      - clipboard image item
          None                         - no hit
        """
        if not select_enabled:
            return None
        x, y = pos
        for hx, hy, hw, hh, htype, hval in self.hit_boxes:
            if hx <= x <= hx + hw and hy <= y <= hy + hh:
                if htype == "mode":
                    return ("mode", hval)
                if htype == "color":
                    return ("color", hval)
                if htype == "clipboard_item":
                    return hval
                if htype == "size":
                    return ("size", int(hval))
                if htype == "opacity":
                    return ("opacity", float(hval))
                if htype == "action":
                    return ("action", hval)
        return self.get_selected_item(x, y)

    def show_clipboard(self) -> None:
        self.clipboard_visible = True
        self.update()

    def show_color_palette(self) -> None:
        self.palette_visible = True

    def hide_all(self) -> None:
        self.clipboard_visible = False
        self.palette_visible = False
