import cv2
import numpy as np
import copy
from .text_renderer import render_text
from collections import deque

class CanvasEngine:
    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height
        self.canvas = np.zeros((height, width, 4), dtype=np.uint8)
        self.items = []
        self._next_item_id = 1
        self.MAX_HISTORY = 20
        self._history = deque([self._copy_state((self.canvas, self.items))], maxlen=self.MAX_HISTORY)
        self._history_index = 0
        self._dirty = True
        self._cached_composite = None
        import logging; logging.getLogger(__name__).info("Initializing CanvasEngine")

    def draw_line(self, start_pos, end_pos, color, thickness):
        """Draws a line on the base canvas."""
        if start_pos:
            p1 = (int(round(start_pos[0])), int(round(start_pos[1])))
            p2 = (int(round(end_pos[0])), int(round(end_pos[1])))
            cv2.line(self.canvas, p1, p2, color, thickness)
            self._dirty = True

    def add_item(self, item):
        """Adds a movable item (text, image) to the canvas."""
        if 'id' not in item:
            item['id'] = self._next_item_id
            self._next_item_id += 1
        self.items.append(item)
        self.record_history()

    def get_item_at(self, pos):
        """Finds the top-most item at a given position."""
        for item in reversed(self.items):
            x, y, w, h = self._get_item_bounds(item)
            if x <= pos[0] <= x + w and y <= pos[1] <= y + h:
                return item
        return None

    def remove_item(self, item_to_remove):
        """Removes a specific item from the canvas."""
        target_id = item_to_remove.get('id')
        if target_id is not None:
            self.items = [item for item in self.items if item.get('id') != target_id]
        else:
            self.items = [item for item in self.items if item is not item_to_remove]
        self._dirty = True

    def erase(self, pos, radius):
        """Erases both the base canvas and any items at the position."""
        # Erase from the base canvas
        cv2.circle(self.canvas, pos, radius, (0, 0, 0, 0), -1)
        self._dirty = True

        # Erase items
        items_to_remove = []
        for item in self.items:
            x, y, w, h = self._get_item_bounds(item)
            # Simple bounding box collision for now
            if self._rect_intersects_circle((x, y, w, h), pos, radius):
                items_to_remove.append(item)
        
        for item in items_to_remove:
            self.remove_item(item)

    def erase_stroke(self, start_pos, end_pos, radius):
        """Erases continuously between points to avoid dotted erasing."""
        if start_pos is None:
            self.erase(end_pos, radius)
            return

        p1 = (int(round(start_pos[0])), int(round(start_pos[1])))
        p2 = (int(round(end_pos[0])), int(round(end_pos[1])))
        cv2.line(self.canvas, p1, p2, (0, 0, 0, 0), radius * 2)
        self._dirty = True

        # Remove items intersecting the eraser stroke segment.
        for item in self.items[:]:
            if self._line_intersects_rect(p1, p2, self._get_item_bounds(item), padding=radius):
                self.remove_item(item)

        self.erase(p2, radius)

    @property
    def is_dirty(self) -> bool:
        return self._dirty

    def get_canvas_with_items(self):
        """Renders items on top of the base canvas and returns the result."""
        if not self._dirty and self._cached_composite is not None:
            return self._cached_composite

        self._cached_composite = self.canvas.copy()
        for item in self.items:
            self._render_item(self._cached_composite, item)
        self._dirty = False
        return self._cached_composite

    def clear_canvas(self):
        """Clears both the base canvas and all items."""
        self.canvas.fill(0)
        self.items.clear()
        self._dirty = True
        self.record_history()

    def undo(self):
        """Reverts to the previous state in history."""
        if self._history_index > 0:
            self._history_index -= 1
            self.canvas, self.items = self._copy_state(list(self._history)[self._history_index])
            self._dirty = True

    def record_history(self):
        """Saves the current canvas and item state to the history."""
        self._dirty = True
        # Truncate forward history on new action after undo
        history_list = list(self._history)
        history_list = history_list[: self._history_index + 1]
        history_list.append(self._copy_state((self.canvas, self.items)))
        self._history = deque(history_list, maxlen=self.MAX_HISTORY)
        self._history_index = len(self._history) - 1

    def paste_image(self, img, x, y):
        if img is None: return
        h, w = img.shape[:2]
        ch, cw = self.canvas.shape[:2]
        
        # Calculate canvas cropping
        x_start = max(0, x - w // 2)
        y_start = max(0, y - h // 2)
        x_end = min(cw, x + w // 2)
        y_end = min(ch, y + h // 2)
        
        # Calculate image cropping using the fix
        img_x_start = max(0, (w // 2) - x)
        img_y_start = max(0, (h // 2) - y)
        img_x_end = img_x_start + (x_end - x_start)
        img_y_end = img_y_start + (y_end - y_start)
        
        if x_start < x_end and y_start < y_end:
            # We copy to base canvas
            self.canvas[y_start:y_end, x_start:x_end] = img[img_y_start:img_y_end, img_x_start:img_x_end]
            self._dirty = True

    def _render_item(self, canvas, item):
        """Renders a single item onto a canvas."""
        item_type = item.get('type')
        content = item.get('content')
        pos = item.get('pos')
        if pos is None:
            return

        if item_type == 'text':
            font_scale = float(item.get('font_scale', 1.0))
            color = item.get('color', (255, 255, 255, 255))
            thickness = int(item.get('thickness', 2))
            if item.get('is_grabbed'):
                font_scale *= 1.1
                thickness = max(2, thickness + 1)
            cv2.putText(canvas, content, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        elif item_type == 'image' and content is not None:
            h, w = content.shape[:2]
            x, y = pos

            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(self.width, x + w)
            y2 = min(self.height, y + h)
            if x1 >= x2 or y1 >= y2:
                return

            img_x1 = x1 - x
            img_y1 = y1 - y
            img_x2 = img_x1 + (x2 - x1)
            img_y2 = img_y1 + (y2 - y1)

            overlay = content[img_y1:img_y2, img_x1:img_x2]
            dst = canvas[y1:y2, x1:x2]

            if overlay.shape[2] == 4:
                alpha = overlay[:, :, 3:4].astype(np.float32) / 255.0
                dst[:, :, :3] = (alpha * overlay[:, :, :3] + (1.0 - alpha) * dst[:, :, :3]).astype(np.uint8)
                dst[:, :, 3] = np.maximum(dst[:, :, 3], overlay[:, :, 3])
            else:
                dst[:, :, :3] = overlay[:, :, :3]
                dst[:, :, 3] = 255

        if item.get('is_grabbed'):
            x, y, w, h = self._get_item_bounds(item)
            cv2.rectangle(canvas, (x - 2, y - 2), (x + w + 2, y + h + 2), (0, 255, 255, 255), 2)

    def _get_item_bounds(self, item):
        """Calculates the bounding box of an item."""
        item_type = item.get('type')
        content = item.get('content')
        pos = item.get('pos')

        if item_type == 'text':
            font_scale = float(item.get('font_scale', 1.0))
            thickness = int(item.get('thickness', 2))
            (w, h), _ = cv2.getTextSize(content, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            return (pos[0], pos[1] - h, w, h)
        elif item_type == 'image' and content is not None:
            h, w = content.shape[:2]
            return (pos[0], pos[1], w, h)
        return (0, 0, 0, 0)

    def _rect_intersects_circle(self, rect, circle_center, circle_radius):
        """Checks if a rectangle and circle intersect."""
        rx, ry, rw, rh = rect
        cx, cy = circle_center
        
        # Find the closest point on the rect to the center of the circle
        closest_x = np.clip(cx, rx, rx + rw)
        closest_y = np.clip(cy, ry, ry + rh)

        distance_x = cx - closest_x
        distance_y = cy - closest_y

        return (distance_x**2 + distance_y**2) < (circle_radius**2)

    def _line_intersects_rect(self, p1, p2, rect, padding=0):
        """Checks whether a segment intersects a rectangle (optionally expanded by padding)."""
        rx, ry, rw, rh = rect
        if rw <= 0 or rh <= 0:
            return False

        rx -= int(padding)
        ry -= int(padding)
        rw += int(padding) * 2
        rh += int(padding) * 2

        x1, y1 = p1
        x2, y2 = p2

        if self._point_in_rect((x1, y1), (rx, ry, rw, rh)) or self._point_in_rect((x2, y2), (rx, ry, rw, rh)):
            return True

        r1 = (rx, ry)
        r2 = (rx + rw, ry)
        r3 = (rx + rw, ry + rh)
        r4 = (rx, ry + rh)

        return (
            self._segments_intersect((x1, y1), (x2, y2), r1, r2)
            or self._segments_intersect((x1, y1), (x2, y2), r2, r3)
            or self._segments_intersect((x1, y1), (x2, y2), r3, r4)
            or self._segments_intersect((x1, y1), (x2, y2), r4, r1)
        )

    def _point_in_rect(self, point, rect):
        x, y = point
        rx, ry, rw, rh = rect
        return rx <= x <= rx + rw and ry <= y <= ry + rh

    def _segments_intersect(self, a1, a2, b1, b2):
        def orient(p, q, r):
            return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

        def on_segment(p, q, r):
            return min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])

        o1 = orient(a1, a2, b1)
        o2 = orient(a1, a2, b2)
        o3 = orient(b1, b2, a1)
        o4 = orient(b1, b2, a2)

        if (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0):
            return True

        if o1 == 0 and on_segment(a1, b1, a2):
            return True
        if o2 == 0 and on_segment(a1, b2, a2):
            return True
        if o3 == 0 and on_segment(b1, a1, b2):
            return True
        if o4 == 0 and on_segment(b1, a2, b2):
            return True

        return False

    def _copy_state(self, state):
        """Deep copies a state tuple."""
        canvas, items = state
        return canvas.copy(), copy.deepcopy(items)

    def get_region(self, x, y, w, h):
        """Extracts a region from the canvas."""
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(self.width, x + w), min(self.height, y + h)
        if x1 >= x2 or y1 >= y2:
            return None
        return self.get_canvas_with_items()[y1:y2, x1:x2].copy()

    def snapshot_item(self, item):
        """Return a clipped BGRA snapshot of an item's current visual bounds."""
        x, y, w, h = self._get_item_bounds(item)
        return self.get_region(x, y, w, h)

    def load_page(self, page_content):
        """Loads page content onto the canvas."""
        if page_content is not None:
            loaded = page_content.copy()
            if len(loaded.shape) == 2:
                loaded = cv2.cvtColor(loaded, cv2.COLOR_GRAY2BGRA)
            elif loaded.shape[2] == 3:
                loaded = cv2.cvtColor(loaded, cv2.COLOR_BGR2BGRA)
            self.canvas = cv2.resize(loaded, (self.width, self.height))
            self.items = []

    def render_text(self, text, pos, font_scale, color):
        """Renders text on the canvas."""
        render_text(self.canvas, text, pos, font_scale, color)

