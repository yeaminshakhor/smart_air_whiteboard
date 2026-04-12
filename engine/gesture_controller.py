
from typing import Tuple, Optional, List, Any

class GestureController:
    """Handles gesture recognition and mapping to actions."""
    def __init__(self, recognizer, state):
        self.recognizer = recognizer
        self.state = state

    def handle_gesture(
        self,
        gesture: str,
        pos: Tuple[int, int],
        ui,
        canvas,
        stroke_points,
        stroke_processor,
        pages,
        _set_mode,
        _apply_ui_action,
        enable_thumb_nav,
        get_mode,
        get_brush_size,
        _can_navigate_page,
        landmarks=None,
    ) -> None:
        # Gesture-driven mode switching
        if gesture == "index_finger":
            if ui.is_point_over_ui(pos[0], pos[1]) and get_mode() in {"clipboard", "palette"}:
                ui_action = ui.handle_click(pos, select_enabled=True)
                self.state.set("prev_pos", pos)
                _apply_ui_action(ui_action, pos)
                return
            if ui.is_point_over_ui(pos[0], pos[1]) and get_mode() in {"drawing", "erasing"}:
                ui_action = ui.handle_click(pos, select_enabled=True)
                _apply_ui_action(ui_action, pos)
                return
            _set_mode("drawing")
            mode = self.state.get("mode")
            prev_pos = self.state.get("prev_pos")
            brush_size = self.state.get("brush_size", 8)
            if mode == "drawing":
                from utils.geometry import interpolate_stroke
                if prev_pos is not None:
                    points = interpolate_stroke([prev_pos, pos], min_dist=2.0)
                    for i in range(1, len(points)):
                        canvas.draw_line(points[i-1], points[i], (255,255,255,255), max(2, int(brush_size)))
            self.state.set("prev_pos", pos)
            return
        elif gesture == "open_hand":
            _set_mode("erasing")
            mode = self.state.get("mode")
            prev_pos = self.state.get("prev_pos")
            brush_size = self.state.get("brush_size", 8)
            if mode == "erasing":
                from utils.geometry import interpolate_stroke
                if prev_pos is not None:
                    points = interpolate_stroke([prev_pos, pos], min_dist=brush_size * 0.7)
                    for i in range(1, len(points)):
                        canvas.erase_stroke(points[i-1], points[i], radius=max(6, int(brush_size * 1.3)))
                else:
                    canvas.erase(pos, radius=max(6, int(brush_size * 1.3)))
            self.state.set("prev_pos", pos)
            return
        elif gesture == "four":
            _set_mode("writing")
            stroke_points.append(pos)
            while len(stroke_points) > 512:
                stroke_points.pop(0)
            stroke_processor.add_point(pos[0], pos[1])
            self.state.set("prev_pos", pos)
            return
        if gesture in ("thumbs_up", "thumbs_down") and enable_thumb_nav:
            if _can_navigate_page(gesture, pos):
                if gesture == "thumbs_up":
                    pages.prev_page()
                else:
                    pages.next_page()
                canvas.load_page(pages.get_current_page())
            return
