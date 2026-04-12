"""
Application State
"""

import os
import sys

try:
    import config
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config
from typing import Dict, Any, List, Callable
import logging
import os
import json

class StateManager:
    def __init__(self, save_path: str = "data/state.json"):
        self.save_path = save_path
        self.state: Dict[str, Any] = {
            "mode": "drawing",
            "tool": "brush",
            "color": (255, 255, 255),
            "brush_size": 8,
            "brush_opacity": 1.0,
            "current_page": 0,
            "is_drawing": False,
            "prev_pos": None,
            "last_position": None,
            "clipboard_visible": False,
            "palette_visible": True,
            "context_menu_visible": False,
            "recording": False,
            "grabbing": False,
            "active_grab": None,
            "grab_offset": (0, 0),
            "grab_path": [],
            "last_copy_ms": 0,
            "last_nav_ms": 0,
            "debug_overlay": True,
        }
        self.observers: List[callable] = []
        self._load()
        import logging
        logging.getLogger(__name__).info("StateManager initialized and loaded from %s", save_path)

    def _load(self):
        import json, os
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, 'r') as f:
                    data = json.load(f)
                    self.state.update({k: v for k, v in data.items() if k in self.state})
            except Exception as e:
                logging.getLogger(__name__).warning("Failed to load state: %s", e)

    def _save(self):
        import json, os
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        try:
            with open(self.save_path, 'w') as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            logging.getLogger(__name__).warning("Failed to save state: %s", e)

    def get(self, key: str, default=None) -> Any:
        return self.state.get(key, default)

    def set(self, key: str, value: Any, save: bool = True):
        old = self.state.get(key)
        self.state[key] = value
        self.notify_observers(key, value, old)
        if save:
            self._save()

    def register_observer(self, observer):
        self.observers.append(observer)

    def notify_observers(self, key: str, value: Any, old_value=None):
        for obs in self.observers[:]:  # Copy to avoid mod during iter
            try:
                obs(key, value, old_value)
            except:
                pass

