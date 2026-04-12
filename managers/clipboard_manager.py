import os
import sys
import time
import uuid
from core.config_manager import ConfigManager
config_manager = ConfigManager()
import json
from typing import List, Dict, Any, Optional

class ClipboardManager:
    def __init__(self):
        self.clipboard: Dict[str, Any] = {"items": [], "selected_index": -1}
        self._version = 0
        import logging; logging.getLogger(__name__).info("Initializing ClipboardManager")

    @property
    def version(self) -> int:
        return self._version

    def add_item(self, item_type: str, content: Any, metadata: Dict = None):
        """Add to clipboard"""
        safe_metadata = dict(metadata) if metadata else {}
        item: Dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "type": item_type,
            "content": content,
            "thumbnail": self._generate_thumbnail(content),
            "timestamp": int(time.time()),
            "metadata": safe_metadata
        }
        self.clipboard["items"].insert(0, item)
        MAX_ITEMS = config_manager.get('clipboard', 'MAX_ITEMS', 10)
        if len(self.clipboard["items"]) > MAX_ITEMS:
            self.clipboard["items"].pop()
        self._version += 1

    def get_item(self, index: int) -> Optional[Dict[str, Any]]:
        """Retrieve specific item"""
        if 0 <= index < len(self.clipboard["items"]):
            return self.clipboard["items"][index]
        return None

    def get_all_items(self) -> List[Dict[str, Any]]:
        """Return all items for UI"""
        return self.clipboard["items"]

    def select_next(self):
        """Navigation"""
        if len(self.clipboard["items"]) > 0:
            self.clipboard["selected_index"] = (self.clipboard["selected_index"] + 1) % len(self.clipboard["items"])

    def select_previous(self):
        """Navigation"""
        if len(self.clipboard["items"]) > 0:
            self.clipboard["selected_index"] = (self.clipboard["selected_index"] - 1 + len(self.clipboard["items"])) % len(self.clipboard["items"])

    def get_selected_item(self) -> Optional[Dict[str, Any]]:
        """Return currently selected"""
        if self.clipboard["selected_index"] != -1:
            return self.get_item(self.clipboard["selected_index"])
        return None

    def clear_clipboard(self):
        """Remove all items"""
        self.clipboard = {"items": [], "selected_index": -1}
        self._version += 1

    def save_to_file(self, filepath: str):
        """Persistence"""
        save_data = {"items": [], "selected_index": self.clipboard["selected_index"]}
        for item in self.clipboard["items"]:
            save_data["items"].append({**item, 'content': None, 'thumbnail': None})
        with open(filepath, 'w') as f:
            json.dump(save_data, f)

    def load_from_file(self, filepath: str):
        """Persistence"""
        try:
            with open(filepath, 'r') as f:
                self.clipboard = json.load(f)
                self._version += 1
        except FileNotFoundError:
            pass

    def _generate_thumbnail(self, content: Any) -> Optional[Any]:
        # Thumbnail generation is implemented in _generate_thumbnail
        return None

    def _get_timestamp(self) -> int:
        import time
        return int(time.time())
