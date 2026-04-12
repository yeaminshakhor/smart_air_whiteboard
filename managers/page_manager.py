
import os
import sys
import logging
try:
    import config
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config
import numpy as np
import cv2
from typing import List, Optional

_log = logging.getLogger(__name__)

class PageManager:
    def __init__(self, width: int, height: int, save_dir: str = 'data/saved_pages'):
        self.width = width
        self.height = height
        self.save_dir = save_dir
        self.pages: List[np.ndarray] = []
        self.current_index: int = -1

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
        self.load_pages()

        if not self.pages:
            self.add_page()

    def add_page(self) -> np.ndarray:
        """Adds a new blank page."""
        blank_page = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        self.pages.append(blank_page)
        self.current_index = len(self.pages) - 1
        return self.pages[self.current_index]

    def next_page(self) -> np.ndarray:
        """Moves to the next page, or creates one if at the end."""
        if self.current_index >= len(self.pages) - 1:
            self.add_page()
        else:
            self.current_index += 1
        return self.pages[self.current_index]

    def prev_page(self) -> np.ndarray:
        """Moves to the previous page."""
        if self.current_index > 0:
            self.current_index -= 1
        return self.pages[self.current_index]

    def get_current_page(self) -> Optional[np.ndarray]:
        """Returns the current page canvas."""
        if 0 <= self.current_index < len(self.pages):
            return self.pages[self.current_index]
        return None

    def update_current_page(self, canvas: np.ndarray) -> None:
        """Updates the content of the current page."""
        if 0 <= self.current_index < len(self.pages):
            self.pages[self.current_index] = canvas.copy()

    def save_page(self, index: int) -> bool:
        """Saves a specific page to a file."""
        if not (0 <= index < len(self.pages)):
            return False
        page_path = os.path.join(self.save_dir, f"page_{index}.png")
        ok = cv2.imwrite(page_path, self.pages[index])
        if not ok:
            _log.error("Failed to write page %d to %s", index, page_path)
        else:
            import logging; logging.getLogger(__name__).info(f"Page {index} saved to {page_path}")
        return ok

    def save_all_pages(self) -> None:
        """Saves all pages to files."""
        for index in range(len(self.pages)):
            self.save_page(index)

    def load_pages(self) -> None:
        """Loads all pages from the save directory."""
        if not os.path.exists(self.save_dir):
            return
            
        files = sorted(os.listdir(self.save_dir))
        png_files = [f for f in files if f.startswith('page_') and f.endswith('.png')]
        
        if not png_files:
            return

        for f in png_files:
            page_path = os.path.join(self.save_dir, f)
            page = cv2.imread(page_path, cv2.IMREAD_UNCHANGED)
            if page is not None:
                if page.shape[2] == 3:
                    page = cv2.cvtColor(page, cv2.COLOR_BGR2BGRA)
                
                # Ensure loaded page has the correct dimensions
                page = cv2.resize(page, (self.width, self.height))
                self.pages.append(page)
        
        if self.pages:
            self.current_index = 0
            import logging; logging.getLogger(__name__).info(f"Loaded {len(self.pages)} pages.")

    def get_current_canvas(self):
        """Return current page's canvas"""
        return self.get_current_page()

