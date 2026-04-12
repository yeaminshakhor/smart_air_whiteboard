"""
CameraThread - runs MediaPipe inference on a dedicated daemon thread.

Key fixes vs original:
  - Frame is copied before writing into shared state so the main thread never
    reads a half-written array.
  - `running` uses a threading.Event so stop() is race-free.
  - CAP_PROP_BUFFERSIZE=1 ensures we always get the freshest frame.
"""
import threading

import cv2
from .hand_tracker import HandTracker


class CameraThread(threading.Thread):
    def __init__(self, camera_index: int, width: int, height: int, hand_tracker: HandTracker):
        super().__init__(daemon=True)
        self.camera = cv2.VideoCapture(camera_index)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # Minimise the driver's internal buffer so we always read the newest frame.
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.hand_tracker = hand_tracker
        self._lock = threading.Lock()
        self._state = {"frame": None, "landmarks": None, "handedness": None}
        self._running = threading.Event()
        self._running.set()

    def run(self) -> None:
        while self._running.is_set():
            ok, frame = self.camera.read()
            if not ok or frame is None:
                continue
            frame = cv2.flip(frame, 1)
            landmarks, handedness = self.hand_tracker.process_frame(frame)
            frame_copy = frame.copy()  # safe snapshot for main thread
            with self._lock:
                self._state["frame"] = frame_copy
                self._state["landmarks"] = landmarks
                self._state["handedness"] = handedness

    def get_state(self):
        """Return (frame, landmarks, handedness) atomically."""
        with self._lock:
            return (
                self._state["frame"],
                self._state["landmarks"],
                self._state["handedness"],
            )

    def stop(self) -> None:
        self._running.clear()
        self.camera.release()
