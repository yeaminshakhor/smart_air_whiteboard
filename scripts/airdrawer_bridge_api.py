"""FastAPI bridge to reuse Python gesture/mapping logic in AirDrawer."""

from __future__ import annotations

import math
import os
import sys
from typing import List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.gesture_recognizer import GestureRecognizer
from utils.coordinate_mapper import hand_to_canvas


class LandmarkPoint(BaseModel):
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)


class AnalyzePrimaryRequest(BaseModel):
    landmarks: List[LandmarkPoint]
    handedness: Optional[str] = None
    frame_width: int = 1280
    frame_height: int = 720
    canvas_width: int = 1280
    canvas_height: int = 720
    roi_scale: float = 0.6
    sensitivity: float = 1.0
    active_zone_size: int = 0


class CanvasPoint(BaseModel):
    x: int
    y: int


class AnalyzePrimaryResponse(BaseModel):
    raw_gesture: str
    confidence: float
    airdrawer_gesture: str
    canvas_point: CanvasPoint


class AnalyzeSecondaryRequest(BaseModel):
    landmarks: List[LandmarkPoint]
    handedness: Optional[str] = None


class AnalyzeSecondaryResponse(BaseModel):
    raw_gesture: str
    confidence: float
    control_gesture: str
    pinch_delta: float
    angle_delta: float


app = FastAPI(title="AirDrawer Python Bridge", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

recognizer_primary = GestureRecognizer(
    debounce_frames=2,
    smoothing_window=3,
    scroll_threshold_px=30,
    two_finger_distance_ratio=0.08,
    min_frame_confidence=0.55,
    smoothing_min_confidence=0.5,
)

recognizer_secondary = GestureRecognizer(
    debounce_frames=2,
    smoothing_window=3,
    scroll_threshold_px=30,
    two_finger_distance_ratio=0.08,
    min_frame_confidence=0.55,
    smoothing_min_confidence=0.5,
)

secondary_last_pinch_distance: Optional[float] = None
secondary_last_hand_angle: Optional[float] = None


def _to_airdrawer_gesture(raw: str) -> str:
    if raw == "index_finger":
        return "DRAW"
    if raw == "open_hand":
        return "ERASE"
    if raw == "fist":
        return "CLEAR"
    if raw in ("peace", "three", "four_fingers"):
        return "MOVE"
    return "IDLE"


def _to_control_gesture(raw: str) -> str:
    if raw == "peace":
        return "CTRL_MOVE"
    if raw == "open_hand":
        return "CTRL_ROTATE"
    return "CTRL_IDLE"


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/analyze-primary", response_model=AnalyzePrimaryResponse)
def analyze_primary(payload: AnalyzePrimaryRequest) -> AnalyzePrimaryResponse:
    if len(payload.landmarks) < 21:
        return AnalyzePrimaryResponse(
            raw_gesture="unknown",
            confidence=0.0,
            airdrawer_gesture="IDLE",
            canvas_point=CanvasPoint(x=0, y=0),
        )

    frame_w = max(1, int(payload.frame_width))
    frame_h = max(1, int(payload.frame_height))
    canvas_w = max(1, int(payload.canvas_width))
    canvas_h = max(1, int(payload.canvas_height))

    points_px = [
        (int(pt.x * frame_w), int(pt.y * frame_h))
        for pt in payload.landmarks
    ]

    raw_gesture, confidence = recognizer_primary.recognize(points_px, payload.handedness)

    index_x, index_y = points_px[8]
    mapped_x, mapped_y = hand_to_canvas(
        index_x,
        index_y,
        (frame_h, frame_w, 3),
        (canvas_w, canvas_h),
        roi_scale=float(payload.roi_scale),
        sensitivity=float(payload.sensitivity),
        active_zone_size=int(payload.active_zone_size),
    )

    return AnalyzePrimaryResponse(
        raw_gesture=raw_gesture,
        confidence=float(confidence),
        airdrawer_gesture=_to_airdrawer_gesture(raw_gesture),
        canvas_point=CanvasPoint(x=int(mapped_x), y=int(mapped_y)),
    )


@app.post("/analyze-secondary", response_model=AnalyzeSecondaryResponse)
def analyze_secondary(payload: AnalyzeSecondaryRequest) -> AnalyzeSecondaryResponse:
    global secondary_last_pinch_distance, secondary_last_hand_angle

    if len(payload.landmarks) < 21:
        secondary_last_pinch_distance = None
        secondary_last_hand_angle = None
        return AnalyzeSecondaryResponse(
            raw_gesture="unknown",
            confidence=0.0,
            control_gesture="CTRL_IDLE",
            pinch_delta=0.0,
            angle_delta=0.0,
        )

    points_px = [(int(pt.x * 1280), int(pt.y * 720)) for pt in payload.landmarks]
    raw_gesture, confidence = recognizer_secondary.recognize(points_px, payload.handedness)

    thumb_tip = payload.landmarks[4]
    index_tip = payload.landmarks[8]
    wrist = payload.landmarks[0]
    middle_base = payload.landmarks[9]

    pinch_dist = math.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)
    hand_angle = math.atan2(middle_base.y - wrist.y, middle_base.x - wrist.x)

    pinch_delta = 0.0
    angle_delta = 0.0
    control_gesture = _to_control_gesture(raw_gesture)

    # Keep secondary semantics aligned with AirDrawer control logic.
    if pinch_dist < 0.06:
        control_gesture = "CTRL_SCALE"
        if secondary_last_pinch_distance is not None:
            pinch_delta = float(pinch_dist - secondary_last_pinch_distance)
        secondary_last_pinch_distance = float(pinch_dist)
        secondary_last_hand_angle = None
    elif control_gesture == "CTRL_ROTATE":
        if secondary_last_hand_angle is not None:
            angle_delta = float(hand_angle - secondary_last_hand_angle)
            if abs(angle_delta) > math.pi:
                angle_delta = 0.0
        secondary_last_hand_angle = float(hand_angle)
        secondary_last_pinch_distance = None
    else:
        secondary_last_pinch_distance = None
        secondary_last_hand_angle = None

    return AnalyzeSecondaryResponse(
        raw_gesture=raw_gesture,
        confidence=float(confidence),
        control_gesture=control_gesture,
        pinch_delta=pinch_delta,
        angle_delta=angle_delta,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8765)
