# Smart Air Whiteboard Refactor TODO

## Phase 1: Foundation (State/Config, Dead Code)
- [x] Update managers/state_manager.py: Extend for all state vars (mode, brush_size, opacity, color, prev_pos, grabbing etc.). Add JSON save/load.
- [x] main.py: Replace self.mode/self.brush_* etc. with self.state.get/set. Use only self.config.get. Remove dead self.*.

- [x] Remove dead files: rm core/adaptive_processor.py utils/finger_counter.py utils/geometry.py (confirm unused later).
- [x] All files with "import config": Replace with ConfigManager instance. (Next: Phase 2)

## Phase 2: Architecture Split
- [ ] Create core/gesture_controller.py from main.py _handle_gesture + gesture logic.
- [ ] Create core/drawing_engine.py wrapping canvas ops.
- [ ] Create ui/ui_renderer.py from ui_panels.py draw_panels.
- [ ] Create core/calibration_controller.py.
- [ ] Refactor main.py to thin orchestrator using controllers.

## Phase 3: Performance
- [ ] main.py: Set camera exact res, no resize. Enhance _should_process_frame motion skip.
- [ ] core/canvas_engine.py: Dirty rects in get_canvas_with_items. Invalidate on change.
- [ ] ui/ui_panels.py: Offscreen UI buf, update on state change. Pre-resize thumbs.
- [ ] managers/page_manager.py: Conditional update.

## Phase 4: Gestures/UI
- [ ] core/hand_tracker.py: Conf threshold, safe handedness access.
- [ ] core/gesture_recognizer.py + feature_extractor.py: Fist=erase, distance curls, consume scroll.
- [ ] ui/ui_panels.py: Hover delay, cursor ring, no borders (bg tints), undo toast.
- [ ] main.py: Add feedback.

## Phase 5: Stability
- [ ] main.py: try/except loop, hand_tracker.close().
- [ ] utils/coordinate_mapper.py: Fix dups, Python clip.
- [ ] File I/O: try/except + UI msg.

## Validation
- [ ] pip install pylint flake8
- [ ] pylint . && flake8 .
- [ ] FPS test
- [ ] Gesture/calib test
- [ ] attempt_completion

Progress: Phase 1 complete. Phase 2 next (architecture split).

