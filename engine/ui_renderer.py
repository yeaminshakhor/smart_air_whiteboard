class UIRenderer:
    """Handles UI rendering and off-screen caching."""
    def __init__(self, ui_manager, state):
        self.ui_manager = ui_manager
        self.state = state
        self.ui_buffer = None
        self.last_state = None

    def render(self, frame, force_redraw=False):
        # TODO: Implement UI rendering with caching
        pass
