# Design System Specification: The Obsidian Instrument

## 1. Overview & Creative North Star
**Creative North Star: "The Silent Navigator"**
This design system is engineered to be a sophisticated, unobtrusive overlay for real-time gesture control and media interaction. It rejects the "app-in-a-box" mentality in favor of a **Studio Instrument** aesthetic. The goal is to create a digital HUD (Heads-Up Display) that feels like it is etched into a pane of dark glass, floating effortlessly over a live camera feed.

By utilizing high-contrast typography against deep, translucent surfaces, we achieve a "Technical Premium" look. We break the rigid grid through **intentional floating modules**, where elements aren't trapped in sidebars but breathe as independent, functional "nodes."

---

## 2. Colors
The palette is rooted in the absence of light, using a single high-energy teal to signal intent and action.

### The Palette (Material Design Tokens)
*   **Background / Surface:** `#131313` (Deep Obsidian)
*   **Primary (Accent):** `#7AE6C0` / `#5DCAA5` (The Kinetic Teal)
*   **Surface Containers:** Range from `Lowest` (`#0E0E0E`) to `Highest` (`#353534`)

### The "No-Line" Rule
Traditional 1px solid borders are strictly prohibited for structural sectioning. Boundaries must be defined through **Background Color Shifts** or **Translucency Tiers**.
*   **Bad:** A teal border around a sidebar.
*   **Good:** A `surface-container-low` panel with `backdrop-filter: blur(20px)` sitting over the `surface` background.

### Surface Hierarchy & Nesting
Treat the UI as a physical stack of glass.
1.  **Base Layer:** The camera feed/canvas.
2.  **Level 1 (Panels):** `surface-container-lowest` at 60% opacity with heavy blur.
3.  **Level 2 (Active Elements):** `surface-container-high` for hovered states or nested child containers.

### The "Glass & Gradient" Rule
To prevent a "flat" digital look, use subtle radial gradients on Primary CTAs. A transition from `primary` (`#7AE6C0`) to `primary-container` (`#5DCAA5`) provides a machined, glow-like quality reminiscent of high-end audio hardware.

---

## 3. Typography
We utilize a dual-font strategy to balance technical precision with modern editorial flair.

*   **Display & Headlines (Space Grotesk):** This font provides the "Instrument" soul. Its geometric, slightly wider stance suggests a technical readout. Use `display-lg` for gesture confirmations and `headline-sm` for panel titles.
*   **Body & Labels (Inter):** Chosen for its exceptional legibility at small sizes over moving video backgrounds.
*   **Hierarchy as Identity:** Use `label-sm` in uppercase with `letter-spacing: 0.05rem` for metadata. This creates a "serialized" look common in professional studio equipment.

---

## 4. Elevation & Depth
Depth is not achieved through shadow, but through **Tonal Layering** and **Atmospheric Optics**.

### The Layering Principle
Stacking `surface-container` tiers creates a natural lift. A `surface-container-highest` button on a `surface-container-low` panel creates an immediate, tactile hierarchy without the need for heavy-handed effects.

### Ambient Shadows
If an element must float (e.g., a cursor ring or a detached modal), use an **Extra-Diffused Glow**.
*   **Shadow Color:** Use a 10% opacity version of `primary` (`#7AE6C0`).
*   **Blur:** 24px - 40px. This mimics the light emission of an LED rather than a physical drop shadow.

### The "Ghost Border" Fallback
Where separation is critical for accessibility, use a **Ghost Border**:
*   **Token:** `outline-variant` (`#3D4944`)
*   **Opacity:** 15%
*   **Weight:** 1px
This ensures the border "shimmers" into view only when needed, maintaining the glass aesthetic.

---

## 5. Components

### Buttons & Pills
*   **Primary:** Pill-shaped (`rounded-full`). Background: `primary` gradient. Text: `on-primary` (`#003829`).
*   **Secondary (Glass):** `surface-container-high` at 40% opacity + `backdrop-blur`. No border.
*   **States:** On hover, increase `backdrop-blur` intensity rather than changing the base color.

### Input Fields & Sliders
*   **Inputs:** Use `surface-container-lowest` as a recessed "well." Labels should be `label-sm` placed above the field, never inside as placeholders.
*   **Sliders:** The track uses `outline-variant` at 20% opacity. The thumb is a `primary` teal circle with a subtle glow.

### Cards & Lists
*   **Anti-Divider Rule:** Forbid the use of line dividers. Use `spacing-4` (0.9rem) or `spacing-5` (1.1rem) of vertical whitespace to separate list items.
*   **Indicators:** Use a `primary` teal vertical "light bar" (2px wide) on the far left of a list item to indicate an `active` or `selected` state.

### Specialized Components
*   **Gesture Rings:** A 2px `primary` stroke ring that expands/contracts based on hand proximity.
*   **HUD Overlays:** Use `surface-low` with a 10% `primary` tint to signify the camera is "active" and the UI is in a listening state.

---

## 6. Do’s and Don’ts

### Do:
*   **Do** use `backdrop-filter: blur(12px)` on all panels to ensure text remains readable over complex video backgrounds.
*   **Do** use the Spacing Scale religiously to maintain "Technical Symmetry."
*   **Do** favor `surface-container` shifts over borders.

### Don’t:
*   **Don’t** use pure white (`#FFFFFF`) for text. Use `on-surface` (`#E5E2E1`) to reduce eye strain in dark environments.
*   **Don’t** use standard 4px or 8px border radii. This system requires `md` (1.5rem) or `full` (9999px) for a soft, ergonomic feel.
*   **Don’t** use more than one accent color. The teal must remain the sole "source of truth" for interaction.