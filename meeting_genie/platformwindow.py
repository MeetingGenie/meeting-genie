"""
MeetingGenie - platform_window.py

Platform-specific window behaviour.

Windows:
    - Topmost
    - Borderless (production)
    - Never activate (implemented later)

macOS:
    - Best-effort floating window

Everything else belongs in overlay.py.
"""

from __future__ import annotations

import platform


class PlatformWindow:

    def __init__(self, root, debug: bool = True):

        self.root = root
        self.debug = debug

        self.system = platform.system()

    # ---------------------------------------------------------

    def configure(self) -> None:

        if self.system == "Windows":

            self._configure_windows()

        elif self.system == "Darwin":

            self._configure_macos()

        else:

            raise NotImplementedError(
                f"{self.system} is not supported."
            )

    # ---------------------------------------------------------

    def show(self) -> None:

        self.root.deiconify()

    # ---------------------------------------------------------

    def hide(self) -> None:

        self.root.withdraw()

    # ---------------------------------------------------------

    def reset_opacity(self) -> None:

        self.root.attributes("-alpha", 1.0)

    # ---------------------------------------------------------

    def set_opacity(
        self,
        alpha: float
    ) -> None:

        alpha = max(0.0, min(1.0, alpha))

        self.root.attributes(
            "-alpha",
            alpha
        )

    # ---------------------------------------------------------

    def _configure_windows(self) -> None:
        """
        Windows configuration.

        Win32 no-focus will be added here.
        """

        self.root.attributes(
            "-topmost",
            True
        )

        if not self.debug:

            self.root.overrideredirect(
                True
            )

    # ---------------------------------------------------------

    def _configure_macos(self) -> None:
        """
        macOS configuration.

        Best-effort Tk implementation.
        """

        self.root.attributes(
            "-topmost",
            True
        )

        if not self.debug:

            self.root.overrideredirect(
                True
            )