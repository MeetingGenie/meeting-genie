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
import ctypes
from ctypes import wintypes


class PlatformWindow:

    SW_SHOWNOACTIVATE = 4

    SW_HIDE = 0

    HWND_TOPMOST = -1

    SWP_NOMOVE = 0x0002

    SWP_NOSIZE = 0x0001

    SWP_NOACTIVATE = 0x0010

    SWP_SHOWWINDOW = 0x0040

    def __init__(self, root, debug: bool = True):
        self.root = root
        self.debug = debug
        self.system = platform.system()
        self.hwnd = None
        self.user32 = None

    def configure(self) -> None:
        if self.system == "Windows":
            self._configure_windows()
        elif self.system == "Darwin":
            self._configure_macos()

        else:

            raise NotImplementedError(
                f"{self.system} is not supported."
            )

    def show(self) -> None:
        if self.system != "Windows":
            self.root.deiconify()
            return
        # Make sure the window exists.
        self.root.update_idletasks()
        # Show without activating.
        self.user32.ShowWindow(self.hwnd,self.SW_SHOWNOACTIVATE)
        # Keep the overlay above normal windows.
        self.user32.SetWindowPos(self.hwnd,self.HWND_TOPMOST,0,0,0,0,self.SWP_NOMOVE| self.SWP_NOSIZE| self.SWP_NOACTIVATE| self.SWP_SHOWWINDOW)

    def hide(self) -> None:
        if self.system != "Windows":
            self.root.withdraw()
            return
        self.user32.ShowWindow(self.hwnd,self.SW_HIDE)

    def reset_opacity(self) -> None:
        self.root.attributes("-alpha", 1.0)

    def set_opacity(self,alpha: float) -> None:
        alpha = max(0.0, min(1.0, alpha))
        self.root.attributes( "-alpha",alpha)

    def _configure_windows(self) -> None:
        """
        Configure the native Windows window.
        """
        # Force Tk to create the native window.
        self.root.update_idletasks()
        # Native window handle.
        self.hwnd = self.root.winfo_id()
        self.user32 = ctypes.windll.user32
        # Always on top.
        self.root.attributes("-topmost",True)
        # During development we keep the title bar.
        if not self.debug:
            self.root.overrideredirect(True)

   
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