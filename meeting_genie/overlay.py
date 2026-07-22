"""
MeetingGenie - overlay.py


Responsibilities
----------------
- Floating desktop overlay.
- Queue-based UI updates.
- Never blocks the transcription pipeline.
- State machine:
    IDLE
    READY
    EXPANDED
    STALE


Business logic, Whisper, Ollama and triggering
belong in other modules.
"""


from __future__ import annotations
from meeting_genie.platformwindow import PlatformWindow
import platform
import ctypes
from ctypes import wintypes
import queue
from enum import Enum
import tkinter as tk
from tkinter import font


DEV_MODE = True

class OverlayState(Enum):
    IDLE = "idle"
    READY = "ready"
    LOADING = "loading"
    EXPANDED = "expanded"
    STALE = "stale"

class Overlay:


    def __init__(self):
        # Runtime
        self.running = True
        self.state = OverlayState.IDLE
        self.message_queue = queue.Queue()
        self.current_text = ""
        # Overlay lifetime
        self.minimum_display_seconds = 8
        self.maximum_display_seconds = 25
        self.words_per_second = 3
        self.fade_steps = 10
        self.fade_interval_ms = 60
        self._stale_after_id = None
        self._fade_after_id = None
        # Window sizes
        self.pill_width = 220
        self.pill_height = 55
        self.window_width = 500
        self.window_width = 500
        # Dynamic popup height
        self.minimum_window_height = 180
        self.maximum_window_height = 450
        self.line_height = 22
        self.top_margin = 40
        self.right_margin = 30
        # Create root
        self.root = tk.Tk()
        self.platform_window = PlatformWindow(self.root,debug=DEV_MODE)
        self._create_window()
        self.platform_window.configure()
        self._create_fonts()
        self._create_frames()
        self._create_widgets()
        self.set_state(OverlayState.READY)
        self.root.after(50,self._process_queue)

    def _calculate_window_height(self) -> int:
        """
        Grow the popup as more text arrives.

        Stops growing once maximum height is reached.
        """
        lines = self.current_text.count("\n") + 1

        estimated = (self.minimum_window_height+ lines * self.line_height)
        return min(max(estimated,self.minimum_window_height,),self.maximum_window_height,)


    def _create_window(self) -> None:
     """
     Configure the base overlay window.
     """
     self.root.title("MeetingGenie")
     # Fixed-size overlay.
     self.root.resizable(False, False)
     self.root.configure(bg="#202124")


    def _configure_windows(self) -> None:
        """
        Windows-specific initialization.
        """
        # Force Tk to create the native window.
        self.root.update_idletasks()
        # Native HWND.
        self.hwnd = self.root.winfo_id()
        # Win32 API handles.
        self.user32 = ctypes.windll.user32


    def _configure_macos(self) -> None:
        """
        macOS-specific window configuration.


        Native Cocoa improvements will be added later.
        """
        pass

    def _calculate_display_time(self):
        words = len(self.current_text.split())
        seconds = max(self.minimum_display_seconds, words / self.words_per_second)
        return min(seconds, self.maximum_display_seconds)

    def _create_fonts(self):
        self.title_font = font.Font(family="Segoe UI",size=14,weight="bold")
        self.message_font = font.Font(family="Segoe UI",size=11)


    def _create_frames(self):
        self.pill_frame = tk.Frame(self.root,bg="#202124")
        self.expanded_frame = tk.Frame(self.root,bg="#202124")


    def _create_widgets(self):
        # READY (pill)
        self.pill_label = tk.Label(self.pill_frame,text="🎙 MeetingGenie Ready",bg="#202124",fg="white",font=self.title_font,padx=15,pady=10)
        self.pill_label.pack(padx=15,pady=1)
        # EXPANDED
        self.title_label = tk.Label(self.expanded_frame,text="MeetingGenie",bg="#202124",fg="white",font=self.title_font)
        self.title_label.pack(pady=(10, 5))
        self.message_label = tk.Label(self.expanded_frame,text="",justify="left",wraplength=460,bg="#202124",fg="white",font=self.message_font)
        self.message_label.pack(padx=15,pady=10,fill="both", expand=True)
        #close button
        self.root.bind("<Control-Shift-Q>",self._developer_quit)


    def _position_window(self,width: int,height: int) -> None:
        screen_width = self.root.winfo_screenwidth()
        x = screen_width - width - self.right_margin
        y = self.top_margin
        self.root.geometry(f"{width}x{height}+{x}+{y}")


    def _clear_frames(self) -> None:
        self.pill_frame.pack_forget()
        self.expanded_frame.pack_forget()


    def set_state(self,state: OverlayState) -> None:
        self.state = state
        self._clear_frames()
        if state == OverlayState.IDLE:
            self.platform_window.hide()
        elif state == OverlayState.READY:
            self.platform_window.reset_opacity()
            self.current_text = ""
            self.message_label.config(text="")
            self._position_window(self.pill_width,self.pill_height)
            self.pill_frame.pack(fill="both",expand=True)
            self.platform_window.show()
        elif state == OverlayState.EXPANDED:
            self._position_window(self.window_width,self._calculate_window_height(),)
            self.expanded_frame.pack(fill="both",expand=True)
            self.platform_window.show()
        elif state == OverlayState.LOADING:
            self.message_label.config(text="Thinking...")
            self._position_window(self.window_width,self._calculate_window_height(),)
            self.expanded_frame.pack(fill="both",expand=True )
            self.platform_window.show()
        elif state == OverlayState.STALE:
            self.platform_window.hide()


    def show_message(self,message: str) -> None:
        self.message_queue.put(("replace",message))


    def show_loading(self) -> None:
        """
        Display a loading message while the LLM
        prepares its response.
        """
        self.show_message("Thinking...")


    def append_text(self,text: str) -> None:
        self.message_queue.put(("append",text))


    def _process_queue(self) -> None:
        updated = False
        while not self.message_queue.empty():
            action, payload = self.message_queue.get_nowait()
            if action == "replace":
                self.current_text = payload
            elif action == "append":
                self.current_text += payload
            updated = True
        if updated:
            self.message_label.config(text=self.current_text,wraplength=self.window_width - 40,)
            height = self._calculate_window_height()
            self._position_window(self.window_width,height,)
            self.set_state(OverlayState.EXPANDED)
            self._restart_stale_timer()
        if self.running:
            self.root.after(50,self._process_queue,)

    def _restart_stale_timer(self) -> None:
        if self._fade_after_id is not None:
            self.root.after_cancel(self._fade_after_id)
            self._fade_after_id = None
        self.platform_window.reset_opacity()
        if self._stale_after_id is not None:
            self.root.after_cancel(self._stale_after_id)
        display_time = self._calculate_display_time()
        self._stale_after_id = self.root.after(int(display_time * 1000),self._begin_fade,)


    def _begin_fade(self) -> None:
        """
        Start the fade animation.
        """
        self._fade_step = 0
        self._fade()


    def _fade(self) -> None:
        """
        Fade smoothly.
        """
        opacity = (1.0 - (self._fade_step / self.fade_steps))
        opacity = max(0.0,opacity,)
        self.platform_window.set_opacity( opacity)
        self._fade_step += 1
        if self._fade_step <= self.fade_steps:
            self._fade_after_id = self.root.after(self.fade_interval_ms,self._fade,)
        else:
            self.platform_window.reset_opacity()
            self.set_state(OverlayState.READY)


    def start(self) -> None:
        """
        Start the overlay.
        """
        self.root.mainloop()


    def stop(self) -> None:
        """
        Shut down the overlay.
        """
        self.running = False
        self.root.quit()
        self.root.destroy()


    def _developer_quit(self, event=None) -> None:
        """
        Temporary developer shortcut.
        Remove before release.
        """
        self.stop()
       


if __name__ == "__main__":


    overlay = Overlay()
    #overlay.start()
    # ----------------------------
    # Test 2 : Expanded
    # ----------------------------
    overlay.show_loading()
    #overlay.show_message("Hello!\n\nThis is a MeetingGenie test.")
    overlay.start()
