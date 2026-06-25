import os, sys, time, base64, threading, io, queue, subprocess
import tkinter as tk
from tkinter import font as tkfont
from datetime import datetime


def pip_install(*packages):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", *packages, "--break-system-packages", "-q"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

for pkg, imp in [("mss", "mss"), ("Pillow", "PIL")]:
    try:
        __import__(imp)
    except ImportError:
        print(f"Installing {pkg}...")
        pip_install(pkg)

import mss
from PIL import Image


CONFIG = {
    "yolo_model":               "yolo26x",
    "yolo_imgsz":               1280,
    "yolo_conf":                0.45,
    "scan_interval_seconds":    8,
    "alarm_beeps":              5,
    "warning_display_seconds":  12,
    "confidence_threshold":     0.55,
    "screenshot_quality":       80,
    "max_screenshot_width":     1280,
}

SOCIAL_KEYWORDS = [
    "instagram", "tiktok", "youtube shorts", "reels", "snapchat",
    "twitter", "reddit", "facebook", "netflix", "twitch", "discord",
    "whatsapp", "telegram", "9gag", "imgur",
]

STRONG_DISTRACTION_OBJECTS = {"cell phone", "remote", "tv"}


def capture_screen() -> Image.Image:
    with mss.mss() as sct:
        mon = sct.monitors[1]
        shot = sct.grab(mon)
        img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
    max_w = CONFIG["max_screenshot_width"]
    if img.width > max_w:
        img = img.resize((max_w, int(img.height * max_w / img.width)), Image.LANCZOS)
    return img

def pil_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=CONFIG["screenshot_quality"])
    return buf.getvalue()


class YoloEngine:
    def __init__(self, model_name: str):
        try:
            from ultralytics import YOLO
        except ImportError:
            print("Installing ultralytics...")
            pip_install("ultralytics")
            from ultralytics import YOLO

        print(f"Loading {model_name}.pt (downloads on first run)...")
        self.model = YOLO(f"{model_name}.pt")
        self._active_title = ""
        threading.Thread(target=self._watch_window_title, daemon=True).start()

    def _watch_window_title(self):
        while True:
            try:
                if sys.platform == "linux":
                    out = subprocess.check_output(
                        ["xdotool", "getactivewindow", "getwindowname"],
                        stderr=subprocess.DEVNULL,
                    ).decode().strip().lower()
                elif sys.platform == "darwin":
                    script = 'tell application "System Events" to get name of first process whose frontmost is true'
                    out = subprocess.check_output(
                        ["osascript", "-e", script], stderr=subprocess.DEVNULL,
                    ).decode().strip().lower()
                elif sys.platform == "win32":
                    import ctypes
                    hwnd = ctypes.windll.user32.GetForegroundWindow()
                    buf = ctypes.create_unicode_buffer(512)
                    ctypes.windll.user32.GetWindowTextW(hwnd, buf, 512)
                    out = buf.value.lower()
                else:
                    out = ""
                self._active_title = out
            except Exception:
                pass
            time.sleep(2)

    def _check_window_title(self):
        for kw in SOCIAL_KEYWORDS:
            if kw in self._active_title:
                return True, f"Browser showing: {kw.title()}"
        return False, ""

    def analyze(self, img: Image.Image) -> dict:
        import numpy as np
        results = self.model(
            np.array(img), verbose=False,
            conf=CONFIG["yolo_conf"], imgsz=CONFIG["yolo_imgsz"],
        )[0]

        detections = [
            (self.model.names[int(b.cls[0])].lower(), float(b.conf[0]))
            for b in results.boxes
        ]

        title_hit, title_reason = self._check_window_title()
        strong_hits = [(l, c) for l, c in detections if l in STRONG_DISTRACTION_OBJECTS]
        people = [(l, c) for l, c in detections if l == "person"]

        if title_hit:
            return {
                "is_distracted": True,
                "confidence": 0.90,
                "category": title_reason,
                "description": "Window title contains a social or entertainment site",
                "yolo_objects": [l for l, _ in detections],
            }

        if strong_hits:
            best_label, best_conf = max(strong_hits, key=lambda x: x[1])
            label_map = {
                "cell phone": "Phone / Social Media",
                "remote": "TV / Streaming",
                "tv": "Streaming video",
            }
            return {
                "is_distracted": True,
                "confidence": min(0.85, best_conf + 0.15),
                "category": label_map[best_label],
                "description": f"Detected {best_label} on screen",
                "yolo_objects": [l for l, _ in detections],
            }

        if len(people) >= 2:
            return {
                "is_distracted": True,
                "confidence": 0.65,
                "category": "Video stream / video call",
                "description": f"{len(people)} people visible — likely video content",
                "yolo_objects": [l for l, _ in detections],
            }

        description = (
            f"Detected: {', '.join(set(l for l, _ in detections[:4]))}"
            if detections else "Screen looks clear"
        )
        return {
            "is_distracted": False,
            "confidence": 0.1,
            "category": None,
            "description": description,
            "yolo_objects": [l for l, _ in detections],
        }


def play_alarm():
    try:
        if sys.platform == "win32":
            import winsound
            for _ in range(CONFIG["alarm_beeps"]):
                winsound.Beep(880, 300); time.sleep(0.15)
        elif sys.platform == "darwin":
            for _ in range(CONFIG["alarm_beeps"]):
                os.system("afplay /System/Library/Sounds/Funk.aiff &"); time.sleep(0.4)
        else:
            for _ in range(CONFIG["alarm_beeps"]):
                os.system("paplay /usr/share/sounds/alsa/Front_Left.wav 2>/dev/null || echo -e '\\a'")
                time.sleep(0.3)
    except Exception:
        for _ in range(CONFIG["alarm_beeps"]):
            print("\a", end="", flush=True); time.sleep(0.3)


class WarningOverlay:
    def __init__(self):
        self.root = None
        self.active = False
        self._queue = queue.Queue()
        threading.Thread(target=self._run_tk, daemon=True).start()

    def _run_tk(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.root.after(100, self._drain_queue)
        self.root.mainloop()

    def _drain_queue(self):
        try:
            while True:
                self._queue.get_nowait()()
        except queue.Empty:
            pass
        if self.root:
            self.root.after(100, self._drain_queue)

    def show(self, info):
        self._queue.put(lambda: self._show(info))

    def hide(self):
        self._queue.put(self._hide)

    def _show(self, info):
        if self.active:
            self._hide()
        self.active = True

        w = tk.Toplevel(self.root)
        self.win = w
        w.title("Distraction detected")
        w.attributes("-topmost", True)
        w.attributes("-alpha", 0.94)
        w.overrideredirect(True)

        sw, sh = w.winfo_screenwidth(), w.winfo_screenheight()
        w.geometry(f"{sw}x{sh}+0+0")
        w.configure(bg="#0a0a0f")

        tk.Frame(w, bg="#ff1a1a").place(relx=.5, rely=.5, anchor="center", width=692, height=492)
        frame = tk.Frame(w, bg="#0d0d1a")
        frame.place(relx=.5, rely=.5, anchor="center", width=684, height=484)

        tk.Label(frame, text="🚨", font=("Segoe UI Emoji", 60), bg="#0d0d1a", fg="#ff3333").pack(pady=(32, 4))
        tk.Label(
            frame, text="DISTRACTION DETECTED",
            font=tkfont.Font(family="Helvetica", size=26, weight="bold"),
            bg="#0d0d1a", fg="#ff3333",
        ).pack()

        cat = info.get("category", "Unknown")
        conf = info.get("confidence", 0)
        tk.Label(
            frame, text=f"⚡  {cat}  ({int(conf * 100)}% confidence)",
            font=tkfont.Font(family="Helvetica", size=15),
            bg="#0d0d1a", fg="#ffaa00",
        ).pack(pady=(10, 2))

        desc = info.get("description", "")
        if desc:
            tk.Label(
                frame, text=desc,
                font=tkfont.Font(family="Helvetica", size=12),
                bg="#0d0d1a", fg="#cccccc", wraplength=580, justify="center",
            ).pack(pady=(2, 6))

        yolo_objs = info.get("yolo_objects")
        if yolo_objs:
            tk.Label(
                frame, text=f"Objects: {', '.join(yolo_objs[:6])}",
                font=tkfont.Font(family="Helvetica", size=10),
                bg="#0d0d1a", fg="#557755",
            ).pack()

        tk.Label(
            frame, text="🎯  Get back to work. Your future self will thank you.",
            font=tkfont.Font(family="Helvetica", size=13, slant="italic"),
            bg="#0d0d1a", fg="#44ff88",
        ).pack(pady=(8, 0))

        bar_frame = tk.Frame(frame, bg="#0d0d1a")
        bar_frame.pack(pady=(16, 0), fill="x", padx=40)

        self._countdown_var = tk.StringVar()
        tk.Label(
            bar_frame, textvariable=self._countdown_var,
            font=tkfont.Font(family="Helvetica", size=11),
            bg="#0d0d1a", fg="#888888",
        ).pack()

        self._progress = tk.Canvas(bar_frame, height=8, bg="#222233", highlightthickness=0)
        self._progress.pack(fill="x", pady=(4, 0))

        tk.Button(
            frame, text="✕  I'm back to work",
            font=tkfont.Font(family="Helvetica", size=13, weight="bold"),
            bg="#ff3333", fg="white", activebackground="#cc0000",
            bd=0, padx=20, pady=10, cursor="hand2",
            command=self._hide,
        ).pack(pady=(14, 0))

        tk.Label(
            frame, text=f"Detected at {datetime.now():%H:%M:%S}",
            font=tkfont.Font(family="Helvetica", size=9),
            bg="#0d0d1a", fg="#444455",
        ).pack(pady=(6, 0))

        self._total = CONFIG["warning_display_seconds"]
        self._remaining = self._total
        self._tick()

    def _tick(self):
        if not self.active:
            return
        try:
            if self._remaining <= 0:
                self._hide()
                return
            self._countdown_var.set(f"Auto-closing in {self._remaining}s")
            width = self._progress.winfo_width() or 580
            fill = int(width * self._remaining / self._total)
            r = int(255 * self._remaining / self._total)
            g = int(255 * (1 - self._remaining / self._total))
            self._progress.delete("all")
            self._progress.create_rectangle(0, 0, fill, 8, fill=f"#{r:02x}{g:02x}33", outline="")
            self._remaining -= 1
            self.win.after(1000, self._tick)
        except tk.TclError:
            pass

    def _hide(self):
        self.active = False
        try:
            if hasattr(self, "win"):
                self.win.destroy()
        except tk.TclError:
            pass


class Dashboard:
    def __init__(self, engine_name):
        self.engine_name = engine_name
        self.scans = 0
        self.distractions = 0
        self.started_at = datetime.now()
        self.last_caught = None
        self.status = "Starting..."
        self._lock = threading.Lock()

    def _uptime(self):
        elapsed = datetime.now() - self.started_at
        h, remainder = divmod(int(elapsed.total_seconds()), 3600)
        m, s = divmod(remainder, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def record(self, result):
        with self._lock:
            self.scans += 1
            if result.get("is_distracted"):
                self.distractions += 1
                self.last_caught = datetime.now()

    def render(self, status, result=None):
        with self._lock:
            os.system("clear" if os.name != "nt" else "cls")
            W = 62
            print(f"\033[1;36m{'─' * W}\033[0m")
            print(f"\033[1;36m  🔍 DISTRACTION DETECTOR  │  {self.engine_name}\033[0m")
            print(f"\033[1;36m{'─' * W}\033[0m")
            print(f"  ⏱  Uptime:        {self._uptime()}")
            print(f"  🔎 Scans:         {self.scans}")

            pct = self.distractions / self.scans * 100 if self.scans else 0
            color = "\033[31m" if pct > 30 else "\033[33m" if pct > 10 else "\033[32m"
            print(f"  🚨 Distractions:  {color}{self.distractions} ({pct:.1f}%)\033[0m")

            if self.last_caught:
                ago = int((datetime.now() - self.last_caught).total_seconds())
                print(f"  ⚡ Last caught:   {ago}s ago")

            print(f"\n  📡 {status}")
            print(f"\033[1;36m{'─' * W}\033[0m")

            if result:
                if result.get("is_distracted"):
                    print(f"\n  \033[31m🚨 {result.get('category', '?')}\033[0m")
                    print(f"  \033[33m   {result.get('description', '')}\033[0m")
                    objs = result.get("yolo_objects")
                    if objs:
                        print(f"  \033[90m   objects: {', '.join(objs[:5])}\033[0m")
                else:
                    print(f"\n  \033[32m✅ {result.get('description', 'Focused')}\033[0m")

            print(f"\n  Ctrl+C to stop\n")


def main():
    print("\033[1;36m")
    print("╔══════════════════════════════════════════════╗")
    print("║   🔍 DISTRACTION DETECTOR                    ║")
    print("║   Engine: YOLOv26 (local, offline)           ║")
    print("╚══════════════════════════════════════════════╝")
    print("\033[0m")

    engine = YoloEngine(CONFIG["yolo_model"])
    engine_label = f"YOLOv26 ({CONFIG['yolo_model']})"
    print(f"✅ {engine_label} ready\n")

    overlay = WarningOverlay()
    dash = Dashboard(engine_label)
    last_result = None

    print(f"Scanning every {CONFIG['scan_interval_seconds']}s — Ctrl+C to stop\n")
    time.sleep(1)

    while True:
        try:
            dash.render("📷 Capturing screen...", last_result)
            img = capture_screen()

            dash.render(f"🤖 Analyzing...", last_result)
            result = engine.analyze(img)
            last_result = result
            dash.record(result)

            is_caught = result.get("is_distracted") and result.get("confidence", 0) >= CONFIG["confidence_threshold"]

            if is_caught:
                dash.render(f"🚨 {result.get('category')}", result)
                threading.Thread(target=play_alarm, daemon=True).start()
                overlay.show(result)
                def auto_hide():
                    time.sleep(CONFIG["warning_display_seconds"])
                    overlay.hide()
                threading.Thread(target=auto_hide, daemon=True).start()
            else:
                overlay.hide()
                dash.render("✅ Focused", result)

            for remaining in range(CONFIG["scan_interval_seconds"], 0, -1):
                dash.render(f"⏳ Next scan in {remaining}s...", last_result)
                time.sleep(1)

        except KeyboardInterrupt:
            print("\n\033[1;33m  👋 Stopped. Stay focused!\033[0m\n")
            overlay.hide()
            sys.exit(0)

        except Exception as e:
            dash.render(f"⚠️  {type(e).__name__}: {str(e)[:55]}", last_result)
            time.sleep(CONFIG["scan_interval_seconds"])


if __name__ == "__main__":
    main()
