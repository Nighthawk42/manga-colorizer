# main.py
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import threading
import os
import sys
import numpy as np
import argparse
import warnings
import platform
import shutil
import subprocess
import queue
import time
import cv2

def _prune_sys_path_configs():
    # Remove entries from sys.path that contain a top-level config.py,
    # so cv2 won't accidentally exec it.
    pruned = []
    for p in list(sys.path):
        try:
            base = p or os.getcwd()
            if os.path.isfile(os.path.join(base, "config.py")):
                # Drop this path entry
                continue
        except Exception:
            pass
        pruned.append(p)
    sys.path[:] = pruned

_prune_sys_path_configs()

# --- THEME IMPORTS ---
import sv_ttk
import darkdetect
if platform.system() == "Windows":
    try:
        import pywinstyles
    except ImportError:
        pywinstyles = None

# --- SUPPRESSION ---
try:
    from torch.serialization import SourceChangeWarning
    warnings.filterwarnings("ignore", category=SourceChangeWarning)
except ImportError:
    pass

# --- APP IMPORTS (Refactored) ---
# <-- FIX: Removed 'backend.' from imports to match file structure
from colorizator import MangaColorizator
from denoisator import MangaDenoiser
from upscalator import MangaUpscaler
from utils.utils import save_image


# =========================
# Theme Manager
# =========================
class ThemeManager:
    def __init__(self, root: tk.Tk):
        self.root = root

    def apply_windows_titlebar_theme(self):
        if platform.system() != "Windows" or pywinstyles is None:
            return
        theme = sv_ttk.get_theme()
        version = sys.getwindowsversion()
        try:
            if version.major == 10 and version.build >= 22000:
                header_color = "#1c1c1c" if theme == "dark" else "#fafafa"
                pywinstyles.change_header_color(self.root, header_color)
            elif version.major == 10:
                pywinstyles.apply_style(self.root, "dark" if theme == "dark" else "normal")
                # little alpha flip to refresh caption colors
                self.root.wm_attributes("-alpha", 0.99)
                self.root.wm_attributes("-alpha", 1.0)
        except Exception:
            pass

    def toggle_theme(self):
        current = sv_ttk.get_theme()
        sv_ttk.set_theme("light" if current == "dark" else "dark")
        self.apply_windows_titlebar_theme()

    def set_initial(self, theme_name: str):
        sv_ttk.set_theme(theme_name)
        self.apply_windows_titlebar_theme()


# =========================
# Processor Pipeline
# =========================
class ProcessorPipeline:
    def __init__(self, config):
        self.config = config
        self.colorizer = MangaColorizator(config) if config.colorize else None
        self.upscaler = MangaUpscaler(config) if config.upscale else None
        self.denoiser = MangaDenoiser(config) if config.denoise else None

    def process(self, image_np: np.ndarray,
                do_denoise: bool, do_colorize: bool, do_upscale: bool) -> np.ndarray:
        out = image_np
        if do_denoise and self.denoiser:
            # Denoiser returns a BGR image, which needs conversion for subsequent steps
            out = self.denoiser.denoise(out, self.config.denoise_sigma)
            # Ensure RGB for colorizer (denoiser output is BGR)
            out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

        if do_colorize and self.colorizer:
            self.colorizer.set_image((out.astype('float32') / 255.0), self.config.colorized_image_size)
            out = self.colorizer.colorize()

        if do_upscale and self.upscaler:
            out = self.upscaler.upscale((out.astype('float32') / 255.0), self.config.upscale_factor)

        return out


# =========================
# Base Tab
# =========================
class BaseTab(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app  # provides access to pipeline, theme, etc.


# =========================
# Single Image Tab
# =========================
class SingleImageTab(BaseTab):
    def __init__(self, parent, app):
        super().__init__(parent, app)
        self.original_image = None
        self.processed_image = None
        self.image_path = None

        self.denoise_var = tk.BooleanVar(value=True)
        self.colorize_var = tk.BooleanVar(value=True)
        self.upscale_var = tk.BooleanVar(value=True)

        self._build()

    def _build(self):
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Controls
        control = ttk.Frame(main_frame)
        control.pack(side=tk.TOP, fill=tk.X, pady=5)

        self.btn_open = ttk.Button(control, text="Open Image", command=self.open_image)
        self.btn_open.pack(side=tk.LEFT, padx=5)

        self.btn_process = ttk.Button(control, text="Process Image",
                                      command=self.start_processing, state=tk.DISABLED)
        self.btn_process.pack(side=tk.LEFT, padx=5)

        self.btn_save = ttk.Button(control, text="Save Image",
                                   command=self.save_image, state=tk.DISABLED)
        self.btn_save.pack(side=tk.LEFT, padx=5)

        # Options
        options = ttk.LabelFrame(main_frame, text="Processing Options", padding=10)
        options.pack(side=tk.TOP, fill=tk.X, pady=10)
        ttk.Checkbutton(options, text="Denoise", variable=self.denoise_var).pack(side=tk.LEFT, padx=10)
        ttk.Checkbutton(options, text="Colorize", variable=self.colorize_var).pack(side=tk.LEFT, padx=10)
        ttk.Checkbutton(options, text="Upscale", variable=self.upscale_var).pack(side=tk.LEFT, padx=10)

        # Image Panels
        image_frame = ttk.Frame(main_frame)
        image_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        self.panel_original = ttk.Label(image_frame, text="Original Image", relief="groove", anchor="center")
        self.panel_original.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.panel_processed = ttk.Label(image_frame, text="Processed Image", relief="groove", anchor="center")
        self.panel_processed.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg *.webp")])
        if not path:
            return
        self.image_path = path
        self.original_image = Image.open(path).convert("RGB")
        self._display_image(self.original_image, self.panel_original)
        self.panel_processed.config(image='', text="Processed Image")
        self.processed_image = None
        self.app.set_status(f"Loaded: {os.path.basename(path)}")
        self._update_buttons()

    def _display_image(self, img: Image.Image, panel: ttk.Label):
        panel_w, panel_h = panel.winfo_width(), panel.winfo_height()
        if panel_w < 2 or panel_h < 2:
            panel_w, panel_h = 550, 550
        p = img.copy()
        p.thumbnail((panel_w, panel_h))
        photo = ImageTk.PhotoImage(p)
        panel.config(image=photo)
        panel.image = photo

    def start_processing(self):
        self._set_ui_state(tk.DISABLED)
        self.app.set_status("Processing...")
        threading.Thread(target=self._process_worker, daemon=True).start()

    def _process_worker(self):
        try:
            src_np = np.array(self.original_image)
            out_np = self.app.pipeline.process(
                src_np, self.denoise_var.get(), self.colorize_var.get(), self.upscale_var.get()
            )
            self.processed_image = Image.fromarray(out_np)
            self.after(0, self._on_done)
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Processing Error", f"An error occurred: {e}"))
            self.after(0, lambda: self._set_ui_state(tk.NORMAL))
            self.after(0, lambda: self.app.set_status("Error during processing."))


    def _on_done(self):
        self._display_image(self.processed_image, self.panel_processed)
        self._set_ui_state(tk.NORMAL)
        self.app.set_status("Processing complete.")
        self._update_buttons()

    def save_image(self):
        if not self.processed_image:
            return
        save_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("WebP", "*.webp")]
        )
        if not save_path:
            return
        try:
            fmt = os.path.splitext(save_path)[1][1:].upper()
            if fmt == "JPG":
                fmt = "JPEG"
            save_image(np.array(self.processed_image), save_path, format=fmt)
            messagebox.showinfo("Success", f"Image saved to {save_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save image: {e}")

    def _update_buttons(self):
        self.btn_process.config(state=tk.NORMAL if self.image_path else tk.DISABLED)
        self.btn_save.config(state=tk.NORMAL if self.processed_image is not None else tk.DISABLED)

    def _set_ui_state(self, state):
        self.btn_open.config(state=state)
        # Re-evaluate process/save button states based on image presence
        self.btn_process.config(state=state if self.image_path else tk.DISABLED)
        self.btn_save.config(state=state if self.processed_image is not None else tk.DISABLED)


# =========================
# Batch Tab
# =========================
class BatchTab(BaseTab):
    def __init__(self, parent, app):
        super().__init__(parent, app)
        self.input_folder = None
        self.output_folder = None
        self.gallery_widgets = {}
        self.thumbnail_refs = []

        self.denoise_var = tk.BooleanVar(value=True)
        self.colorize_var = tk.BooleanVar(value=True)
        self.upscale_var = tk.BooleanVar(value=True)

        self._build()

    def _build(self):
        main = ttk.Frame(self, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        control = ttk.Frame(main)
        control.pack(side=tk.TOP, fill=tk.X, pady=5)

        self.btn_folders = ttk.Button(control, text="Select Folders", command=self.select_folders)
        self.btn_folders.pack(side=tk.LEFT, padx=5)

        self.btn_start = ttk.Button(control, text="Start Batch Process",
                                    command=self.start_batch, state=tk.DISABLED)
        self.btn_start.pack(side=tk.LEFT, padx=5)

        self.folder_status = ttk.Label(control, text="No folders selected.")
        self.folder_status.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)

        # Options
        options = ttk.LabelFrame(main, text="Processing Options", padding=10)
        options.pack(side=tk.TOP, fill=tk.X, pady=10)
        ttk.Checkbutton(options, text="Denoise", variable=self.denoise_var).pack(side=tk.LEFT, padx=10)
        ttk.Checkbutton(options, text="Colorize", variable=self.colorize_var).pack(side=tk.LEFT, padx=10)
        ttk.Checkbutton(options, text="Upscale", variable=self.upscale_var).pack(side=tk.LEFT, padx=10)

        # Gallery
        gallery = ttk.LabelFrame(main, text="Image Preview", padding=10)
        gallery.pack(fill=tk.BOTH, expand=True, pady=10)

        self.canvas = tk.Canvas(gallery, highlightthickness=0)
        self.scroll = ttk.Scrollbar(gallery, orient="vertical", command=self.canvas.yview)
        self.scrollable = ttk.Frame(self.canvas)

        self.scrollable.bind("<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scrollable, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scroll.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scroll.pack(side="right", fill="y")

        self.progress = ttk.Progressbar(main, orient="horizontal", mode="determinate")
        self.progress.pack(side=tk.BOTTOM, fill=tk.X, pady=5)

    def select_folders(self):
        in_dir = filedialog.askdirectory(title="Select Input Folder")
        if not in_dir:
            return
        out_dir = filedialog.askdirectory(title="Select Output Folder")
        if not out_dir:
            return
        self.input_folder = in_dir
        self.output_folder = out_dir
        self.folder_status.config(text=f"In: ...{in_dir[-30:]} | Out: ...{out_dir[-30:]}")
        self.btn_start.config(state=tk.NORMAL)
        threading.Thread(target=self._load_thumbs, daemon=True).start()

    def _load_thumbs(self):
        self._set_ui_state(tk.DISABLED)
        self.app.set_status("Loading thumbnails...")
        for w in self.scrollable.winfo_children():
            w.destroy()
        self.thumbnail_refs.clear()
        self.gallery_widgets.clear()

        if not self.input_folder:
            self._set_ui_state(tk.NORMAL)
            return

        files = [f for f in os.listdir(self.input_folder)
                 if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]

        for i, fname in enumerate(files):
            try:
                img = Image.open(os.path.join(self.input_folder, fname))
                img.thumbnail((128, 128))
                photo = ImageTk.PhotoImage(img)
                self.thumbnail_refs.append(photo)
                self.after(0, self._add_thumb, photo, fname, i)
            except Exception as e:
                print(f"Thumb fail {fname}: {e}")

        self.after(0, lambda: self.app.set_status(f"Loaded {len(files)} images. Ready for batch."))
        self.after(0, lambda: self._set_ui_state(tk.NORMAL))

    def _add_thumb(self, photo, fname, idx):
        lbl = ttk.Label(self.scrollable, image=photo, text=fname, compound="top")
        r, c = divmod(idx, 5)
        lbl.grid(row=r, column=c, padx=5, pady=5)
        self.gallery_widgets[fname] = lbl

    def _update_thumb(self, fname, npimg):
        try:
            img = Image.fromarray(npimg)
            img.thumbnail((128, 128))
            ph = ImageTk.PhotoImage(img)
            w = self.gallery_widgets.get(fname)
            if w:
                w.config(image=ph)
                w.image = ph
        except Exception as e:
            print(f"Update thumb fail {fname}: {e}")

    def start_batch(self):
        self._set_ui_state(tk.DISABLED)
        threading.Thread(target=self._batch_worker, daemon=True).start()

    def _batch_worker(self):
        try:
            files = [f for f in os.listdir(self.input_folder)
                     if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
            total = len(files)
            self.after(0, lambda: self.progress.config(maximum=total, value=0))
            for i, fname in enumerate(files, 1):
                self.after(0, lambda i=i, f=fname, t=total:
                           self.app.set_status(f"Processing [{i}/{t}]: {f}"))
                src = os.path.join(self.input_folder, fname)
                npimg = np.array(Image.open(src).convert("RGB"))
                out = self.app.pipeline.process(
                    npimg, self.denoise_var.get(), self.colorize_var.get(), self.upscale_var.get()
                )
                # Save as PNG
                base, _ = os.path.splitext(fname)
                save_path = os.path.join(self.output_folder, f"{base}.png")
                save_image(out, save_path, format="PNG")
                self.after(0, self._update_thumb, fname, out)
                self.after(0, self.progress.step)
            self.after(0, lambda: messagebox.showinfo("Success", "Batch processing finished successfully."))
            self.after(0, lambda: self.app.set_status(f"Batch complete. Processed {total} files."))
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Batch Processing Error", f"An error occurred: {e}"))
            self.after(0, lambda: self.app.set_status("Error during batch processing."))
        finally:
            self.after(0, lambda: self._set_ui_state(tk.NORMAL))
            self.after(0, lambda: self.progress.config(value=0))

    def _set_ui_state(self, state):
        self.btn_folders.config(state=state)
        self.btn_start.config(state=state if self.input_folder else tk.DISABLED)


# # =========================
# # Downloaders Tab (MangaDex, NHentai, possibly more.)
# # =========================
# class DownloadersTab(BaseTab):
    # def __init__(self, parent, app):
        # super().__init__(parent, app)
        # self.out_dir = tk.StringVar(value=os.path.abspath("downloads"))
        # self._build()

        # self.proc_thread = None
        # self.log_q = queue.Queue()
        # self.stop_flag = threading.Event()

    # # ---------- UI ----------
    # def _build(self):
        # container = ttk.Frame(self, padding=10)
        # container.pack(fill=tk.BOTH, expand=True)

        # # Output row
        # out_row = ttk.Frame(container)
        # out_row.pack(fill=tk.X, pady=(0, 10))
        # ttk.Label(out_row, text="Output Folder:").pack(side=tk.LEFT)
        # self.out_entry = ttk.Entry(out_row, textvariable=self.out_dir)
        # self.out_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        # ttk.Button(out_row, text="Browse", command=self._pick_out).pack(side=tk.LEFT)

        # # Notebook for providers
        # self.provider_nb = ttk.Notebook(container)
        # self.provider_nb.pack(fill=tk.BOTH, expand=True)

        # self._build_mangadex_tab()
        # self._build_nhentai_tab()

        # # Log + actions
        # bottom = ttk.Frame(container)
        # bottom.pack(fill=tk.X, pady=(10, 0))

        # ttk.Button(bottom, text="Open Folder", command=self._open_folder).pack(side=tk.LEFT)
        # ttk.Button(bottom, text="Use as Batch Input", command=self._use_as_batch_input).pack(side=tk.LEFT, padx=8)

        # self.log = tk.Text(container, height=12, wrap="word")
        # self.log.pack(fill=tk.BOTH, expand=False, pady=(10, 0))
        # self.log.configure(state="disabled")

    # def _build_mangadex_tab(self):
        # tab = ttk.Frame(self.provider_nb, padding=10)
        # self.provider_nb.add(tab, text="MangaDex")

        # ttk.Label(tab, text="MangaDex URL:").pack(anchor="w")
        # self.md_url = tk.StringVar()
        # ttk.Entry(tab, textvariable=self.md_url).pack(fill=tk.X, pady=5)

        # help_md = (
            # "Requirements:\n"
            # "- pip install mangadex-downloader  (optional extras: [optional])\n"
            # "Usage examples:\n"
            # "  mangadex-dl \"<URL>\"\n"
            # "  python -m mangadex_downloader \"<URL>\"\n"
        # )
        # ttk.Label(tab, text=help_md, justify="left").pack(anchor="w", pady=(0, 5))

        # btns = ttk.Frame(tab)
        # btns.pack(fill=tk.X, pady=5)
        # self.btn_md_dl = ttk.Button(btns, text="Download", command=self._start_mangadex)
        # self.btn_md_dl.pack(side=tk.LEFT)

    # def _build_nhentai_tab(self):
        # tab = ttk.Frame(self.provider_nb, padding=10)
        # self.provider_nb.add(tab, text="nhentai")

        # ttk.Label(tab, text="User-Agent (recommended):").pack(anchor="w")
        # self.nh_ua = tk.StringVar()
        # ttk.Entry(tab, textvariable=self.nh_ua).pack(fill=tk.X, pady=3)

        # ttk.Label(tab, text="Cookie (csrftoken=...; sessionid=...; cf_clearance=...):").pack(anchor="w")
        # self.nh_cookie = tk.StringVar()
        # ttk.Entry(tab, textvariable=self.nh_cookie).pack(fill=tk.X, pady=3)

        # ttk.Label(tab, text="Mode:").pack(anchor="w", pady=(6, 0))
        # self.nh_mode = tk.StringVar(value="id")
        # mode_row = ttk.Frame(tab); mode_row.pack(anchor="w", pady=2)
        # ttk.Radiobutton(mode_row, text="IDs", variable=self.nh_mode, value="id").pack(side=tk.LEFT)
        # ttk.Radiobutton(mode_row, text="Search", variable=self.nh_mode, value="search").pack(side=tk.LEFT)
        # ttk.Radiobutton(mode_row, text="Favorites (login cookie required)", variable=self.nh_mode, value="favorites").pack(side=tk.LEFT)

        # self.nh_ids = tk.StringVar()
        # self.nh_query = tk.StringVar()
        # self.nh_page = tk.StringVar(value="1")
        # self.nh_download = tk.BooleanVar(value=True)
        # self.nh_cbz = tk.BooleanVar(value=False)
        # self.nh_pdf = tk.BooleanVar(value=False)
        # self.nh_delay = tk.StringVar(value="0")

        # # IDs
        # ids_row = ttk.Frame(tab); ids_row.pack(fill=tk.X, pady=3)
        # ttk.Label(ids_row, text="IDs (space separated):").pack(side=tk.LEFT)
        # ttk.Entry(ids_row, textvariable=self.nh_ids).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)

        # # Search
        # srch_row = ttk.Frame(tab); srch_row.pack(fill=tk.X, pady=3)
        # ttk.Label(srch_row, text="Search:").pack(side=tk.LEFT)
        # ttk.Entry(srch_row, textvariable=self.nh_query).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        # ttk.Label(srch_row, text="Page:").pack(side=tk.LEFT, padx=(8, 2))
        # ttk.Entry(srch_row, width=6, textvariable=self.nh_page).pack(side=tk.LEFT)

        # # Options
        # opt_row = ttk.Frame(tab); opt_row.pack(fill=tk.X, pady=5)
        # ttk.Checkbutton(opt_row, text="Download", variable=self.nh_download).pack(side=tk.LEFT)
        # ttk.Checkbutton(opt_row, text="CBZ", variable=self.nh_cbz).pack(side=tk.LEFT, padx=6)
        # ttk.Checkbutton(opt_row, text="PDF", variable=self.nh_pdf).pack(side=tk.LEFT, padx=6)
        # ttk.Label(opt_row, text="Delay (s):").pack(side=tk.LEFT, padx=(10, 2))
        # ttk.Entry(opt_row, width=6, textvariable=self.nh_delay).pack(side=tk.LEFT)

        # help_nh = (
            # "Tips:\n"
            # "- pip install nhentai\n"
            # "- To bypass Cloudflare rate limits, set both --cookie and --useragent.\n"
            # "- Use same IP/User-Agent as when the cookie was obtained.\n"
        # )
        # ttk.Label(tab, text=help_nh, justify="left").pack(anchor="w", pady=(4, 6))

        # btns = ttk.Frame(tab); btns.pack(fill=tk.X, pady=5)
        # self.btn_nh_go = ttk.Button(btns, text="Run nhentai", command=self._start_nhentai)
        # self.btn_nh_go.pack(side=tk.LEFT)

    # # ---------- Actions ----------
    # def _pick_out(self):
        # d = filedialog.askdirectory(title="Select Output Folder")
        # if d:
            # self.out_dir.set(d)

    # def _open_folder(self):
        # path = self.out_dir.get()
        # if not os.path.isdir(path):
            # messagebox.showerror("Error", "Output folder does not exist.")
            # return
        # if platform.system() == "Windows":
            # os.startfile(path)
        # elif platform.system() == "Darwin":
            # subprocess.Popen(["open", path])
        # else:
            # subprocess.Popen(["xdg-open", path])

    # def _use_as_batch_input(self):
        # # Tell App to switch to Batch tab and set input folder
        # self.app.use_folder_as_batch_input(self.out_dir.get())

    # # ---------- Logging ----------
    # def _log(self, text: str):
        # self.log.configure(state="normal")
        # self.log.insert("end", text)
        # self.log.see("end")
        # self.log.configure(state="disabled")

    # def _pump_logq(self):
        # try:
            # while True:
                # line = self.log_q.get_nowait()
                # self._log(line)
        # except queue.Empty:
            # pass
        # if not self.stop_flag.is_set():
            # self.after(100, self._pump_logq)

    # # ---------- Runner helpers ----------
    # def _which_or_module(self, cli_names, module_cmd):
        # """
        # Return a list representing the command to execute. Try CLIs in order,
        # else return ['python', '-m', module_cmd].
        # """
        # for name in cli_names:
            # if shutil.which(name):
                # return [name]
        # # Fallback to python -m module
        # py = shutil.which("python3") or shutil.which("python") or sys.executable
        # return [py, "-m", module_cmd]

    # def _start_mangadex(self):
        # url = self.md_url.get().strip()
        # if not url:
            # messagebox.showerror("Error", "Please enter a MangaDex URL.")
            # return
        # out = self.out_dir.get()
        # os.makedirs(out, exist_ok=True)

        # base = self._which_or_module(["mangadex-dl", "mangadex-downloader"], "mangadex_downloader")
        # cmd = base + [url, "-o", out]

        # self._run_cmd_threaded("MangaDex", cmd)

    # def _start_nhentai(self):
        # out = self.out_dir.get()
        # os.makedirs(out, exist_ok=True)

        # base = self._which_or_module(["nhentai"], "nhentai")
        # cmd = base + ["-o", out]

        # ua = self.nh_ua.get().strip()
        # ck = self.nh_cookie.get().strip()
        # if ua:
            # cmd += ["--useragent", ua]
        # if ck:
            # cmd += ["--cookie", ck]

        # mode = self.nh_mode.get()
        # if mode == "id":
            # ids = self.nh_ids.get().strip()
            # if not ids:
                # messagebox.showerror("Error", "Enter at least one ID.")
                # return
            # cmd += ["--id"] + ids.split()
        # elif mode == "search":
            # q = self.nh_query.get().strip()
            # if not q:
                # messagebox.showerror("Error", "Enter a search query.")
                # return
            # page = self.nh_page.get().strip() or "1"
            # cmd += ["--search", q, "--page", page]
            # if self.nh_download.get():
                # cmd.append("--download")
        # else:  # favorites
            # cmd += ["--favorites"]
            # if self.nh_download.get():
                # cmd.append("--download")
            # d = self.nh_delay.get().strip()
            # if d and d.isdigit():
                # cmd += ["--delay", d]

        # if self.nh_cbz.get():
            # cmd.append("--cbz")
        # if self.nh_pdf.get():
            # cmd.append("--pdf")

        # self._run_cmd_threaded("nhentai", cmd)

    # def _run_cmd_threaded(self, label, cmd):
        # if self.proc_thread and self.proc_thread.is_alive():
            # messagebox.showwarning("Busy", "A download is already in progress.")
            # return

        # self.stop_flag.clear()
        # self._log(f"\n[{label}] Running: {' '.join(cmd)}\n")
        # self.after(100, self._pump_logq)

        # def worker():
            # try:
                # proc = subprocess.Popen(
                    # cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                    # creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0
                # )
                # for line in iter(proc.stdout.readline, ''):
                    # self.log_q.put(line)
                # proc.stdout.close()
                # rc = proc.wait()
                # self.log_q.put(f"[{label}] Finished with code {rc}\n")
                # if rc == 0:
                    # self.log_q.put("Done. You can Open Folder or Use as Batch Input.\n")
            # except FileNotFoundError:
                # self.log_q.put(f"[{label}] Command not found. Is the tool installed?\n")
            # except Exception as e:
                # self.log_q.put(f"[{label}] Error: {e}\n")
            # finally:
                # self.stop_flag.set()

        # self.proc_thread = threading.Thread(target=worker, daemon=True)
        # self.proc_thread.start()


# =========================
# App
# =========================
class App:
    def __init__(self, root: tk.Tk, config):
        self.root = root
        self.root.title("Manga Image Processor")
        self.root.geometry("1200x800")
        self.config = config

        # Theme
        self.theme = ThemeManager(root)

        # Pipeline
        try:
            self.pipeline = ProcessorPipeline(config)
            print("[+] Components initialized successfully.")
        except Exception as e:
            messagebox.showerror("Initialization Error", f"Failed to initialize components: {e}")
            root.destroy()
            return

        # Header
        header = ttk.Frame(root, padding=(10, 10, 10, 0))
        header.pack(fill=tk.X)
        ttk.Label(header, text="").pack(side=tk.LEFT, expand=True)
        ttk.Button(header, text="Toggle Theme", command=self.theme.toggle_theme).pack(side=tk.RIGHT)

        # Tabs
        self.nb = ttk.Notebook(root)
        self.nb.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        self.tab_single = SingleImageTab(self.nb, self)
        self.tab_batch = BatchTab(self.nb, self)
        # self.tab_dl = DownloadersTab(self.nb, self)

        self.nb.add(self.tab_single, text="Single Image Processing")
        self.nb.add(self.tab_batch, text="Batch Folder Processing")
        # self.nb.add(self.tab_dl, text="Downloaders")

        # Status
        self.status = ttk.Label(root, text="Ready", padding=(10, 5))
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    def set_status(self, text: str):
        self.status.config(text=text)

    def use_folder_as_batch_input(self, folder: str):
        if not os.path.isdir(folder):
            messagebox.showerror("Error", "Output folder does not exist.")
            return
        # Switch to Batch tab & set input/output quickly
        self.nb.select(self.tab_batch)
        self.tab_batch.input_folder = folder
        # If user hasn’t chosen output, default to "<folder>_processed"
        default_out = folder.rstrip("/\\") + "_processed"
        os.makedirs(default_out, exist_ok=True)
        self.tab_batch.output_folder = default_out
        self.tab_batch.folder_status.config(
            text=f"In: ...{folder[-30:]} | Out: ...{default_out[-30:]}")
        self.tab_batch.btn_start.config(state=tk.NORMAL)
        threading.Thread(target=self.tab_batch._load_thumbs, daemon=True).start()


# =========================
# Entrypoint
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="GUI for Manga Image Processor")
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda', help='Device to use')
    
    # --- Paths (Refactored) ---
    # <-- FIX: Removed 'backend' from default paths
    parser.add_argument('--colorizer_path', default='networks/generator.zip')
    parser.add_argument('--extractor_path', default='networks/extractor.pth')
    parser.add_argument('--upscaler_path', default='networks/RealESRGAN_x4plus_anime_6B.pt')
    
    parser.add_argument('--upscaler_type', choices=['ESRGAN', 'GigaGAN'], default='ESRGAN')
    parser.add_argument('--no-upscale', dest='upscale', action='store_false', default=True)
    parser.add_argument('--no-colorize', dest='colorize', action='store_false', default=True)
    parser.add_argument('--no-denoise', dest='denoise', action='store_false', default=True)
    parser.add_argument('--upscale_factor', choices=[2, 4], default=4, type=int)
    parser.add_argument('--denoise_sigma', default=25, type=int)

    # Extra runtime tunables you already set later:
    args = parser.parse_args()
    args.upscaler_tile_size = 256
    args.colorizer_tile_size = 0
    args.tile_pad = 8
    args.colorized_image_size = 576
    return args


def main():
    config = parse_args()
    root = tk.Tk()
    app = App(root, config)
    initial_theme = "dark" if darkdetect.isDark() else "light"
    root.after(10, lambda: app.theme.set_initial(initial_theme))
    root.mainloop()


if __name__ == "__main__":
    main()