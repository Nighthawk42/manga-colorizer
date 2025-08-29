# train_model.py — Rich UI fine-tuner (Windows-safe, CSV+plot logging, graceful)

import os, sys, time, json, random, platform, logging, csv, datetime, shutil
from pathlib import Path
from typing import Optional

# ---- Paths (edit if needed) ----
GEN_ZIP = r"C:\Users\Nighthawk\Desktop\manga_colorize\networks\generator.zip"
OUT_DIR = r"C:\Users\Nighthawk\Desktop\manga_colorize\finetune_out"
PAUSE_ON_EXIT = False  # set True if you want a final "Press Enter..." pause

IS_WINDOWS = platform.system() == "Windows"

# ---- Third-party ----
import numpy as np
from PIL import Image, ImageFilter, ImageDraw

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import amp as torch_amp  # modern AMP API

# Optional plotting
HAVE_MPL = True
try:
    import matplotlib.pyplot as plt
except Exception:
    HAVE_MPL = False

# HuggingFace datasets (optional)
HAVE_DATASETS = True
try:
    from datasets import load_dataset
except Exception:
    HAVE_DATASETS = False

# Rich UI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.progress import (
    Progress, TextColumn, BarColumn, TimeElapsedColumn,
    TimeRemainingColumn
)
from rich.logging import RichHandler

console = Console()

# Import your model from the repo
sys.path.insert(0, str(Path(__file__).parent))
from networks.colorizer import Colorizer


# ----------------- Logging -----------------

def setup_logging(out_dir: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(out_dir) / "train.log"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):
        logger.removeHandler(h)

    ch = RichHandler(console=console, show_path=False, rich_tracebacks=True)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    logger.addHandler(ch)
    logger.addHandler(fh)
    logging.info(f"Logging to: {log_path}")


# ----------------- Utilities -----------------

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def set_seed(seed: int = 42, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def device_info(device: str):
    if device == "cuda":
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return f"[bold cyan]CUDA[/]: {name} (CC {cap[0]}.{cap[1]}), {mem:.1f} GB"
    return "[bold yellow]CPU[/]"

def mem_stats():
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / (1024**2)
        reserv = torch.cuda.memory_reserved() / (1024**2)
        return f"VRAM alloc {alloc:.0f}MB / reserved {reserv:.0f}MB"
    return "CPU memory"

def to_uint8(img01: np.ndarray) -> np.ndarray:
    return np.clip(img01 * 255.0, 0, 255).round().astype(np.uint8)

def save_image_row(path: str, images: list):
    if not images: return
    h, w, _ = images[0].shape
    from PIL import Image as _I
    grid = _I.new("RGB", (w * len(images), h))
    x = 0
    for im in images:
        grid.paste(_I.fromarray(im), (x, 0))
        x += w
    grid.save(path)

def rgb_to_L(rgb_np: np.ndarray) -> np.ndarray:
    return np.expand_dims(np.dot(rgb_np[..., :3], [0.299, 0.587, 0.114]), 2)

def random_scribbles(w, h, n=3):
    hint = Image.new("RGB", (w, h), (128, 128, 128))
    mask = Image.new("L", (w, h), 0)
    dh, dm = ImageDraw.Draw(hint), ImageDraw.Draw(mask)
    for _ in range(n):
        color = tuple(np.random.randint(0, 256, 3).tolist())
        import numpy as _np
        pts = [(_np.random.randint(0, w), _np.random.randint(0, h))]
        for _ in range(8):
            x = int(np.clip(pts[-1][0] + _np.random.randint(-w // 6, w // 6), 0, w - 1))
            y = int(np.clip(pts[-1][1] + _np.random.randint(-h // 6, h // 6), 0, h - 1))
            pts.append((x, y))
        width = _np.random.randint(8, 20)
        dh.line(pts, fill=color, width=width)
        dm.line(pts, fill=255, width=width)
    hint = hint.filter(ImageFilter.GaussianBlur(1.0))
    return np.asarray(hint), np.asarray(mask)

def state_dict_load(path: str):
    logging.info(f"Loading weights: {path}")
    sd = torch.load(path, map_location="cpu")
    if not isinstance(sd, dict):
        raise RuntimeError("Loaded object is not a state_dict (dict).")
    return sd

def latest_ckpt(dir_path: str) -> Optional[Path]:
    d = Path(dir_path)
    if not d.exists(): return None
    cks = sorted(d.glob("ckpt_step*.pt"), key=lambda p: p.stat().st_mtime)
    return cks[-1] if cks else None

def rotate_checkpoints(dir_path: str, keep: int = 5):
    d = Path(dir_path)
    cks = sorted(d.glob("ckpt_step*.pt"), key=lambda p: p.stat().st_mtime)
    for p in cks[:-keep]:
        try: p.unlink()
        except: pass

def safe_prompt(prompt: str, default: str) -> str:
    try:
        val = Prompt.ask(prompt, default=default)
        return val
    except (KeyboardInterrupt, EOFError):
        console.print("\n[bold yellow]Input cancelled by user.[/] Using default.")
        return default

# ---- Loss CSV / Plot ----

def init_loss_csv(out_dir: str) -> str:
    csv_path = os.path.join(out_dir, "loss_log.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["step", "loss", "images_per_sec", "elapsed_sec"])
    return csv_path

def append_loss(csv_path: str, step: int, loss_val: float, ips: float, elapsed: float):
    try:
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([step, f"{loss_val:.6f}", f"{ips:.3f}", f"{int(elapsed)}"])
    except Exception as e:
        logging.warning(f"Could not write loss CSV: {e}")

def try_plot_loss(csv_path: str, out_dir: str):
    if not HAVE_MPL:
        logging.info("matplotlib not installed; skipping loss plot.")
        return
    try:
        steps, losses = [], []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                steps.append(int(row["step"]))
                losses.append(float(row["loss"]))
        if steps and losses:
            plt.figure()
            plt.plot(steps, losses)
            plt.xlabel("step")
            plt.ylabel("loss (L1 + 0.1*aux)")
            plt.title("Training Loss")
            plt.tight_layout()
            p = os.path.join(out_dir, "loss_curve.png")
            plt.savefig(p, dpi=120)
            plt.close()
            logging.info(f"Saved loss plot: {p}")
    except Exception as e:
        logging.warning(f"Could not plot loss curve: {e}")


# ----------------- Datasets -----------------

class FolderPairs(Dataset):
    def __init__(self, root: str, crop: int = 512, use_scribbles: bool = True):
        self.root = root
        self.crop = crop
        self.use_scribbles = use_scribbles
        exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
        self.paths = [str(Path(dp)/fn) for dp,_,files in os.walk(root)
                      for fn in files if fn.lower().endswith(exts)]
        if not self.paths:
            raise RuntimeError(f"No images found under: {root}")

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        s = max(self.crop, min(img.size))
        img = img.resize((s, s), Image.BICUBIC)
        import numpy as _np
        x0 = _np.random.randint(0, s - self.crop + 1)
        y0 = _np.random.randint(0, s - self.crop + 1)
        img = img.crop((x0, y0, x0 + self.crop, y0 + self.crop))

        rgb = np.asarray(img).astype(np.float32) / 255.0
        L = rgb_to_L(rgb)

        if self.use_scribbles:
            hint_rgb, hint_mask = random_scribbles(self.crop, self.crop, n=np.random.randint(2, 5))
            hint_rgb = (hint_rgb.astype(np.float32) / 255.0 - 0.5) / 0.5
            hint_mask = (hint_mask.astype(np.float32) / 255.0)[..., None]
        else:
            hint_rgb = np.zeros((self.crop, self.crop, 3), dtype=np.float32)
            hint_mask = np.zeros((self.crop, self.crop, 1), dtype=np.float32)

        inp = np.concatenate([L, hint_rgb * hint_mask, hint_mask], axis=2)
        inp = torch.from_numpy(inp).permute(2, 0, 1)
        tgt = torch.from_numpy(rgb).permute(2, 0, 1) * 2 - 1
        return inp, tgt

class HFPairs(Dataset):
    def __init__(self, dataset_id: str, split: str = "train",
                 crop: int = 512, streaming: bool = False, use_scribbles: bool = False):
        if not HAVE_DATASETS:
            raise RuntimeError("Please install: pip install datasets")
        self.crop = crop
        self.use_scribbles = use_scribbles
        self.streaming = streaming
        self.ds = load_dataset(dataset_id, split=split, streaming=streaming)
        feats = getattr(self.ds, "features", None)
        self.has_bw = bool(feats and "bw_image" in feats)
        if streaming:
            self._it = iter(self.ds)

    def __len__(self): return len(self.ds) if not self.streaming else 10**9

    def _prep(self, bw_pil: Optional[Image.Image], color_pil: Image.Image):
        color = color_pil.convert("RGB")
        s = max(self.crop, min(color.size))
        color = color.resize((s, s), Image.BICUBIC)
        import numpy as _np
        x0 = _np.random.randint(0, s - self.crop + 1)
        y0 = _np.random.randint(0, s - self.crop + 1)
        color = color.crop((x0, y0, x0 + self.crop, y0 + self.crop))
        rgb = np.asarray(color).astype(np.float32) / 255.0

        if bw_pil is None:
            L = rgb_to_L(rgb)
        else:
            bw = bw_pil.convert("L").resize((s, s), Image.BICUBIC).crop((x0, y0, x0 + self.crop, y0 + self.crop))
            L = (np.asarray(bw, dtype=np.float32) / 255.0)[..., None]

        if self.use_scribbles:
            hint_rgb, hint_mask = random_scribbles(self.crop, self.crop, n=np.random.randint(2, 5))
            hint_rgb = (hint_rgb.astype(np.float32) / 255.0 - 0.5) / 0.5
            hint_mask = (hint_mask.astype(np.float32) / 255.0)[..., None]
        else:
            hint_rgb = np.zeros((self.crop, self.crop, 3), dtype=np.float32)
            hint_mask = np.zeros((self.crop, self.crop, 1), dtype=np.float32)

        inp = np.concatenate([L, hint_rgb * hint_mask, hint_mask], axis=2)
        inp = torch.from_numpy(inp).permute(2, 0, 1)
        tgt = torch.from_numpy(rgb).permute(2, 0, 1) * 2 - 1
        return inp, tgt

    def __getitem__(self, idx):
        ex = next(self._it) if self.streaming else self.ds[int(idx)]
        color = ex.get("color_image") or ex.get("image")
        if color is None:
            raise RuntimeError("Example missing 'color_image' (or 'image').")
        bw = ex.get("bw_image") if self.has_bw else None
        return self._prep(bw, color)


# ----------------- Training -----------------

def build_dataloader(source_type: str, crop: int, batch: int, workers: int, use_scribbles: bool):
    if source_type == "folder":
        folder = safe_prompt(f"[bold]Folder of COLOR images[/] [default {OUT_DIR}\\demo_images]", f"{OUT_DIR}\\demo_images")
        ds = FolderPairs(folder, crop=crop, use_scribbles=use_scribbles)
        desc = f"Folder: {folder}  (N={len(ds)})"
    else:
        if not HAVE_DATASETS:
            raise RuntimeError("Install datasets: pip install datasets")
        ds_id = safe_prompt("HF dataset id", "MichaelP84/manga-colorization-dataset")
        split = safe_prompt("Split", "train")
        streaming = Confirm.ask("Enable streaming?", default=False)
        ds = HFPairs(ds_id, split=split, crop=crop, streaming=streaming, use_scribbles=use_scribbles)
        n = len(ds) if not streaming else "∞"
        desc = f"HF: {ds_id} / {split} (N={n}, streaming={streaming})"

    # Windows-safe DataLoader defaults:
    num_workers = 0 if IS_WINDOWS else max(0, workers)
    pin_memory = (not IS_WINDOWS)

    dl_kwargs = dict(
        batch_size=batch,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        worker_init_fn=lambda _: set_seed()
    )
    if (not IS_WINDOWS) and num_workers > 0:
        dl_kwargs["prefetch_factor"] = 2
        dl_kwargs["persistent_workers"] = False

    dl = DataLoader(ds, **dl_kwargs)
    return dl, desc

def train_loop(cfg: dict):
    ensure_dir(OUT_DIR)
    ckpt_dir = os.path.join(OUT_DIR, "checkpoints"); ensure_dir(ckpt_dir)
    sample_dir = os.path.join(OUT_DIR, "samples"); ensure_dir(sample_dir)

    # logging file
    setup_logging(OUT_DIR)
    with open(os.path.join(OUT_DIR, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(Panel.fit(device_info(device), border_style="cyan"))
    if device != "cuda":
        console.print("[yellow]CUDA not available. CPU training will be slow.[/]")
    logging.info(mem_stats())

    net = Colorizer().to(device)
    sd = state_dict_load(GEN_ZIP)
    missing, unexpected = net.generator.load_state_dict(sd, strict=False)
    if missing: logging.info(f"Missing keys: {len(missing)}")
    if unexpected: logging.info(f"Unexpected keys: {len(unexpected)}")

    # freeze encoder warmup
    for p in net.generator.encoder.parameters(): p.requires_grad = False
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()),
                            lr=cfg["lr"], betas=(0.9,0.999), weight_decay=1e-4)
    scaler = torch_amp.GradScaler(device="cuda" if device=="cuda" else "cpu", enabled=(device=="cuda"))
    l1 = nn.L1Loss()

    dl, ds_desc = build_dataloader(cfg["source_type"], cfg["crop"], cfg["batch_size"], cfg["workers"], cfg["use_scribbles"])
    console.print(Panel.fit(f"[bold]Data:[/]\n{ds_desc}\n\n[dim]{mem_stats()}[/]"))
    logging.info(ds_desc)

    # resume?
    start_step = 0
    latest = latest_ckpt(ckpt_dir)
    if latest and Confirm.ask(f"Resume from {latest.name}?", default=True):
        ckpt = torch.load(latest, map_location="cpu")
        net.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        scaler.load_state_dict(ckpt["scaler"])
        start_step = ckpt["step"]
        logging.info(f"Resumed from step {start_step}")

    # init loss CSV
    csv_path = init_loss_csv(OUT_DIR)

    # progress bar
    progress = Progress(
        TextColumn("[bold]Step[/] {task.completed}/{task.total}"),
        BarColumn(),
        TextColumn("loss {task.fields[loss]:.4f}"),
        TextColumn("{task.fields[ips]}"),
        TimeElapsedColumn(),
        TextColumn("ETA"),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )
    task = progress.add_task("train", total=cfg["steps"], loss=0.0, ips="0.0 img/s")

    accum = max(1, cfg["grad_accum"])
    unfreeze_at = min(1000, cfg["steps"] // 5)
    next_sample = cfg["sample_every"]
    next_ckpt = cfg["save_every"]
    grad_clip = cfg["grad_clip"]

    seen = 0
    start_time = time.time()
    step = start_step

    try:
        with progress:
            while step < cfg["steps"]:
                for inp, tgt in dl:
                    step += 1
                    inp = inp.to(device, non_blocking=True)
                    tgt = tgt.to(device, non_blocking=True)
                    L = inp[:, :1]; hint4 = inp[:, 1:]
                    x = torch.cat([L, hint4], dim=1)

                    # Modern autocast API
                    with torch_amp.autocast(device_type=("cuda" if device=="cuda" else "cpu"), enabled=(device=="cuda")):
                        pred, aux = net(x)
                        loss = l1(pred, tgt) + 0.1*l1(aux, tgt)
                        loss = loss / accum

                    scaler.scale(loss).backward()

                    if step % accum == 0:
                        if grad_clip:
                            scaler.unscale_(opt)
                            nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
                        scaler.step(opt); scaler.update()
                        opt.zero_grad(set_to_none=True)

                    # unfreeze
                    if step == unfreeze_at:
                        for p in net.generator.encoder.parameters(): p.requires_grad = True
                        for g in opt.param_groups: g["lr"] = cfg["lr"] * 0.25
                        logging.info("Unfroze encoder; lowered LR")
                        logging.info(mem_stats())

                    # progress + CSV
                    seen += inp.size(0)
                    ips_float = seen / max(1e-6, time.time() - start_time)
                    ips = f"{ips_float:.1f} img/s"
                    progress.update(task, advance=1, loss=(loss.item()*accum), ips=ips)
                    append_loss(csv_path, step=step, loss_val=(loss.item()*accum),
                                ips=ips_float, elapsed=(time.time() - start_time))

                    # sample
                    if step >= next_sample or step == cfg["steps"]:
                        try:
                            with torch.no_grad():
                                pv = (pred[0].clamp(-1,1).add(1).mul(0.5)).cpu().permute(1,2,0).numpy()
                                tv = (tgt[0].clamp(-1,1).add(1).mul(0.5)).cpu().permute(1,2,0).numpy()
                                lv = L[0,0].cpu().numpy(); lv = np.repeat(lv[...,None], 3, axis=2)
                                outp = os.path.join(OUT_DIR, "samples", f"sample_step{step}.jpg")
                                save_image_row(outp, [to_uint8(lv), to_uint8(pv), to_uint8(tv)])
                                logging.info(f"Sample saved: {outp}")
                        except Exception as e:
                            logging.warning(f"Sample save failed: {e}")
                        next_sample += cfg["sample_every"]

                    # ckpt
                    if step >= next_ckpt or step == cfg["steps"]:
                        ck = {
                            "step": step,
                            "model": net.state_dict(),
                            "opt": opt.state_dict(),
                            "scaler": scaler.state_dict(),
                            "config": cfg,
                        }
                        pth = os.path.join(OUT_DIR, "checkpoints", f"ckpt_step{step}.pt")
                        torch.save(ck, pth)
                        rotate_checkpoints(os.path.join(OUT_DIR, "checkpoints"), keep=cfg["keep_last"])
                        logging.info(f"Checkpoint: {pth}")
                        next_ckpt += cfg["save_every"]

                    if step >= cfg["steps"]:
                        break

    except KeyboardInterrupt:
        console.print("\n[bold yellow]Training interrupted by user.[/]")
        ck = {
            "step": step,
            "model": net.state_dict(),
            "opt": opt.state_dict(),
            "scaler": scaler.state_dict(),
            "config": cfg,
        }
        pth = os.path.join(OUT_DIR, "checkpoints", f"ckpt_step{step}_INT.pt")
        torch.save(ck, pth)
        logging.info(f"Saved interrupt checkpoint: {pth}")

    finally:
        # Export generator-only weights for your app
        final_zip = os.path.join(OUT_DIR, f"generator_finetuned_step{step}.zip")
        torch.save(net.generator.state_dict(), final_zip)
        console.print(f"[bold green]Exported generator weights:[/] {final_zip}")
        logging.info(f"Exported generator weights: {final_zip}")

        # Loss curve (optional)
        try_plot_loss(csv_path, OUT_DIR)

        # ---- Compatibility export ----
        # Backup old app weight file, then drop-in replace it with the new one.
        try:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            if os.path.exists(GEN_ZIP):
                bak = f"{GEN_ZIP}.{ts}.bak"
                shutil.copy2(GEN_ZIP, bak)
                logging.info(f"Backed up previous generator.zip -> {bak}")
            shutil.copy2(final_zip, GEN_ZIP)
            console.print(f"[bold green]Updated app weights:[/] {GEN_ZIP}")
            logging.info(f"Updated app weights: {GEN_ZIP}")
        except Exception as e:
            logging.warning(f"Could not overwrite app weights: {e}")
            console.print("[yellow]Note:[/] Could not overwrite app weights automatically. "
                          f"Copy {final_zip} to {GEN_ZIP} manually.")


# ----------------- Main (Rich prompts, graceful) -----------------

def main():
    console.print(Panel.fit("Manga Colorizer — Fine-tune", border_style="magenta"))
    if not os.path.exists(GEN_ZIP):
        console.print(f"[red]Model not found:[/] {GEN_ZIP}\nEdit GEN_ZIP at top of the script.")
        return

    set_seed(42, deterministic=True)

    # Prompts (graceful + defaults; workers default to 0 on Windows)
    src = safe_prompt("Data source — (1) Local folder (2) Hugging Face", "2")
    source_type = "folder" if src.strip() == "1" else "hf"

    crop = int(safe_prompt("Crop size (divisible by 32)", "512"))
    steps = int(safe_prompt("Total training steps", "4000"))
    batch = int(safe_prompt("Batch size", "4"))
    lr    = float(safe_prompt("Learning rate", "2e-4"))
    accum = int(safe_prompt("Grad accumulation (for bigger effective batch)", "1"))
    default_workers = "0" if IS_WINDOWS else "2"
    workers = int(safe_prompt("Dataloader workers", default_workers))
    save_every = int(safe_prompt("Save checkpoint every N steps", "500"))
    sample_every = int(safe_prompt("Save sample image every N steps", "200"))
    keep_last = int(safe_prompt("Keep last K checkpoints", "5"))
    use_scribbles = Confirm.ask("Teach scribble hints too?", default=False)
    gc = float(safe_prompt("Gradient clip (0 = off)", "0"))
    grad_clip = None if gc <= 0 else gc

    cfg = {
        "source_type": source_type,
        "crop": crop,
        "steps": steps,
        "batch_size": batch,
        "lr": lr,
        "grad_accum": accum,
        "workers": workers,
        "save_every": save_every,
        "sample_every": sample_every,
        "keep_last": keep_last,
        "use_scribbles": use_scribbles,
        "grad_clip": grad_clip,
    }

    # Show config table
    tbl = Table(title="Run Config", show_header=False, box=None)
    for k, v in cfg.items():
        tbl.add_row(f"[cyan]{k}[/]", f"{v}")
    console.print(tbl)

    train_loop(cfg)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Exited by user during setup.[/]")
    finally:
        if PAUSE_ON_EXIT:
            try:
                input("\nDone. Press Enter to exit...")
            except Exception:
                pass
