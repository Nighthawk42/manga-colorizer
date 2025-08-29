#!/usr/bin/env python3
# Minimal cloud trainer: argparse only, CUDA-friendly, HF or folder, multi-dataset ratios

import os, sys, time, json, math, random, logging, argparse, shutil, datetime
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageFilter, ImageDraw

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from torch import amp as torch_amp

# ---- optional HF datasets
HAVE_DATASETS = True
try:
    from datasets import load_dataset
except Exception:
    HAVE_DATASETS = False

# repo-local import
sys.path.insert(0, str(Path(__file__).parent))
from networks.colorizer import Colorizer

# ---------------- utils ----------------

def set_seed(seed=42, deterministic=True):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def to_uint8(img01: np.ndarray) -> np.ndarray:
    return np.clip(img01 * 255.0, 0, 255).round().astype(np.uint8)

def rgb_to_L(rgb_np: np.ndarray) -> np.ndarray:
    return np.expand_dims(np.dot(rgb_np[..., :3], [0.299, 0.587, 0.114]), 2)

def random_scribbles(w, h, n=3):
    hint = Image.new("RGB", (w, h), (128, 128, 128))
    mask = Image.new("L", (w, h), 0)
    dh, dm = ImageDraw.Draw(hint), ImageDraw.Draw(mask)
    for _ in range(n):
        color = tuple(np.random.randint(0, 256, 3).tolist())
        pts = [(np.random.randint(0, w), np.random.randint(0, h))]
        for _ in range(8):
            x = int(np.clip(pts[-1][0] + np.random.randint(-w//6, w//6), 0, w-1))
            y = int(np.clip(pts[-1][1] + np.random.randint(-h//6, h//6), 0, h-1))
            pts.append((x, y))
        width = np.random.randint(8, 20)
        dh.line(pts, fill=color, width=width)
        dm.line(pts, fill=255, width=width)
    hint = hint.filter(ImageFilter.GaussianBlur(1.0))
    return np.asarray(hint), np.asarray(mask)

def save_triplet(path, L, pred, tgt):
    L3 = np.repeat(L[..., None], 3, axis=2)
    row = np.concatenate([to_uint8(L3), to_uint8(pred), to_uint8(tgt)], axis=1)
    Image.fromarray(row).save(path)

def parse_id_ratio(s: str) -> List[Tuple[str, float]]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    ids, ratios = [], []
    for p in parts:
        if "@ " in p:  # in case user types "@ "
            p = p.replace("@ ", "@")
        if "@" in p:
            did, r = p.split("@", 1)
            ids.append(did.strip()); ratios.append(float(r))
        else:
            ids.append(p); ratios.append(1.0)
    ssum = sum(ratios)
    ratios = [r/ssum for r in ratios]
    return list(zip(ids, ratios))

# ---------------- datasets ----------------

class FolderPairs(Dataset):
    def __init__(self, root: str, crop: int = 512, scribbles: bool = False):
        exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
        self.paths = [str(Path(dp)/fn) for dp,_,files in os.walk(root)
                      for fn in files if fn.lower().endswith(exts)]
        if not self.paths:
            raise RuntimeError(f"No images under {root}")
        self.crop = crop
        self.scribbles = scribbles

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        s = max(self.crop, min(img.size))
        img = img.resize((s, s), Image.BICUBIC)
        x0 = np.random.randint(0, s - self.crop + 1)
        y0 = np.random.randint(0, s - self.crop + 1)
        img = img.crop((x0, y0, x0 + self.crop, y0 + self.crop))
        rgb = np.asarray(img).astype(np.float32)/255.0
        L = rgb_to_L(rgb)

        if self.scribbles:
            hint_rgb, hint_mask = random_scribbles(self.crop, self.crop, n=np.random.randint(2,5))
            hint_rgb = (hint_rgb.astype(np.float32)/255.0 - 0.5)/0.5
            hint_mask = (hint_mask.astype(np.float32)/255.0)[..., None]
        else:
            hint_rgb = np.zeros((self.crop, self.crop, 3), np.float32)
            hint_mask = np.zeros((self.crop, self.crop, 1), np.float32)

        inp = np.concatenate([L, hint_rgb*hint_mask, hint_mask], axis=2)
        inp = torch.from_numpy(inp).permute(2,0,1)
        tgt = torch.from_numpy(rgb).permute(2,0,1)*2 - 1
        return inp, tgt

class HFPairs(Dataset):
    def __init__(self, dataset_id: str, split="train", crop=512, streaming=False, scribbles=False):
        if not HAVE_DATASETS:
            raise RuntimeError("pip install datasets")
        self.ds = load_dataset(dataset_id, split=split, streaming=streaming)
        self.crop = crop
        self.streaming = streaming
        self.scribbles = scribbles
        self.features = getattr(self.ds, "features", None)
        self.has_bw = bool(self.features and "bw_image" in self.features)
        if streaming:
            self._it = iter(self.ds)

    def __len__(self): return len(self.ds) if not self.streaming else 10**9

    def _prep(self, bw_pil: Optional[Image.Image], color_pil: Image.Image):
        color = color_pil.convert("RGB")
        s = max(self.crop, min(color.size))
        color = color.resize((s, s), Image.BICUBIC)
        x0 = np.random.randint(0, s - self.crop + 1)
        y0 = np.random.randint(0, s - self.crop + 1)
        color = color.crop((x0, y0, x0 + self.crop, y0 + self.crop))
        rgb = np.asarray(color).astype(np.float32)/255.0

        if bw_pil is None:
            L = rgb_to_L(rgb)
        else:
            bw = bw_pil.convert("L").resize((s, s), Image.BICUBIC).crop((x0, y0, x0 + self.crop, y0 + self.crop))
            L = (np.asarray(bw, dtype=np.float32)/255.0)[..., None]

        if self.scribbles:
            hint_rgb, hint_mask = random_scribbles(self.crop, self.crop, n=np.random.randint(2,5))
            hint_rgb = (hint_rgb.astype(np.float32)/255.0 - 0.5)/0.5
            hint_mask = (hint_mask.astype(np.float32)/255.0)[..., None]
        else:
            hint_rgb = np.zeros((self.crop, self.crop, 3), np.float32)
            hint_mask = np.zeros((self.crop, self.crop, 1), np.float32)

        inp = np.concatenate([L, hint_rgb*hint_mask, hint_mask], axis=2)
        inp = torch.from_numpy(inp).permute(2,0,1)
        tgt = torch.from_numpy(rgb).permute(2,0,1)*2 - 1
        return inp, tgt

    def __getitem__(self, idx):
        ex = next(self._it) if self.streaming else self.ds[int(idx)]
        color = ex.get("color_image") or ex.get("image")
        if color is None:
            raise RuntimeError("Example missing 'color_image' or 'image'")
        bw = ex.get("bw_image") if self.has_bw else None
        return self._prep(bw, color)

# -------------- building dataloader --------------

def build_dl_from_args(args):
    if args.source == "folder":
        ds = FolderPairs(args.folder, crop=args.crop, scribbles=args.scribbles)
        n = len(ds)
        num_workers = max(0, args.workers)
        dl = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        num_workers=num_workers, pin_memory=True,
                        drop_last=True, worker_init_fn=lambda _: set_seed())
        desc = f"Folder: {args.folder} (N={n})"
        return dl, desc

    # HF (single or multi)
    id_ratios = parse_id_ratio(args.datasets)
    datasets = []
    for ds_id, _r in id_ratios:
        d = HFPairs(ds_id, split=args.split, crop=args.crop,
                    streaming=args.streaming, scribbles=args.scribbles)
        datasets.append(d)

    if args.streaming:
        class StreamMix(Dataset):
            def __init__(self, dsets): self.ds = dsets
            def __len__(self): return 10**9
            def __getitem__(self, idx): return self.ds[idx % len(self.ds)][idx]
        mixed = StreamMix(datasets)
        sampler = None
        n = "∞"
    else:
        mixed = ConcatDataset(datasets)
        # weights by dataset id
        weights, membership = [], []
        ds_weight = {ds_id: ratio for ds_id, ratio in id_ratios}
        for (ds_id, ratio), d in zip(id_ratios, datasets):
            membership += [ds_id] * len(d)
        for ds_id in membership:
            weights.append(ds_weight[ds_id])
        sampler = WeightedRandomSampler(weights, num_samples=len(membership), replacement=True)
        n = sum(len(d) for d in datasets)

    num_workers = max(0, args.workers)
    dl = DataLoader(mixed,
                    batch_size=args.batch,
                    shuffle=False if sampler else True,
                    sampler=sampler,
                    num_workers=num_workers,
                    pin_memory=True,
                    drop_last=True,
                    worker_init_fn=lambda _: set_seed())
    parts = [f"{ds_id}@{ratio:.2f}" for ds_id, ratio in id_ratios]
    desc = f"HF mix: {', '.join(parts)} / {args.split} (N={n}, streaming={args.streaming})"
    return dl, desc

# ---------------- training ----------------

def train(args):
    set_seed(42, deterministic=True)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "samples"), exist_ok=True)

    # logging
    log_path = os.path.join(args.out_dir, "train.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")]
    )
    logging.info(f"Args: {vars(args)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        logging.warning("CUDA not available; training on CPU will be slow.")
    else:
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    # model
    net = Colorizer().to(device)

    # load base weights (generator.zip state_dict)
    if not os.path.exists(args.gen_zip):
        raise FileNotFoundError(f"Base weights not found: {args.gen_zip}")
    base_sd = torch.load(args.gen_zip, map_location="cpu")
    missing, unexpected = net.generator.load_state_dict(base_sd, strict=False)
    if missing: logging.info(f"Missing keys: {len(missing)}")
    if unexpected: logging.info(f"Unexpected keys: {len(unexpected)}")

    # freeze encoder warmup
    for p in net.generator.encoder.parameters(): p.requires_grad = False

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()),
                            lr=args.lr, betas=(0.9,0.999), weight_decay=1e-4)
    scaler = torch_amp.GradScaler(device="cuda" if device=="cuda" else "cpu", enabled=(device=="cuda"))
    l1 = nn.L1Loss()

    dl, desc = build_dl_from_args(args)
    logging.info(f"Data: {desc}")

    # optional resume
    step = 0
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location="cpu")
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        step = ck.get("step", 0)
        logging.info(f"Resumed from: {args.resume} (step {step})")

    # schedule: unfreeze encoder after 20% of remaining steps
    unfreeze_at = step + max(1000, (args.steps - step)//5)
    next_ckpt = min(args.steps, step + args.save_every)
    next_sample = min(args.steps, step + args.sample_every)

    start = time.time()
    seen = 0

    try:
        while step < args.steps:
            for inp, tgt in dl:
                step += 1
                inp = inp.to(device, non_blocking=True)
                tgt = tgt.to(device, non_blocking=True)
                x = inp  # our Colorizer already expects [L, hint*mask, mask] in channels

                with torch_amp.autocast(device_type=("cuda" if device=="cuda" else "cpu"), enabled=(device=="cuda")):
                    pred, aux = net(x)
                    loss = l1(pred, tgt) + 0.1*l1(aux, tgt)
                    loss = loss / max(1, args.grad_accum)

                scaler.scale(loss).backward()
                if step % max(1, args.grad_accum) == 0:
                    if args.grad_clip > 0:
                        scaler.unscale_(opt)
                        nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
                    scaler.step(opt); scaler.update()
                    opt.zero_grad(set_to_none=True)

                if step == unfreeze_at:
                    for p in net.generator.encoder.parameters(): p.requires_grad = True
                    for g in opt.param_groups: g["lr"] = args.lr * 0.25
                    logging.info(f"Unfroze encoder at step {step}, lowered LR to {args.lr*0.25:g}")

                # logging
                seen += inp.size(0)
                if step % 50 == 0 or step in (1, args.steps):
                    ips = seen / max(1e-6, time.time()-start)
                    logging.info(f"Step {step}/{args.steps}  loss {loss.item()*max(1,args.grad_accum):.4f}  {ips:.1f} img/s")

                # sample
                if step >= next_sample or step == args.steps:
                    try:
                        with torch.no_grad():
                            L = inp[0,0].detach().cpu().numpy()
                            pv = (pred[0].clamp(-1,1).add(1).mul(0.5)).cpu().permute(1,2,0).numpy()
                            tv = (tgt[0].clamp(-1,1).add(1).mul(0.5)).cpu().permute(1,2,0).numpy()
                            sp = os.path.join(args.out_dir, "samples", f"sample_step{step}.jpg")
                            save_triplet(sp, L, pv, tv)
                            logging.info(f"Sample: {sp}")
                    except Exception as e:
                        logging.warning(f"Sample failed: {e}")
                    next_sample += args.sample_every

                # checkpoint
                if step >= next_ckpt or step == args.steps:
                    ck = {
                        "step": step,
                        "model": net.state_dict(),
                        "opt": opt.state_dict(),
                        "scaler": scaler.state_dict(),
                        "config": vars(args),
                    }
                    pth = os.path.join(args.out_dir, "checkpoints", f"ckpt_step{step}.pt")
                    torch.save(ck, pth)
                    logging.info(f"Checkpoint: {pth}")
                    # rotate last K
                    keep = args.keep_last
                    cks = sorted(Path(args.out_dir, "checkpoints").glob("ckpt_step*.pt"), key=lambda p: p.stat().st_mtime)
                    for old in cks[:-keep]:
                        try: old.unlink()
                        except: pass
                    next_ckpt += args.save_every

                if step >= args.steps:
                    break

    except KeyboardInterrupt:
        logging.info("Interrupted by user, saving interrupt checkpoint…")
        ck = {
            "step": step,
            "model": net.state_dict(),
            "opt": opt.state_dict(),
            "scaler": scaler.state_dict(),
            "config": vars(args),
        }
        pth = os.path.join(args.out_dir, "checkpoints", f"ckpt_step{step}_INT.pt")
        torch.save(ck, pth)
        logging.info(f"Saved: {pth}")

    # final export
    final_zip = os.path.join(args.out_dir, f"generator_finetuned_step{step}.zip")
    torch.save(net.generator.state_dict(), final_zip)
    logging.info(f"Exported generator: {final_zip}")

    # copy into app path
    try:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if os.path.exists(args.gen_zip):
            bak = f"{args.gen_zip}.{ts}.bak"
            shutil.copy2(args.gen_zip, bak)
            logging.info(f"Backup: {bak}")
        shutil.copy2(final_zip, args.gen_zip)
        logging.info(f"Updated app weights: {args.gen_zip}")
    except Exception as e:
        logging.warning(f"Could not overwrite app weights: {e}")

# ---------------- main ----------------

def main():
    parser = argparse.ArgumentParser(
        description="Manga Colorizer — Cloud Trainer (non-interactive)")
    # IO
    parser.add_argument("--gen-zip", default="networks/generator.zip", help="base generator weights path")
    parser.add_argument("--out-dir", default="finetune_out", help="output directory")
    parser.add_argument("--resume", default="", help="path to checkpoint .pt to resume")
    # data
    parser.add_argument("--source", choices=["hf","folder"], default="hf")
    parser.add_argument("--datasets", default="MichaelP84/manga-colorization-dataset",
                        help="HF ids, comma separated, optional @ratio (e.g. id1@0.7,id2@0.3)")
    parser.add_argument("--split", default="train")
    parser.add_argument("--streaming", action="store_true", help="HF streaming mode")
    parser.add_argument("--folder", default="data", help="folder of COLOR images when --source folder")
    parser.add_argument("--crop", type=int, default=512)
    parser.add_argument("--scribbles", action="store_true", help="teach scribble hints")
    # train
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--sample-every", type=int, default=200)
    parser.add_argument("--keep-last", type=int, default=5)
    parser.add_argument("--grad-clip", type=float, default=0.0)

    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()
