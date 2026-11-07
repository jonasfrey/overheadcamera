#!/usr/bin/env python3
"""
Local image-to-image worker for Overhead Camera.

The app starts this script in a Python environment of its own (created and filled with the
packages below on first use) or with the interpreter chosen under *Inference*, and talks to it
over stdin/stdout, one JSON object per line. The model (SDXL-Turbo by default, through Hugging
Face diffusers) is loaded once and kept in memory until the app stops the worker or exits. The
first run downloads the model (several GB) into the Hugging Face cache.

Needs:  pip install torch diffusers transformers accelerate pillow

Commands (stdin), one per line:
  {"cmd": "generate", "image": <base64 JPEG or PNG>, "prompt": "...",
   "model": "stabilityai/sdxl-turbo", "device": "auto", "size": 512,
   "strength": 0.5, "steps": 4, "seed": -1}
  {"cmd": "cancel"}                       stops the generate in progress

Events (stdout), one per line:
  {"event": "ready", "python": "3.12.3"}  once at startup
  {"event": "stage", "text": "..."}       what the worker is doing
  {"event": "progress", "step": 1, "steps": 4}
  {"event": "done", "image": <base64 PNG>, "width": 512, "height": 384}
  {"event": "error", "message": "..."}    "cancelled" after a cancel; "missing": true when
                                          torch or diffusers can't be imported
"""
import base64
import io
import json
import math
import os
import queue
import sys
import threading
import time

PROTOCOL = sys.stdout
# Whatever a library prints must not get mixed into the protocol.
sys.stdout = sys.stderr

INSTALL_HINT = (
    "Install the packages for the local model into the Python chosen under Inference: "
    "pip install torch diffusers transformers accelerate pillow"
)
# SDXL works on latents of 1/8 the image size that are downsampled further inside the model;
# 64 keeps every stage an integer and is what SDXL-Turbo was trained with.
MULTIPLE = 64


def send(**event):
    try:
        PROTOCOL.write(json.dumps(event) + "\n")
        PROTOCOL.flush()
    except (BrokenPipeError, OSError):
        # The app is gone.
        os._exit(0)


class Cancelled(Exception):
    pass


def fit(size, max_side):
    """`size` scaled to a long side of `max_side`, each side a multiple of MULTIPLE."""
    w, h = size
    scale = max_side / max(w, h)
    round_side = lambda v: max(MULTIPLE, int(round(v * scale / MULTIPLE)) * MULTIPLE)
    return round_side(w), round_side(h)


def describe(e):
    text = " ".join(str(e).split())[:400] or type(e).__name__
    if "out of memory" in text.lower():
        text += " (try a smaller size, or device cpu)"
    return text


class Worker:
    def __init__(self):
        self.pipe = None
        self.loaded = None  # (model, device) of self.pipe
        self.cancel = threading.Event()

    def check(self):
        if self.cancel.is_set():
            raise Cancelled()

    @staticmethod
    def pick_device(torch, wanted):
        wanted = (wanted or "auto").strip()
        if wanted != "auto":
            return wanted
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"

    def load(self, model, device):
        import torch
        from diffusers import AutoPipelineForImage2Image

        device = self.pick_device(torch, device)
        if self.loaded == (model, device):
            return
        if self.pipe is not None:
            self.pipe, self.loaded = None, None
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        dtype = torch.float32 if device == "cpu" else torch.float16
        send(event="stage", text=f"Loading {model} on {device} (downloaded the first time)…")
        started = time.monotonic()
        try:
            if dtype == torch.float16:
                pipe = AutoPipelineForImage2Image.from_pretrained(
                    model, torch_dtype=dtype, variant="fp16"
                )
            else:
                raise ValueError("no fp16 variant wanted")
        except Exception:
            # Not every model ships fp16 weights; the default ones get converted on load.
            pipe = AutoPipelineForImage2Image.from_pretrained(model, torch_dtype=dtype)
        self.check()
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        self.pipe, self.loaded = pipe, (model, device)
        send(event="stage", text=f"Model loaded in {time.monotonic() - started:.0f} s")

    def generate(self, req):
        import torch
        from PIL import Image

        self.load(req["model"], req.get("device", "auto"))
        self.check()
        image = Image.open(io.BytesIO(base64.b64decode(req["image"]))).convert("RGB")
        image = image.resize(fit(image.size, int(req.get("size", 512))), Image.LANCZOS)
        strength = min(max(float(req.get("strength", 0.5)), 0.01), 1.0)
        # SDXL-Turbo needs steps * strength >= 1 to run at least one denoising step.
        steps = max(int(req.get("steps", 4)), 1, math.ceil(1 / strength))
        total = max(int(steps * strength), 1)
        seed = int(req.get("seed", -1))
        generator = torch.Generator(device="cpu").manual_seed(seed) if seed >= 0 else None

        def on_step(pipe, step, timestep, kwargs):
            self.check()
            send(event="progress", step=step + 1, steps=total)
            return kwargs

        send(event="stage", text="Generating…")
        result = self.pipe(
            prompt=req["prompt"],
            image=image,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=float(req.get("guidance", 0.0)),  # Turbo is trained without CFG
            generator=generator,
            callback_on_step_end=on_step,
        ).images[0]
        self.check()
        out = io.BytesIO()
        result.save(out, format="PNG")
        send(
            event="done",
            image=base64.b64encode(out.getvalue()).decode("ascii"),
            width=result.width,
            height=result.height,
        )


def read_commands(commands, worker):
    """Reads stdin on its own thread, so a cancel gets through while a run is busy."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            cmd = json.loads(line)
        except ValueError:
            send(event="error", message="unreadable command")
            continue
        if cmd.get("cmd") == "cancel":
            worker.cancel.set()
        else:
            commands.put(cmd)
    commands.put(None)


def main():
    worker = Worker()
    commands = queue.Queue()
    threading.Thread(target=read_commands, args=(commands, worker), daemon=True).start()
    send(event="ready", python=sys.version.split()[0])
    while True:
        cmd = commands.get()
        if cmd is None:
            return
        if cmd.get("cmd") != "generate":
            send(event="error", message=f"unknown command {cmd.get('cmd')!r}")
            continue
        worker.cancel.clear()
        try:
            worker.generate(cmd)
        except Cancelled:
            send(event="error", message="cancelled")
        except ImportError as e:
            # `missing` lets the app install the packages and try again.
            send(event="error", message=f"{e}. {INSTALL_HINT}", missing=True)
        except Exception as e:
            send(event="error", message=describe(e))


if __name__ == "__main__":
    main()
