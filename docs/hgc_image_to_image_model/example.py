"""
Simple local image-to-image with SDXL-Turbo (Hugging Face diffusers).

Install once:
    pip install torch diffusers transformers accelerate pillow

Usage:
    python img2img.py input.jpg "a watercolor painting of a cat" -o out.png -s 0.5

The model (~7 GB) downloads automatically on the first run and is cached afterwards.
"""
import argparse

import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image


def main():
    p = argparse.ArgumentParser(description="Local image-to-image")
    p.add_argument("image", help="input image path")
    p.add_argument("prompt", help="what the output should look like")
    p.add_argument("-o", "--output", default="output.png")
    p.add_argument("-s", "--strength", type=float, default=0.5,
                   help="0.1 = barely change, 0.9 = almost new image")
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    # Pick the best available device
    if torch.cuda.is_available():
        device, dtype = "cuda", torch.float16
    elif torch.backends.mps.is_available():
        device, dtype = "mps", torch.float16
    else:
        device, dtype = "cpu", torch.float32
    print(f"Using {device}")

    pipe = AutoPipelineForImage2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
    ).to(device)

    init = load_image(args.image).convert("RGB").resize((512, 512))

    # SDXL-Turbo needs steps * strength >= 1
    steps = max(args.steps, int(1 / args.strength + 0.999))

    generator = None
    if args.seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(args.seed)

    result = pipe(
        prompt=args.prompt,
        image=init,
        strength=args.strength,
        num_inference_steps=steps,
        guidance_scale=0.0,  # Turbo is trained without CFG
        generator=generator,
    ).images[0]

    result.save(args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()


# pip install torch diffusers transformers accelerate pillow
# python img2img.py photo.jpg "oil painting of a mountain lake at sunset" -s 0.5 -o result.png