"""Convert ComfyUI IMAGE tensors to llama.cpp OpenAI-compatible image URLs."""

import base64
from io import BytesIO


MAX_IMAGES = 4
MAX_INPUT_PIXELS = 16_777_216
MAX_SIDE = 1536
MAX_PNG_BYTES = 10 * 1024 * 1024


def image_content_parts(image):
    """Encode an IMAGE batch [B,H,W,C] into content-array image_url parts.

    The input is processed on the CPU before loading the external llama-server
    so ComfyUI can safely release any of its GPU-held models afterward.
    """
    if image is None:
        return []
    shape = getattr(image, "shape", None)
    if shape is None or len(shape) != 4 or not hasattr(image, "detach"):
        raise ValueError("LlamaForge: image must be a ComfyUI IMAGE tensor [batch, height, width, channels].")
    batch, height, width, channels = (int(x) for x in shape)
    if not 1 <= batch <= MAX_IMAGES:
        raise ValueError(f"LlamaForge: image batch must contain 1 to {MAX_IMAGES} frames.")
    if height <= 0 or width <= 0 or height * width > MAX_INPUT_PIXELS:
        raise ValueError("LlamaForge: image dimensions are invalid or exceed 16 megapixels per image.")
    if channels not in (1, 3, 4):
        raise ValueError("LlamaForge: IMAGE must have 1 (gray), 3 (RGB), or 4 (RGBA) channels.")

    try:
        import numpy as np
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("LlamaForge: image input requires Pillow and NumPy in ComfyUI's Python environment.") from error

    parts = []
    for index in range(batch):
        # ComfyUI tensors contain floating point pixels in [0,1] (including
        # GPU tensors and sometimes float16 or bfloat16).
        frame = image[index].detach().cpu().float()
        frame = frame.nan_to_num(nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        pixels = np.rint(frame.numpy() * 255.0).astype(np.uint8)
        if channels == 1:
            pixels = pixels[..., 0]
        if channels == 4:
            rgba = Image.fromarray(pixels)
            white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            picture = Image.alpha_composite(white, rgba).convert("RGB")
        else:
            picture = Image.fromarray(pixels).convert("RGB")
        picture.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)

        buffer = BytesIO()
        picture.save(buffer, format="PNG")
        encoded = buffer.getvalue()
        if len(encoded) > MAX_PNG_BYTES:
            raise ValueError(f"LlamaForge: image {index + 1} exceeds the 10 MB PNG upload limit.")
        uri = "data:image/png;base64," + base64.b64encode(encoded).decode("ascii")
        parts.append({"type": "image_url", "image_url": {"url": uri}})
    return parts


def chat_content(prompt, image=None):
    images = image_content_parts(image)
    if not images:
        return prompt
    return [{"type": "text", "text": prompt}, *images]
