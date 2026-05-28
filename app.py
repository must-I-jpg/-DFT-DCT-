#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web interface for DFT image watermark embedding.

Run:
    python app.py
Then open:
    http://127.0.0.1:5000
"""

from __future__ import annotations

import base64
import io
import uuid
from pathlib import Path

import numpy as np
from flask import Flask, render_template, request, send_from_directory
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "web_outputs"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}
DEFAULT_ALPHA = 0.05

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def read_cover(file_storage) -> np.ndarray:
    image = Image.open(file_storage.stream).convert("RGB")
    return np.asarray(image, dtype=np.float64) / 255.0


def read_watermark(file_storage) -> Image.Image:
    return Image.open(file_storage.stream).convert("L")


def prepare_watermark(watermark: Image.Image, cover_shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    height, width = cover_shape[:2]
    wm_size = min(max(32, min(height, width) // 8), 192, max(8, min(height, width) // 2 - 4))
    resized = watermark.resize((wm_size, wm_size), Image.Resampling.LANCZOS)
    wm_gray = np.asarray(resized, dtype=np.uint8)
    wm_bin = np.where(wm_gray > 127, 255, 0).astype(np.uint8)
    wm_pm1 = np.where(wm_bin > 0, 1.0, -1.0)
    return wm_bin, wm_pm1


def dft_embed_position(
    image_shape: tuple[int, int, int],
    wm_shape: tuple[int, int],
    offset_ratio: float = 0.08,
) -> tuple[int, int, int, int]:
    height, width = image_shape[:2]
    wm_height, wm_width = wm_shape
    center_y, center_x = height // 2, width // 2
    y = center_y - wm_height // 2
    x = center_x + max(1, int(min(height, width) * offset_ratio // 4))
    sy = 2 * center_y - y - wm_height
    sx = 2 * center_x - x - wm_width
    if (
        min(y, x, sy, sx) < 0
        or y + wm_height > height
        or x + wm_width > width
        or sy + wm_height > height
        or sx + wm_width > width
    ):
        raise ValueError("水印尺寸过大，DFT 嵌入区域越界。")
    return y, x, sy, sx


def dft_embed_channel(
    channel: np.ndarray,
    wm_pm1: np.ndarray,
    pos: tuple[int, int, int, int],
    alpha: float,
) -> np.ndarray:
    y, x, sy, sx = pos
    wm_height, wm_width = wm_pm1.shape
    region = (slice(y, y + wm_height), slice(x, x + wm_width))
    sym_region = (slice(sy, sy + wm_height), slice(sx, sx + wm_width))

    spectrum = np.fft.fftshift(np.fft.fft2(channel))
    floor = np.percentile(np.abs(spectrum), 75) * 0.02 + 1e-12

    coeff = spectrum[region]
    scale = np.maximum(np.abs(coeff), floor)
    phase = np.exp(1j * np.angle(coeff))
    spectrum[region] = coeff + alpha * scale * wm_pm1 * phase

    coeff_sym = spectrum[sym_region]
    scale_sym = np.maximum(np.abs(coeff_sym), floor)
    phase_sym = np.exp(1j * np.angle(coeff_sym))
    spectrum[sym_region] = coeff_sym + alpha * scale_sym * np.flipud(np.fliplr(wm_pm1)) * phase_sym

    embedded = np.real(np.fft.ifft2(np.fft.ifftshift(spectrum)))
    return np.clip(embedded, 0, 1)


def embed_watermark(cover: np.ndarray, wm_pm1: np.ndarray, alpha: float) -> np.ndarray:
    pos = dft_embed_position(cover.shape, wm_pm1.shape)
    channels = [dft_embed_channel(cover[:, :, index], wm_pm1, pos, alpha) for index in range(3)]
    return np.stack(channels, axis=2)


def to_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(image * 255), 0, 255).astype(np.uint8)


def image_data_uri(image: np.ndarray | Image.Image) -> str:
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(to_uint8(image), mode="RGB")
    else:
        pil_image = image
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    data = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{data}"


def quality_metrics(cover: np.ndarray, watermarked: np.ndarray) -> tuple[float, float]:
    cover_u8 = to_uint8(cover)
    watermarked_u8 = to_uint8(watermarked)
    psnr_value = float(peak_signal_noise_ratio(cover_u8, watermarked_u8, data_range=255))
    ssim_value = float(structural_similarity(cover_u8, watermarked_u8, channel_axis=2, data_range=255))
    return psnr_value, ssim_value


@app.route("/", methods=["GET", "POST"])
def index():
    context = {"alpha": DEFAULT_ALPHA}

    if request.method == "POST":
        cover_file = request.files.get("cover")
        watermark_file = request.files.get("watermark")
        alpha = float(request.form.get("alpha", DEFAULT_ALPHA))
        context["alpha"] = alpha

        if not cover_file or not watermark_file or cover_file.filename == "" or watermark_file.filename == "":
            context["error"] = "请同时上传载体图片和水印图片。"
            return render_template("index.html", **context), 400
        if not allowed_file(cover_file.filename) or not allowed_file(watermark_file.filename):
            context["error"] = "仅支持 PNG、JPG、JPEG、BMP、WEBP 格式。"
            return render_template("index.html", **context), 400

        try:
            cover = read_cover(cover_file)
            watermark = read_watermark(watermark_file)
            wm_bin, wm_pm1 = prepare_watermark(watermark, cover.shape)
            watermarked = embed_watermark(cover, wm_pm1, alpha)
            psnr_value, ssim_value = quality_metrics(cover, watermarked)

            output_name = f"watermarked_{uuid.uuid4().hex}.png"
            Image.fromarray(to_uint8(watermarked), mode="RGB").save(OUTPUT_DIR / output_name)

            context.update(
                result=True,
                cover_preview=image_data_uri(cover),
                watermark_preview=image_data_uri(Image.fromarray(wm_bin, mode="L")),
                result_preview=image_data_uri(watermarked),
                download_name=output_name,
                psnr=f"{psnr_value:.2f}",
                ssim=f"{ssim_value:.4f}",
                watermark_size=f"{wm_bin.shape[1]} x {wm_bin.shape[0]}",
            )
        except Exception as exc:
            context["error"] = f"处理失败：{exc}"
            return render_template("index.html", **context), 500

    return render_template("index.html", **context)


@app.route("/download/<filename>")
def download(filename: str):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    app.run(host="127.0.0.1", port=5000, debug=False)
