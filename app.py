#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Desktop web interface for frequency-domain image watermark embedding."""

from __future__ import annotations

import base64
import io
import math
import threading
import uuid
import webbrowser
from pathlib import Path

import numpy as np
from flask import Flask, render_template, request, send_from_directory
from PIL import Image
from werkzeug.serving import BaseWSGIServer, make_server


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = Path.home() / ".frequency_watermark" / "web_outputs"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}
DEFAULT_ALPHA = 0.05
DCT_SIZE = 8
DCT_POSITIONS = ((3, 3), (3, 4), (4, 3), (4, 4))
METHODS = {
    "dft": "DFT 频域嵌入",
    "dct": "DCT QIM 分块嵌入",
    "dwt": "Haar DWT 小波嵌入",
}
SERVER: BaseWSGIServer | None = None

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


def dft_embed(cover: np.ndarray, wm_pm1: np.ndarray, alpha: float) -> np.ndarray:
    pos = dft_embed_position(cover.shape, wm_pm1.shape)
    channels = [dft_embed_channel(cover[:, :, index], wm_pm1, pos, alpha) for index in range(3)]
    return np.stack(channels, axis=2)


def luminance(image: np.ndarray) -> np.ndarray:
    return image[:, :, 0] * 0.299 + image[:, :, 1] * 0.587 + image[:, :, 2] * 0.114


def apply_luminance(image: np.ndarray, original: np.ndarray, modified: np.ndarray) -> np.ndarray:
    return np.clip(image + (modified - original)[:, :, np.newaxis], 0, 1)


def dct_matrix(size: int = DCT_SIZE) -> np.ndarray:
    positions = np.arange(size)
    matrix = np.cos(np.pi * (2 * positions[np.newaxis, :] + 1) * positions[:, np.newaxis] / (2 * size))
    matrix[0] *= 1 / np.sqrt(size)
    matrix[1:] *= np.sqrt(2 / size)
    return matrix


DCT_MATRIX = dct_matrix()


def dct_embed(cover: np.ndarray, wm_bin: np.ndarray, alpha: float) -> np.ndarray:
    source = luminance(cover)
    work = source * 255
    bits = (wm_bin.ravel() > 0).astype(np.uint8)
    capacity = (source.shape[0] // DCT_SIZE) * (source.shape[1] // DCT_SIZE) * len(DCT_POSITIONS)
    if bits.size > capacity:
        raise ValueError("载体图片过小，无法容纳当前 DCT 水印。")

    delta = max(4.0, alpha * 720)
    index = 0
    for y in range(0, source.shape[0] - DCT_SIZE + 1, DCT_SIZE):
        for x in range(0, source.shape[1] - DCT_SIZE + 1, DCT_SIZE):
            block = work[y:y + DCT_SIZE, x:x + DCT_SIZE]
            coeff = DCT_MATRIX @ block @ DCT_MATRIX.T
            for row, column in DCT_POSITIONS:
                if index >= bits.size:
                    break
                quantized = int(np.floor(coeff[row, column] / delta + 0.5))
                if quantized % 2 != int(bits[index]):
                    quantized += 1
                coeff[row, column] = quantized * delta
                index += 1
            work[y:y + DCT_SIZE, x:x + DCT_SIZE] = DCT_MATRIX.T @ coeff @ DCT_MATRIX
            if index >= bits.size:
                return apply_luminance(cover, source, np.clip(work / 255, 0, 1))
    return apply_luminance(cover, source, np.clip(work / 255, 0, 1))


def haar_dwt2(channel: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if channel.shape[0] % 2 or channel.shape[1] % 2:
        channel = np.pad(channel, ((0, channel.shape[0] % 2), (0, channel.shape[1] % 2)), mode="edge")
    a, b = channel[0::2, 0::2], channel[0::2, 1::2]
    c, d = channel[1::2, 0::2], channel[1::2, 1::2]
    return (a + b + c + d) / 2, ((a - b + c - d) / 2, (a + b - c - d) / 2, (a - b - c + d) / 2)


def haar_idwt2(
    ll: np.ndarray,
    detail: tuple[np.ndarray, np.ndarray, np.ndarray],
    shape: tuple[int, int],
) -> np.ndarray:
    lh, hl, hh = detail
    result = np.zeros((ll.shape[0] * 2, ll.shape[1] * 2))
    result[0::2, 0::2] = (ll + lh + hl + hh) / 2
    result[0::2, 1::2] = (ll - lh + hl - hh) / 2
    result[1::2, 0::2] = (ll + lh - hl - hh) / 2
    result[1::2, 1::2] = (ll - lh - hl + hh) / 2
    return result[:shape[0], :shape[1]]


def resize_pm1(wm_pm1: np.ndarray, width: int, height: int) -> np.ndarray:
    watermark = Image.fromarray(np.where(wm_pm1 > 0, 255, 0).astype(np.uint8), mode="L")
    resized = watermark.resize((width, height), Image.Resampling.NEAREST)
    return np.where(np.asarray(resized) > 0, 1.0, -1.0)


def dwt_embed(cover: np.ndarray, wm_pm1: np.ndarray, alpha: float) -> np.ndarray:
    source = luminance(cover)
    ll, (lh, hl, hh) = haar_dwt2(source)
    watermark = resize_pm1(wm_pm1, lh.shape[1], lh.shape[0])
    lh_watermarked = lh + alpha * (np.std(lh) + 1e-8) * watermark
    hl_watermarked = hl + alpha * (np.std(hl) + 1e-8) * watermark
    modified = np.clip(haar_idwt2(ll, (lh_watermarked, hl_watermarked, hh), source.shape), 0, 1)
    return apply_luminance(cover, source, modified)


def embed_watermark(method: str, cover: np.ndarray, wm_bin: np.ndarray, wm_pm1: np.ndarray, alpha: float) -> np.ndarray:
    if method == "dft":
        return dft_embed(cover, wm_pm1, alpha)
    if method == "dct":
        return dct_embed(cover, wm_bin, alpha)
    if method == "dwt":
        return dwt_embed(cover, wm_pm1, alpha)
    raise ValueError("未知的水印算法。")


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
    mse = float(np.mean((cover - watermarked) ** 2))
    psnr_value = math.inf if mse == 0 else 10 * math.log10(1 / mse)

    # Global SSIM is sufficient for the UI quality summary and keeps the desktop
    # package independent from the much larger scikit-image distribution.
    means_cover = np.mean(cover, axis=(0, 1))
    means_watermarked = np.mean(watermarked, axis=(0, 1))
    centered_cover = cover - means_cover
    centered_watermarked = watermarked - means_watermarked
    variance_cover = np.mean(centered_cover**2, axis=(0, 1))
    variance_watermarked = np.mean(centered_watermarked**2, axis=(0, 1))
    covariance = np.mean(centered_cover * centered_watermarked, axis=(0, 1))
    c1, c2 = 0.01**2, 0.03**2
    channel_ssim = (
        (2 * means_cover * means_watermarked + c1) * (2 * covariance + c2)
        / ((means_cover**2 + means_watermarked**2 + c1) * (variance_cover + variance_watermarked + c2))
    )
    ssim_value = float(np.mean(channel_ssim))
    return psnr_value, ssim_value


@app.route("/", methods=["GET", "POST"])
def index():
    context = {"alpha": DEFAULT_ALPHA, "method": "dft", "methods": METHODS}

    if request.method == "POST":
        cover_file = request.files.get("cover")
        watermark_file = request.files.get("watermark")
        alpha = float(request.form.get("alpha", DEFAULT_ALPHA))
        method = request.form.get("method", "dft")
        context.update(alpha=alpha, method=method)

        if not cover_file or not watermark_file or cover_file.filename == "" or watermark_file.filename == "":
            context["error"] = "请同时上传载体图片和水印图片。"
            return render_template("index.html", **context), 400
        if not allowed_file(cover_file.filename) or not allowed_file(watermark_file.filename):
            context["error"] = "仅支持 PNG、JPG、JPEG、BMP、WEBP 格式。"
            return render_template("index.html", **context), 400
        if method not in METHODS:
            context["error"] = "请选择有效的水印算法。"
            return render_template("index.html", **context), 400

        try:
            cover = read_cover(cover_file)
            watermark = read_watermark(watermark_file)
            wm_bin, wm_pm1 = prepare_watermark(watermark, cover.shape)
            watermarked = embed_watermark(method, cover, wm_bin, wm_pm1, alpha)
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
                method_name=METHODS[method],
            )
        except Exception as exc:
            context["error"] = f"处理失败：{exc}"
            return render_template("index.html", **context), 500

    return render_template("index.html", **context)


@app.route("/download/<filename>")
def download(filename: str):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)


@app.post("/shutdown")
def shutdown():
    if SERVER is not None:
        threading.Thread(target=SERVER.shutdown, daemon=True).start()
    return "应用已退出，可以关闭此页面。"


def open_browser(url: str) -> None:
    webbrowser.open(url, new=1)


def run_desktop_app() -> None:
    global SERVER
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SERVER = make_server("127.0.0.1", 0, app)
    url = f"http://127.0.0.1:{SERVER.server_port}"
    print(f"Open {url}", flush=True)
    threading.Timer(0.5, open_browser, args=(url,)).start()
    SERVER.serve_forever()


if __name__ == "__main__":
    run_desktop_app()
