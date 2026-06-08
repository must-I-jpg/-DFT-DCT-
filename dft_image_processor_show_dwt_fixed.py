#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像频域数字水印实验

功能：
1. DFT 非盲水印嵌入与提取
2. alpha 嵌入强度扫描
3. JPEG、缩放、加噪鲁棒性测试
4. DCT QIM、DWT、DFT 盲 QIM 方法对比

运行前请准备：
- input.png      载体图像
- watermark.png  水印图像
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".matplotlib-cache"))

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib import font_manager
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


ALPHA = 0.05
DCT_DELTA = 36
DFT_QIM_DELTA = 50.0
DPI = 300
RNG = np.random.default_rng(2026)


def configure_chinese_font():
    for path in [
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
    ]:
        if os.path.exists(path):
            font_manager.fontManager.addfont(path)

    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ["STHeiti", "Hiragino Sans GB", "Songti SC", "PingFang SC", "SimHei"]:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


configure_chinese_font()


def load_inputs():
    cover_path = ROOT / "input.png"
    wm_path = ROOT / "watermark.png"
    cover = cv2.imread(str(cover_path), cv2.IMREAD_COLOR)
    watermark = cv2.imread(str(wm_path), cv2.IMREAD_GRAYSCALE)
    if cover is None:
        raise FileNotFoundError(f"请将载体图像命名为 input.png 并放在脚本目录：{cover_path}")
    if watermark is None:
        raise FileNotFoundError(f"请将水印图像命名为 watermark.png 并放在脚本目录：{wm_path}")
    return cover.astype(np.float64) / 255.0, watermark


def prepare_watermark(watermark, cover_shape):
    h, w = cover_shape[:2]
    wm_size = min(max(32, min(h, w) // 8), 192, max(8, min(h, w) // 2 - 4))
    wm = cv2.resize(watermark, (wm_size, wm_size), interpolation=cv2.INTER_AREA)
    _, wm_bin = cv2.threshold(wm, 127, 255, cv2.THRESH_BINARY)
    wm_pm1 = np.where(wm_bin > 0, 1.0, -1.0)
    return wm_bin.astype(np.uint8), wm_pm1


def dft_embed_position(image_shape, wm_shape, offset_ratio=0.08):
    h, w = image_shape[:2]
    wh, ww = wm_shape
    cy, cx = h // 2, w // 2
    y = cy - wh // 2
    x = cx + max(1, int(min(h, w) * offset_ratio // 4))
    sy = 2 * cy - y - wh
    sx = 2 * cx - x - ww
    if min(y, x, sy, sx) < 0 or y + wh > h or x + ww > w or sy + wh > h or sx + ww > w:
        raise ValueError("水印尺寸过大，DFT 嵌入区域越界。")
    return y, x, sy, sx


def normalized_correlation(wm_a, wm_b):
    if wm_a.shape != wm_b.shape:
        wm_b = cv2.resize(wm_b, (wm_a.shape[1], wm_a.shape[0]), interpolation=cv2.INTER_NEAREST)
    a = np.where(wm_a > 0, 1.0, -1.0).ravel()
    b = np.where(wm_b > 0, 1.0, -1.0).ravel()
    return float(np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def quality_metrics(cover, watermarked):
    a = (cover * 255).astype(np.uint8)
    b = (watermarked * 255).astype(np.uint8)
    return float(psnr(a, b)), float(ssim(a, b, channel_axis=2))


def dft_embed_channel(channel, wm_pm1, pos, alpha):
    y, x, sy, sx = pos
    wh, ww = wm_pm1.shape
    region = (slice(y, y + wh), slice(x, x + ww))
    sym_region = (slice(sy, sy + wh), slice(sx, sx + ww))

    f = np.fft.fftshift(np.fft.fft2(channel))
    before = f.copy()
    floor = np.percentile(np.abs(f), 75) * 0.02 + 1e-12

    coeff = f[region]
    scale = np.maximum(np.abs(coeff), floor)
    phase = np.exp(1j * np.angle(coeff))
    f[region] = coeff + alpha * scale * wm_pm1 * phase

    coeff_sym = f[sym_region]
    scale_sym = np.maximum(np.abs(coeff_sym), floor)
    phase_sym = np.exp(1j * np.angle(coeff_sym))
    f[sym_region] = coeff_sym + alpha * scale_sym * np.flipud(np.fliplr(wm_pm1)) * phase_sym

    out = np.real(np.fft.ifft2(np.fft.ifftshift(f)))
    return np.clip(out, 0, 1), before, f


def dft_extract_channel(original, watermarked, wm_shape, pos, alpha):
    y, x, sy, sx = pos
    wh, ww = wm_shape
    region = (slice(y, y + wh), slice(x, x + ww))
    sym_region = (slice(sy, sy + wh), slice(sx, sx + ww))

    f0 = np.fft.fftshift(np.fft.fft2(original))
    f1 = np.fft.fftshift(np.fft.fft2(watermarked))
    floor = np.percentile(np.abs(f0), 75) * 0.02 + 1e-12

    coeff = f0[region]
    scale = np.maximum(np.abs(coeff), floor)
    phase = np.exp(1j * np.angle(coeff))
    ext = np.real((f1[region] - coeff) * np.conj(phase)) / (alpha * scale + 1e-12)

    coeff_sym = f0[sym_region]
    scale_sym = np.maximum(np.abs(coeff_sym), floor)
    phase_sym = np.exp(1j * np.angle(coeff_sym))
    ext_sym = np.real((f1[sym_region] - coeff_sym) * np.conj(phase_sym)) / (alpha * scale_sym + 1e-12)

    return np.where((ext + np.flipud(np.fliplr(ext_sym))) / 2 > 0, 255, 0).astype(np.uint8)


def dft_embed(cover, wm_pm1, pos, alpha=ALPHA):
    channels, f_before, f_after = [], [], []
    for c in range(3):
        ch, fb, fa = dft_embed_channel(cover[:, :, c], wm_pm1, pos, alpha)
        channels.append(ch)
        f_before.append(fb)
        f_after.append(fa)
    return np.stack(channels, axis=2), f_before, f_after


def dft_extract(cover, watermarked, wm_shape, pos, alpha=ALPHA):
    extracted = [
        dft_extract_channel(cover[:, :, c], watermarked[:, :, c], wm_shape, pos, alpha)
        for c in range(3)
    ]
    return np.where(np.sum(extracted, axis=0) > 255 * 1.5, 255, 0).astype(np.uint8)


def save_basic_result(cover, watermarked, wm_bin, wm_ext, metrics):
    p, s, n = metrics
    cover_u8 = (cover * 255).astype(np.uint8)
    wm_u8 = (watermarked * 255).astype(np.uint8)
    diff = cv2.normalize(cv2.absdiff(cover_u8, wm_u8), None, 0, 255, cv2.NORM_MINMAX)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    items = [
        (cover_u8, "原始载体", None),
        (wm_u8, f"含水印图像\nPSNR={p:.2f}dB, SSIM={s:.4f}", None),
        (diff, "差异放大", None),
        (wm_bin, "原始水印", "gray"),
        (wm_ext, f"提取水印\nNC={n:.4f}", "gray"),
    ]
    for ax, (img, title, cmap) in zip(axes.ravel(), items):
        if cmap == "gray":
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis("off")

    axes[1, 2].axis("off")
    axes[1, 2].text(0.1, 0.5, f"DFT 非盲水印\nPSNR: {p:.2f} dB\nSSIM: {s:.4f}\nNC: {n:.4f}\nalpha: {ALPHA}",
                    fontsize=13, va="center")
    plt.tight_layout()
    plt.savefig(ROOT / "basic_dft_result.png", dpi=DPI, bbox_inches="tight")
    plt.close()


def save_frequency_diff(f_before, f_after):
    diff = np.abs(np.log1p(np.abs(f_after)) - np.log1p(np.abs(f_before)))
    plt.figure(figsize=(10, 8))
    plt.imshow(diff, cmap="hot")
    plt.colorbar(label="频谱差异（对数）")
    plt.title("DFT 频谱嵌入前后差异")
    plt.savefig(ROOT / "frequency_diff.png", dpi=DPI, bbox_inches="tight")
    plt.close()


def add_embed_boxes(ax, pos, wm_shape):
    y, x, sy, sx = pos
    wh, ww = wm_shape
    for label, px, py in [("main", x, y), ("sym", sx, sy)]:
        ax.add_patch(Rectangle((px, py), ww, wh, fill=False, edgecolor="cyan", linewidth=1.5))
        ax.text(px, py - 3, label, color="cyan", fontsize=7, weight="bold", va="bottom")


def save_frequency_comparison(f_before, f_after, pos, wm_shape):
    channel_names = ["B", "G", "R"]
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    fig.suptitle("DFT 水印嵌入前后频域图像对比", fontsize=16)

    for row, name in enumerate(channel_names):
        before = np.log1p(np.abs(f_before[row]))
        after = np.log1p(np.abs(f_after[row]))
        diff = np.abs(after - before)
        spectrum_vmax = max(float(before.max()), float(after.max()))
        diff_vmax = max(float(diff.max()), 1e-12)

        items = [
            (before, f"{name} 通道：DFT 嵌入前幅度谱", "gray", 0.0, spectrum_vmax),
            (after, f"{name} 通道：DFT 嵌入后幅度谱", "gray", 0.0, spectrum_vmax),
            (diff, f"{name} 通道：频域变化差异", "hot", 0.0, diff_vmax),
        ]

        for col, (img, title, cmap, vmin, vmax) in enumerate(items):
            ax = axes[row, col]
            im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
            add_embed_boxes(ax, pos, wm_shape)
            ax.set_title(title)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(ROOT / "frequency_comparison.png", dpi=DPI, bbox_inches="tight")
    plt.close()


def alpha_scan(cover, wm_bin, wm_pm1, pos):
    alphas = np.linspace(0.01, 0.3, 15)
    psnr_vals, ssim_vals, nc_vals = [], [], []
    for a in alphas:
        wm_img, _, _ = dft_embed(cover, wm_pm1, pos, a)
        p, s = quality_metrics(cover, wm_img)
        ext = dft_extract(cover, wm_img, wm_bin.shape, pos, a)
        psnr_vals.append(p)
        ssim_vals.append(s)
        nc_vals.append(normalized_correlation(wm_bin, ext))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, vals, title, ylabel, color in [
        (axes[0], psnr_vals, "PSNR vs alpha", "PSNR (dB)", "tab:blue"),
        (axes[1], ssim_vals, "SSIM vs alpha", "SSIM", "tab:red"),
        (axes[2], nc_vals, "NC vs alpha", "NC", "tab:green"),
    ]:
        ax.plot(alphas, vals, "o-", color=color, markersize=4)
        ax.set_title(title)
        ax.set_xlabel("alpha")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(ROOT / "alpha_scan.png", dpi=DPI, bbox_inches="tight")
    plt.close()


def jpeg_attack(img, quality):
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def scale_attack(img, scale):
    h, w = img.shape[:2]
    small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def noise_attack(img, sigma):
    noisy = img.astype(np.float64) + RNG.normal(0, sigma, img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def robustness_tests(cover, watermarked, wm_bin, pos):
    cover_u8 = (cover * 255).astype(np.uint8)
    wm_u8 = (watermarked * 255).astype(np.uint8)
    wm_shape = wm_bin.shape

    tests = {
        "JPEG 压缩": [(q, jpeg_attack(wm_u8, q)) for q in [90, 70, 50, 30, 20, 10]],
        "缩放": [(s, scale_attack(wm_u8, s)) for s in [0.5, 0.75, 1.25, 1.5, 2.0]],
        "高斯噪声": [(n, noise_attack(wm_u8, n)) for n in [5, 10, 15, 20, 30, 50]],
    }

    results = {}
    for name, attacked_list in tests.items():
        rows = []
        for strength, attacked in attacked_list:
            attacked_float = attacked.astype(np.float64) / 255.0
            ext = dft_extract(cover, attacked_float, wm_shape, pos, ALPHA)
            rows.append((strength, normalized_correlation(wm_bin, ext), psnr(cover_u8, attacked)))
        results[name] = rows
    return results


def save_robustness_summary(results):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ["steelblue", "coral", "seagreen"]
    xlabels = {
        "JPEG 压缩": "JPEG 质量因子 Q（越低压缩越强）",
        "缩放": "缩放比例 s（先缩放到 s 倍，再恢复原尺寸）",
        "高斯噪声": "噪声标准差 sigma（像素值 0-255）",
    }
    for ax, (name, rows), color in zip(axes, results.items(), colors):
        labels = [str(r[0]) for r in rows]
        ncs = [r[1] for r in rows]
        ax.bar(range(len(labels)), ncs, tick_label=labels, color=color)
        ax.set_title(name + "鲁棒性")
        ax.set_xlabel(xlabels[name])
        ax.set_ylabel("NC")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(ROOT / "robustness_summary.png", dpi=DPI, bbox_inches="tight")
    plt.close()


def dct_embed(channel, wm_bin, delta=DCT_DELTA):
    h, w = channel.shape
    work = channel * 255.0
    bits = (wm_bin.ravel() > 0).astype(np.uint8)
    idx = 0
    positions = [(3, 3), (3, 4), (4, 3), (4, 4)]
    if bits.size > (h // 8) * (w // 8) * len(positions):
        raise ValueError("DCT 容量不足，水印太大。")

    for y in range(0, h - 7, 8):
        for x in range(0, w - 7, 8):
            block = cv2.dct(work[y:y + 8, x:x + 8])
            for r, c in positions:
                if idx >= bits.size:
                    break
                q = int(np.floor(block[r, c] / delta + 0.5))
                if q % 2 != int(bits[idx]):
                    q += 1
                block[r, c] = q * delta
                idx += 1
            work[y:y + 8, x:x + 8] = cv2.idct(block)
            if idx >= bits.size:
                return np.clip(work / 255.0, 0, 1)
    return np.clip(work / 255.0, 0, 1)


def dct_extract(channel, wm_shape, delta=DCT_DELTA):
    h, w = channel.shape
    total = wm_shape[0] * wm_shape[1]
    work = channel * 255.0
    bits = []
    positions = [(3, 3), (3, 4), (4, 3), (4, 4)]
    for y in range(0, h - 7, 8):
        for x in range(0, w - 7, 8):
            block = cv2.dct(work[y:y + 8, x:x + 8])
            for r, c in positions:
                bits.append(int(np.floor(block[r, c] / delta + 0.5)) % 2)
                if len(bits) == total:
                    return (np.array(bits, dtype=np.uint8).reshape(wm_shape) * 255)
    raise ValueError("DCT 提取容量不足。")


def dct_experiment(cover, wm_bin):
    cover_u8 = (cover * 255).astype(np.uint8)
    ycrcb = cv2.cvtColor(cover_u8, cv2.COLOR_BGR2YCrCb)
    y = ycrcb[:, :, 0].astype(np.float64) / 255.0
    y_wm = dct_embed(y, wm_bin)
    ext = dct_extract(y_wm, wm_bin.shape)
    ycrcb[:, :, 0] = np.clip(y_wm * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR), ext


def haar_dwt2(x):
    if x.shape[0] % 2 or x.shape[1] % 2:
        x = np.pad(x, ((0, x.shape[0] % 2), (0, x.shape[1] % 2)), mode="edge")
    a, b = x[0::2, 0::2], x[0::2, 1::2]
    c, d = x[1::2, 0::2], x[1::2, 1::2]
    return (a + b + c + d) / 2, ((a - b + c - d) / 2, (a + b - c - d) / 2, (a - b - c + d) / 2)


def haar_idwt2(ll, detail, shape):
    lh, hl, hh = detail
    out = np.zeros((ll.shape[0] * 2, ll.shape[1] * 2))
    out[0::2, 0::2] = (ll + lh + hl + hh) / 2
    out[0::2, 1::2] = (ll - lh + hl - hh) / 2
    out[1::2, 0::2] = (ll + lh - hl - hh) / 2
    out[1::2, 1::2] = (ll - lh - hl + hh) / 2
    return out[:shape[0], :shape[1]]


def dwt_experiment(cover, wm_bin, wm_pm1, alpha=0.08):
    cover_u8 = (cover * 255).astype(np.uint8)
    ycrcb = cv2.cvtColor(cover_u8, cv2.COLOR_BGR2YCrCb)
    y = ycrcb[:, :, 0].astype(np.float64) / 255.0
    ll, (lh, hl, hh) = haar_dwt2(y)
    wm = cv2.resize(wm_pm1, (lh.shape[1], lh.shape[0]), interpolation=cv2.INTER_NEAREST)
    lh_wm = lh + alpha * (np.std(lh) + 1e-8) * wm
    hl_wm = hl + alpha * (np.std(hl) + 1e-8) * wm
    y_wm = np.clip(haar_idwt2(ll, (lh_wm, hl_wm, hh), y.shape), 0, 1)

    _, (lh2, hl2, _) = haar_dwt2(y_wm)
    ext = ((lh2 - lh) / (alpha * (np.std(lh) + 1e-8)) + (hl2 - hl) / (alpha * (np.std(hl) + 1e-8))) / 2
    ext = cv2.resize(ext, (wm_bin.shape[1], wm_bin.shape[0]), interpolation=cv2.INTER_LINEAR)
    ext_bin = np.where(ext > np.mean(ext), 255, 0).astype(np.uint8)

    ycrcb[:, :, 0] = np.clip(y_wm * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR), ext_bin


def blind_dft_embed(channel, wm_bin, pos, delta=DFT_QIM_DELTA):
    y, x, sy, sx = pos
    wh, ww = wm_bin.shape
    f = np.fft.fftshift(np.fft.fft2(channel))
    for oy, ox, bits in [(y, x, wm_bin > 0), (sy, sx, np.flipud(np.fliplr(wm_bin > 0)))]:
        for i in range(wh):
            for j in range(ww):
                coeff = f[oy + i, ox + j]
                q = int(np.floor(abs(coeff) / delta + 0.5))
                if q % 2 != int(bits[i, j]):
                    q += 1
                f[oy + i, ox + j] = q * delta * np.exp(1j * np.angle(coeff))
    return np.clip(np.real(np.fft.ifft2(np.fft.ifftshift(f))), 0, 1)


def blind_dft_extract(channel, wm_shape, pos, delta=DFT_QIM_DELTA):
    y, x, sy, sx = pos
    wh, ww = wm_shape
    f = np.fft.fftshift(np.fft.fft2(channel))
    main = np.zeros(wm_shape)
    sym = np.zeros(wm_shape)
    for i in range(wh):
        for j in range(ww):
            main[i, j] = int(np.floor(abs(f[y + i, x + j]) / delta + 0.5)) % 2
            sym[i, j] = int(np.floor(abs(f[sy + i, sx + j]) / delta + 0.5)) % 2
    return np.where((main + np.flipud(np.fliplr(sym))) / 2 > 0.5, 255, 0).astype(np.uint8)


def blind_dft_experiment(cover, wm_bin, pos):
    cover_u8 = (cover * 255).astype(np.uint8)
    gray = cv2.cvtColor(cover_u8, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    gray_wm = blind_dft_embed(gray, wm_bin, pos)
    ext = blind_dft_extract(gray_wm, wm_bin.shape, pos)
    return cv2.cvtColor((gray_wm * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR), ext


def save_method_group_figure(filename, title, selected_names, cover_u8, wm_bin, method_images, method_ext, metrics):
    meta = {
        "DFT 非盲": ("DFT", "加性嵌入", "非盲"),
        "DCT QIM": ("DCT", "QIM 量化", "盲"),
        "DWT": ("DWT", "加性嵌入", "非盲"),
        "DFT 盲 QIM": ("DFT", "QIM 量化", "盲"),
    }
    cols = len(selected_names) + 1
    fig = plt.figure(figsize=(4.8 * cols, 11))

    ax = fig.add_subplot(3, cols, 1)
    ax.imshow(cv2.cvtColor(cover_u8, cv2.COLOR_BGR2RGB))
    ax.set_title("原始载体")
    ax.axis("off")

    for i, name in enumerate(selected_names, start=2):
        p, _, _ = metrics[name]
        ax = fig.add_subplot(3, cols, i)
        ax.imshow(cv2.cvtColor(method_images[name], cv2.COLOR_BGR2RGB))
        ax.set_title(f"{name}\nPSNR={p:.1f}")
        ax.axis("off")

    ax = fig.add_subplot(3, cols, cols + 1)
    ax.imshow(wm_bin, cmap="gray", vmin=0, vmax=255)
    ax.set_title("原始水印")
    ax.axis("off")

    for i, name in enumerate(selected_names, start=cols + 2):
        _, _, n = metrics[name]
        ax = fig.add_subplot(3, cols, i)
        ax.imshow(method_ext[name], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{name}\nNC={n:.3f}")
        ax.axis("off")

    table_ax = fig.add_subplot(3, cols, (2 * cols + 1, 3 * cols))
    table_ax.axis("off")
    rows = []
    for name in selected_names:
        p, s, n = metrics[name]
        domain, embedding, extraction = meta[name]
        rows.append([name, domain, embedding, extraction, f"{p:.2f}", f"{s:.4f}", f"{n:.4f}"])
    table = table_ax.table(
        cellText=rows,
        colLabels=["方法", "变换域", "嵌入方式", "提取方式", "PSNR(dB)", "SSIM", "NC"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    table_ax.set_title(title)

    plt.tight_layout()
    plt.savefig(ROOT / filename, dpi=DPI, bbox_inches="tight")
    plt.close()


def compare_methods(cover, wm_bin, wm_pm1, pos, dft_wm, dft_ext, dft_metrics):
    cover_u8 = (cover * 255).astype(np.uint8)
    gray_cover = cv2.cvtColor(cover_u8, cv2.COLOR_BGR2GRAY)

    method_images = {"DFT 非盲": (dft_wm * 255).astype(np.uint8)}
    method_ext = {"DFT 非盲": dft_ext}
    metrics = {"DFT 非盲": dft_metrics}

    for name, runner in [
        ("DCT QIM", lambda: dct_experiment(cover, wm_bin)),
        ("DWT", lambda: dwt_experiment(cover, wm_bin, wm_pm1)),
        ("DFT 盲 QIM", lambda: blind_dft_experiment(cover, wm_bin, pos)),
    ]:
        img, ext = runner()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        method_images[name] = img
        method_ext[name] = ext
        metrics[name] = (float(psnr(gray_cover, gray)), float(ssim(gray_cover, gray)), normalized_correlation(wm_bin, ext))

    save_method_group_figure(
        "qim_comparison.png",
        "QIM 量化嵌入方法对比",
        ["DCT QIM", "DFT 盲 QIM"],
        cover_u8,
        wm_bin,
        method_images,
        method_ext,
        metrics,
    )
    save_method_group_figure(
        "dct_dwt_comparison.png",
        "DCT 与 DWT 变换域方法对比",
        ["DCT QIM", "DWT"],
        cover_u8,
        wm_bin,
        method_images,
        method_ext,
        metrics,
    )
    return metrics


def print_report(metrics, wm_shape):
    print("\n" + "=" * 58)
    print("图像频域水印实验报告")
    print("=" * 58)
    print(f"水印尺寸: {wm_shape[0]} x {wm_shape[1]}")
    print(f"DFT alpha: {ALPHA}")
    print(f"DCT QIM delta: {DCT_DELTA}")
    print(f"DFT 盲 QIM delta: {DFT_QIM_DELTA}")
    print("-" * 58)
    print(f"{'方法':<14} {'PSNR(dB)':<12} {'SSIM':<12} {'NC':<12}")
    for name, (p, s, n) in metrics.items():
        print(f"{name:<14} {p:<12.2f} {s:<12.4f} {n:<12.4f}")
    print("=" * 58)


def main():
    print("图像频域水印实验")
    cover, watermark = load_inputs()
    wm_bin, wm_pm1 = prepare_watermark(watermark, cover.shape)
    pos = dft_embed_position(cover.shape, wm_bin.shape)

    print(f"载体尺寸: {cover.shape[:2]}")
    print(f"水印尺寸: {wm_bin.shape}")
    print(f"DFT 嵌入区域: 主区域({pos[0]}, {pos[1]}), 对称区域({pos[2]}, {pos[3]})")

    watermarked_dft, f_before, f_after = dft_embed(cover, wm_pm1, pos)
    extracted_dft = dft_extract(cover, watermarked_dft, wm_bin.shape, pos)
    dft_metrics = (*quality_metrics(cover, watermarked_dft), normalized_correlation(wm_bin, extracted_dft))
    print(f"DFT 非盲: PSNR={dft_metrics[0]:.2f}dB, SSIM={dft_metrics[1]:.4f}, NC={dft_metrics[2]:.4f}")

    save_basic_result(cover, watermarked_dft, wm_bin, extracted_dft, dft_metrics)
    save_frequency_diff(f_before[0], f_after[0])
    save_frequency_comparison(f_before, f_after, pos, wm_bin.shape)
    alpha_scan(cover, wm_bin, wm_pm1, pos)
    robustness = robustness_tests(cover, watermarked_dft, wm_bin, pos)
    save_robustness_summary(robustness)
    metrics = compare_methods(cover, wm_bin, wm_pm1, pos, watermarked_dft, extracted_dft, dft_metrics)
    print_report(metrics, wm_bin.shape)

    print("\n输出文件：")
    for name in [
        "basic_dft_result.png",
        "frequency_diff.png",
        "frequency_comparison.png",
        "alpha_scan.png",
        "robustness_summary.png",
        "qim_comparison.png",
        "dct_dwt_comparison.png",
    ]:
        print(" -", ROOT / name)


if __name__ == "__main__":
    main()
