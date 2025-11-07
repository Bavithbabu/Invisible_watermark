import io
import os
from typing import List

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from watermark_embed import embed_watermark_bgr, save_image
from watermark_detect import detect_watermark_bgr
from metrics import compute_psnr, compute_mse, compute_ssim


st.set_page_config(page_title="Invisible Watermark (DCT)", layout="wide")

st.title("Invisible Watermark System (DCT)")

with st.sidebar:
	st.header("Settings")
	alpha = st.slider("Watermark Strength (alpha)", min_value=2.0, max_value=25.0, value=12.0, step=1.0)
	seed = st.number_input("Random Seed", value=12345, step=1)
	mode = st.selectbox("Watermark mode", options=["robust", "fragile"], index=0)
	embed_integrity = st.checkbox("Embed integrity fingerprint (detect tamper)", value=False)
	output_quality = st.slider("JPEG Quality", min_value=50, max_value=100, value=95, step=5)
	max_dim = st.selectbox("Max image size (auto downscale)", options=[1024, 1536, 2048, 3072, 4096], index=2)
	detect_threshold = st.slider("Detection threshold (tamper)", min_value=0.4, max_value=0.95, value=0.6, step=0.05)

st.subheader("Embed Watermark")
cols = st.columns(2)
with cols[0]:
	uploaded_files = st.file_uploader("Upload one or more images (JPG/PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
	watermark_text = st.text_input("Watermark message")
	embed_btn = st.button("Embed")

with cols[1]:
	st.write("Output directory: `watermarked/`")

if embed_btn:
	if not uploaded_files:
		st.warning("Please upload at least one image")
	elif not watermark_text:
		st.warning("Please enter watermark text")
	else:
		os.makedirs("watermarked", exist_ok=True)
		progress = st.progress(0.0)
		status = st.empty()
		total = len(uploaded_files)
		for idx, f in enumerate(uploaded_files, start=1):
			try:
				status.write(f"Processing {f.name} ({idx}/{total})…")
				file_bytes = np.frombuffer(f.read(), np.uint8)
				img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
				if img_bgr is None:
					st.error(f"Failed to read {f.name}")
					progress.progress(min(1.0, idx / total))
					continue
				# Optional downscale to speed up heavy images
				h, w = img_bgr.shape[:2]
				if max(h, w) > max_dim:
					scale = max_dim / float(max(h, w))
					img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
				wm_bgr, info = embed_watermark_bgr(img_bgr, watermark_text, alpha=alpha, seed=seed, mode=mode)
				wm_bgr, info = embed_watermark_bgr(img_bgr, watermark_text, alpha=alpha, seed=seed, mode=mode, embed_integrity=embed_integrity)
				psnr = compute_psnr(img_bgr, wm_bgr)
				# Display side-by-side
				c1, c2 = st.columns(2)
				with c1:
					st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption=f"Original: {f.name}", use_container_width=True)
				with c2:
					st.image(cv2.cvtColor(wm_bgr, cv2.COLOR_BGR2RGB), caption=f"Watermarked (PSNR {psnr:.2f} dB, mode {mode})", use_container_width=True)
				# Save
				out_path = os.path.join("watermarked", os.path.splitext(os.path.basename(f.name))[0] + "_wm.jpg")
				save_image(out_path, wm_bgr, quality=output_quality)
				st.success(f"Saved: {out_path}")
			except Exception as e:
				st.error(f"Error embedding {f.name}: {e}")
			finally:
				progress.progress(min(1.0, idx / total))

st.markdown("---")

st.subheader("Detect Watermark")
cols2 = st.columns(2)
with cols2[0]:
	file_to_check = st.file_uploader("Upload watermarked image", type=["jpg", "jpeg", "png"], key="detect_upload")
	orig_text = st.text_input("Original watermark text", key="detect_text")
	detect_btn = st.button("Detect")

# Helper for severity label

def _tamper_severity(score: float, threshold: float) -> str:
	if score >= threshold:
		return "Authentic"
	if score >= threshold - 0.1:
		return "Mild tamper"
	if score >= threshold - 0.3:
		return "Moderate tamper"
	return "Severe tamper"

if detect_btn:
	if not file_to_check:
		st.warning("Please upload a watermarked image")
	elif not orig_text:
		st.warning("Please enter the original watermark text")
	else:
		try:
			img_bgr = cv2.imdecode(np.frombuffer(file_to_check.read(), np.uint8), cv2.IMREAD_COLOR)
			if img_bgr is None:
				st.error("Failed to read the uploaded image")
			else:
				score, info = detect_watermark_bgr(img_bgr, orig_text, seed=seed, mode=mode, check_integrity=embed_integrity)
				st.write(f"Similarity: {score*100:.2f}%  |  Threshold: {detect_threshold*100:.0f}%  |  Mode: {mode}")
				verdict = "✅ AUTHENTIC" if score >= detect_threshold else "⚠️ TAMPERED/NOT PRESENT"
				st.write("Verdict:", verdict, f"| Tamper severity: {_tamper_severity(score, detect_threshold)}")
				if info.get("integrity_ok") is not None:
					st.write("Integrity check:", "OK ✅" if info.get("integrity_ok") else "FAILED ⚠️")
				st.progress(min(1.0, max(0.0, score)))
		except Exception as e:
			st.error(f"Error during detection: {e}")

st.markdown("---")

st.subheader("Tamper / Attack Playground")
colA, colB = st.columns(2)
with colA:
	atk_file = st.file_uploader("Upload a watermarked image to attack", type=["jpg", "jpeg", "png"], key="atk_upload")
	atk_text = st.text_input("Original watermark text (for detection)", key="atk_text")
	atk_type = st.selectbox("Attack type", ["JPEG", "Resize", "Rotate", "Noise", "Crop", "Blur", "Translate", "Desync Crop", "Chain (Very strong)"])
	# Custom slider(s)
	if atk_type == "JPEG":
		param = st.slider("Quality", 30, 100, 70, 5)
	elif atk_type == "Resize":
		param = st.slider("Scale (%)", 20, 100, 75, 5)
	elif atk_type == "Rotate":
		param = st.slider("Angle (deg)", -20, 20, 5, 1)
	elif atk_type == "Noise":
		param = st.slider("Noise sigma", 0.0, 25.0, 5.0, 0.5)
	elif atk_type == "Crop":
		param = st.slider("Center crop (%)", 50, 95, 70, 5)
	elif atk_type == "Blur":
		param = st.slider("Kernel size (odd)", 3, 21, 9, 2)
	elif atk_type == "Translate":
		param = st.slider("Shift pixels (±)", 1, 16, 4, 1)
	elif atk_type == "Desync Crop":
		param = st.slider("Crop (%) then offset paste", 60, 95, 80, 5)
	else:
		param = 0
	# Quick presets
	preset = st.radio("Quick strength presets", ["Custom", "Mild", "Strong", "Very strong"], index=0)
	gen_btn = st.button("Generate attacked and detect")

with colB:
	atk_placeholder = st.empty()
	res_placeholder = st.empty()
	download_placeholder = st.empty()
	save_toggle = st.checkbox("Also save attacked image to results/", value=False)
	filename_input = st.text_input("Filename (without path)", value="attacked_demo.jpg")

# Attack helpers

def _atk_jpeg(img: np.ndarray, q: int) -> np.ndarray:
	encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(q)]
	_, enc = cv2.imencode('.jpg', img, encode_param)
	return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def _atk_resize(img: np.ndarray, scale_pct: int) -> np.ndarray:
	scale = max(0.1, scale_pct / 100.0)
	h, w = img.shape[:2]
	small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
	return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def _atk_rotate(img: np.ndarray, angle: int) -> np.ndarray:
	h, w = img.shape[:2]
	M = cv2.getRotationMatrix2D((w // 2, h // 2), float(angle), 1.0)
	return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


def _atk_noise(img: np.ndarray, sigma: float) -> np.ndarray:
	noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
	noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
	return noisy


def _atk_crop(img: np.ndarray, crop_pct: int) -> np.ndarray:
	# center crop then resize back
	h, w = img.shape[:2]
	pct = np.clip(crop_pct, 50, 95) / 100.0
	ch, cw = int(h * pct), int(w * pct)
	y1 = (h - ch) // 2
	x1 = (w - cw) // 2
	cropped = img[y1:y1 + ch, x1:x1 + cw]
	return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_CUBIC)


def _atk_blur(img: np.ndarray, k: int) -> np.ndarray:
	k = max(3, int(k) | 1)
	return cv2.GaussianBlur(img, (k, k), 0)


def _atk_translate(img: np.ndarray, shift: int) -> np.ndarray:
	# Circular shift to misalign 8x8 blocks
	shift = int(shift)
	return np.roll(np.roll(img, shift, axis=0), shift, axis=1)


def _atk_desync_crop(img: np.ndarray, crop_pct: int) -> np.ndarray:
	# Crop and paste back with an offset to break block grid
	h, w = img.shape[:2]
	pct = np.clip(crop_pct, 60, 95) / 100.0
	ch, cw = int(h * pct), int(w * pct)
	y1 = (h - ch) // 2
	x1 = (w - cw) // 2
	cropped = img[y1:y1 + ch, x1:x1 + cw]
	canvas = np.zeros_like(img)
	# offset by 3 pixels each direction
	off = 3
	yst = min(max(0, y1 + off), h - ch)
	xst = min(max(0, x1 + off), w - cw)
	canvas[yst:yst + ch, xst:xst + cw] = cropped
	return canvas


def _atk_chain(img: np.ndarray) -> np.ndarray:
	# Very strong composite: translate 4 -> resize 50% -> JPEG 50 -> rotate 10 -> noise 10
	out = _atk_translate(img, 4)
	out = _atk_resize(out, 50)
	out = _atk_jpeg(out, 50)
	out = _atk_rotate(out, 10)
	out = _atk_noise(out, 10.0)
	return out

# Mapping for presets

def _preset_param(kind: str, level: str):
	if level == "Custom":
		return None
	if kind == "JPEG":
		return {"Mild": 85, "Strong": 60, "Very strong": 45}[level]
	if kind == "Resize":
		return {"Mild": 85, "Strong": 60, "Very strong": 40}[level]
	if kind == "Rotate":
		return {"Mild": 3, "Strong": 8, "Very strong": 12}[level]
	if kind == "Noise":
		return {"Mild": 4.0, "Strong": 10.0, "Very strong": 15.0}[level]
	if kind == "Crop":
		return {"Mild": 85, "Strong": 70, "Very strong": 60}[level]
	if kind == "Blur":
		return {"Mild": 7, "Strong": 11, "Very strong": 15}[level]
	if kind == "Translate":
		return {"Mild": 2, "Strong": 6, "Very strong": 10}[level]
	if kind == "Desync Crop":
		return {"Mild": 90, "Strong": 80, "Very strong": 70}[level]
	return None

if gen_btn:
	if not atk_file:
		res_placeholder.warning("Upload a watermarked image first")
	elif not atk_text:
		res_placeholder.warning("Enter the original watermark text")
	else:
		try:
			img_bgr = cv2.imdecode(np.frombuffer(atk_file.read(), np.uint8), cv2.IMREAD_COLOR)
			if img_bgr is None:
				res_placeholder.error("Failed to read the image")
			else:
				if atk_type == "Chain (Very strong)":
					atk = _atk_chain(img_bgr)
					used_param = "Translate4+Resize50+JPEG50+Rot10+Noise10"
				else:
					p = _preset_param(atk_type, preset)
					use_param = param if p is None else p
					used_param = use_param
					if atk_type == "JPEG":
						atk = _atk_jpeg(img_bgr, int(use_param))
					elif atk_type == "Resize":
						atk = _atk_resize(img_bgr, int(use_param))
					elif atk_type == "Rotate":
						atk = _atk_rotate(img_bgr, int(use_param))
					elif atk_type == "Noise":
						atk = _atk_noise(img_bgr, float(use_param))
					elif atk_type == "Crop":
						atk = _atk_crop(img_bgr, int(use_param))
					elif atk_type == "Blur":
						atk = _atk_blur(img_bgr, int(use_param))
					elif atk_type == "Translate":
						atk = _atk_translate(img_bgr, int(use_param))
					else:
						atk = _atk_desync_crop(img_bgr, int(use_param))
				atk_placeholder.image(cv2.cvtColor(atk, cv2.COLOR_BGR2RGB), caption=f"Attacked: {atk_type} ({used_param})", use_container_width=True)
				score, _ = detect_watermark_bgr(atk, atk_text, seed=seed, mode=mode, check_integrity=embed_integrity)
				verdict = "✅ AUTHENTIC" if score >= detect_threshold else "⚠️ TAMPERED/NOT PRESENT"
				res_placeholder.write(f"Similarity: {score*100:.2f}% | Threshold: {detect_threshold*100:.0f}% → {verdict}")
				# Download attacked image
				ok, enc = cv2.imencode('.jpg', atk, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
				if ok:
					download_placeholder.download_button("Download attacked image", data=enc.tobytes(), file_name=(filename_input or "attacked.jpg"), mime="image/jpeg")
				# Optional save to results
				if save_toggle:
					os.makedirs("results", exist_ok=True)
					outp = os.path.join("results", filename_input or f"attacked_chain.jpg")
					cv2.imwrite(outp, atk)
					res_placeholder.success(f"Saved to {outp}")
		except Exception as e:
			res_placeholder.error(f"Error: {e}")
