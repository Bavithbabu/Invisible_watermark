import argparse
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd

from watermark_detect import detect_watermark_bgr
from metrics import compute_psnr, compute_mse, compute_ssim


def jpeg_compress(img_bgr: np.ndarray, quality: int) -> np.ndarray:
	encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
	_, enc = cv2.imencode('.jpg', img_bgr, encode_param)
	return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def resize_scale(img_bgr: np.ndarray, scale: float) -> np.ndarray:
	h, w = img_bgr.shape[:2]
	res = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
	# restore original size to compare
	res = cv2.resize(res, (w, h), interpolation=cv2.INTER_CUBIC)
	return res


def rotate_small(img_bgr: np.ndarray, angle_deg: float) -> np.ndarray:
	h, w = img_bgr.shape[:2]
	center = (w // 2, h // 2)
	M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
	rot = cv2.warpAffine(img_bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
	return rot


def add_gaussian_noise(img_bgr: np.ndarray, sigma: float) -> np.ndarray:
	noise = np.random.normal(0, sigma, img_bgr.shape).astype(np.float32)
	noisy = np.clip(img_bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
	return noisy


def evaluate(img_bgr: np.ndarray, ref_bgr: np.ndarray, text: str) -> Dict[str, float]:
	psnr = compute_psnr(ref_bgr, img_bgr)
	mse = compute_mse(ref_bgr, img_bgr)
	ssim = compute_ssim(ref_bgr, img_bgr)
	score, _ = detect_watermark_bgr(img_bgr, text)
	return {"PSNR": psnr, "MSE": mse, "SSIM": ssim, "Similarity": score}


def main():
	parser = argparse.ArgumentParser(description="Robustness evaluation for watermarked images")
	parser.add_argument("--input", required=True, help="Watermarked image path")
	parser.add_argument("--text", required=True, help="Original watermark text")
	parser.add_argument("--out_dir", default="results", help="Output directory")
	args = parser.parse_args()

	os.makedirs(args.out_dir, exist_ok=True)

	ref = cv2.imread(args.input, cv2.IMREAD_COLOR)
	if ref is None:
		raise FileNotFoundError(args.input)

	rows: List[Dict[str, float]] = []

	# 1) JPEG compression
	for q in [90, 70, 50]:
		atk = jpeg_compress(ref, q)
		metrics = evaluate(atk, ref, args.text)
		metrics.update({"Attack": f"JPEG_{q}", "Param": q})
		rows.append(metrics)
		cv2.imwrite(os.path.join(args.out_dir, f"atk_jpeg_{q}.jpg"), atk)

	# 2) Resize
	for s in [0.75, 0.5]:
		atk = resize_scale(ref, s)
		metrics = evaluate(atk, ref, args.text)
		metrics.update({"Attack": f"RESIZE_{s}", "Param": s})
		rows.append(metrics)
		cv2.imwrite(os.path.join(args.out_dir, f"atk_resize_{int(s*100)}.jpg"), atk)

	# 3) Rotation
	for a in [2.0, -2.0, 5.0]:
		atk = rotate_small(ref, a)
		metrics = evaluate(atk, ref, args.text)
		metrics.update({"Attack": f"ROT_{a}", "Param": a})
		rows.append(metrics)
		cv2.imwrite(os.path.join(args.out_dir, f"atk_rot_{int(a)}.jpg"), atk)

	# 4) Gaussian noise
	for s in [2.0, 5.0, 10.0]:
		atk = add_gaussian_noise(ref, s)
		metrics = evaluate(atk, ref, args.text)
		metrics.update({"Attack": f"NOISE_{s}", "Param": s})
		rows.append(metrics)
		cv2.imwrite(os.path.join(args.out_dir, f"atk_noise_{int(s)}.jpg"), atk)

	df = pd.DataFrame(rows)
	csv_path = os.path.join(args.out_dir, "robustness_results.csv")
	df.to_csv(csv_path, index=False)
	print(f"Saved metrics CSV: {csv_path}")

	# Defer PDF generation to report.py to keep concerns separated
	try:
		from report import generate_pdf_report
		pdf_path = os.path.join(args.out_dir, "robustness_report.pdf")
		generate_pdf_report(df, pdf_path, title="Invisible Watermark Robustness Report")
		print(f"Saved PDF report: {pdf_path}")
	except Exception as e:
		print(f"PDF report generation skipped: {e}")


if __name__ == "__main__":
	main()
