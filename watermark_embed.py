import argparse
import os
from typing import Tuple, List

import cv2
import numpy as np
from PIL import Image
import hashlib

from metrics import compute_psnr


# --- DCT helpers ---

def _block_view(arr: np.ndarray, block_size: int) -> np.ndarray:
	H, W = arr.shape
	bh, bw = block_size, block_size
	assert H % bh == 0 and W % bw == 0, "Image dimensions must be multiples of block size"
	return arr.reshape(H // bh, bh, W // bw, bw).swapaxes(1, 2)


def _idct2(block: np.ndarray) -> np.ndarray:
	return cv2.idct(block)


def _dct2(block: np.ndarray) -> np.ndarray:
	return cv2.dct(block)


# --- Watermark encoding utilities ---

def text_to_bits(text: str) -> np.ndarray:
	data = text.encode("utf-8")
	bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
	return bits.astype(np.int8)


def bits_to_text(bits: np.ndarray) -> str:
	# Truncate to multiple of 8
	usable = (bits.size // 8) * 8
	bits = bits[:usable].astype(np.uint8)
	if usable == 0:
		return ""
	bytes_arr = np.packbits(bits)
	try:
		return bytes(bytes_arr.tobytes()).decode("utf-8", errors="ignore")
	except Exception:
		return ""


def _select_indices(block_size: int, mode: str) -> List[Tuple[int, int]]:
	# robust: mid-band; fragile: high-band (more sensitive). fragile_strict uses even higher freqs.
	indices = []
	for i in range(block_size):
		for j in range(block_size):
			if (i, j) == (0, 0):
				continue
			freq_sum = i + j
			if mode == "fragile_strict":
				if freq_sum >= 10:
					indices.append((i, j))
			elif mode == "fragile":
				if freq_sum >= 8:  # high band for 8x8
					indices.append((i, j))
			else:
				if 2 <= freq_sum <= 6:
					indices.append((i, j))
	return indices


def embed_watermark_bgr(image_bgr: np.ndarray, watermark_text: str, alpha: float = 12.0, seed: int = 12345, mode: str = "robust", embed_integrity: bool = False) -> Tuple[np.ndarray, dict]:
	"""Embed watermark text using spread-spectrum in DCT coefficients on Y channel.

	mode: 'robust' (mid-band) or 'fragile' (high-band, less redundant, more tamper-sensitive)
	Returns watermarked BGR image and info dict with internal parameters.
	"""
	if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
		raise ValueError("Input must be a BGR color image")

	mode = mode.lower().strip()
	if mode not in ("robust", "fragile"):
		mode = "robust"

	# Convert to YCrCb and use Y channel
	ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
	Y = ycrcb[:, :, 0].astype(np.float32)

	# Pad to multiples of 8
	h, w = Y.shape
	pad_h = (8 - (h % 8)) % 8
	pad_w = (8 - (w % 8)) % 8
	Y_padded = cv2.copyMakeBorder(Y, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

	# Prepare watermark bits and PRN sequences
	bits = text_to_bits(watermark_text)
	if bits.size == 0:
		raise ValueError("Watermark text is empty after encoding")

	def _compute_fingerprint_bits(img_gray: np.ndarray, size: int = 64) -> np.ndarray:
		"""Compute a compact SHA-256 fingerprint of a downsampled grayscale image and return 256 bits."""
		# Apply a mild low-pass (blur) to remove mid/high-frequency watermark energy,
		# then downsample to fixed small size so the fingerprint encodes coarse structure
		blur = cv2.GaussianBlur(img_gray.astype(np.uint8), (9, 9), sigmaX=3)
		small = cv2.resize(blur, (size, size), interpolation=cv2.INTER_AREA)
		hbytes = small.tobytes()
		digest = hashlib.sha256(hbytes).digest()
		bits_arr = np.unpackbits(np.frombuffer(digest, dtype=np.uint8)).astype(np.int8)
		return bits_arr

	# embed_integrity parameter controls whether a 256-bit SHA-256 fingerprint of the image is appended

	indices = _select_indices(8, mode)
	rng = np.random.default_rng(seed)
	pn_sequences = {}
	seq_len = 64
	for idx in indices:
		pn = rng.choice([-1, 1], size=seq_len, replace=True).astype(np.int8)
		pn_sequences[idx] = pn

	blocks = _block_view(Y_padded, 8)
	bh, bw = blocks.shape[:2]
	n_total_blocks = bh * bw
	# If integrity embedding enabled, append fingerprint bits to the message bits
	if embed_integrity:
		# compute fingerprint on chroma channels (Cr,Cb) to avoid modification by Y-channel embedding
		fp_bits = _compute_fingerprint_bits(ycrcb[:, :, 1:])
		payload_bits = np.concatenate([bits, fp_bits])
	else:
		payload_bits = bits

	if mode == "fragile":
		repeat_factor = 1
		alpha_use = max(4.0, alpha * 0.6)
	else:
		repeat_factor = max(1, n_total_blocks // max(1, payload_bits.size))
		alpha_use = alpha

	repeated_bits = np.repeat(payload_bits, repeat_factor)
	# ensure the payload covers at most the available number of blocks
	repeated_bits = repeated_bits[: n_total_blocks]

	# Note: if message is too short for the number of blocks in fragile mode, we still limit by n_total_blocks
	# If desired, callers can pass longer messages or enable repeat (robust mode)

	wm_Y = Y_padded.copy()
	block_idx = 0
	for by in range(bh):
		for bx in range(bw):
			# protect against shorter repeated_bits by wrapping (cyclic mapping)
			bit = int(repeated_bits[block_idx % repeated_bits.size]) if repeated_bits.size > 0 else 0
			block = blocks[by, bx].astype(np.float32)
			block_dct = _dct2(block)
			for k, (i, j) in enumerate(indices):
				pn = pn_sequences[(i, j)]
				sign = pn[k % seq_len]
				block_dct[i, j] += alpha_use * sign * (1 if bit == 1 else -1)
			block_idct = _idct2(block_dct)
			wm_Y[by * 8 : (by + 1) * 8, bx * 8 : (bx + 1) * 8] = block_idct
			block_idx += 1

	wm_Y = wm_Y[:h, :w]
	ycrcb[:, :, 0] = np.clip(wm_Y, 0, 255).astype(np.uint8)
	wm_bgr = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

	info = {
		"alpha": float(alpha_use),
		"seed": int(seed),
		"bits_len": int(bits.size),
		"payload_bits_len": int(payload_bits.size),
		"repeat_blocks": int(repeat_factor),
		"mode": mode,
		"embed_integrity": bool(embed_integrity),
	}
	return wm_bgr, info


def save_image(path: str, img_bgr: np.ndarray, quality: int = 95) -> None:
	ext = os.path.splitext(path)[1].lower()
	if ext in [".jpg", ".jpeg"]:
		cv2.imwrite(path, img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
	elif ext == ".png":
		cv2.imwrite(path, img_bgr, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
	else:
		cv2.imwrite(path, img_bgr)


def main():
	parser = argparse.ArgumentParser(description="DCT-based invisible watermark embedding")
	parser.add_argument("--input", required=True, help="Input image path (JPG/PNG)")
	parser.add_argument("--text", required=True, help="Watermark text message")
	parser.add_argument("--alpha", type=float, default=12.0, help="Watermark strength")
	parser.add_argument("--seed", type=int, default=12345, help="Random seed for PN sequences")
	parser.add_argument("--mode", choices=["robust", "fragile"], default="robust", help="Embedding mode")
	parser.add_argument("--output", required=True, help="Output path in watermarked/")
	args = parser.parse_args()

	img_bgr = cv2.imread(args.input, cv2.IMREAD_COLOR)
	if img_bgr is None:
		raise FileNotFoundError(f"Cannot read image: {args.input}")

	wm_bgr, info = embed_watermark_bgr(img_bgr, args.text, alpha=args.alpha, seed=args.seed, mode=args.mode)
	os.makedirs(os.path.dirname(args.output), exist_ok=True)
	save_image(args.output, wm_bgr)

	psnr = compute_psnr(img_bgr, wm_bgr)
	print(f"Saved watermarked image to: {args.output}")
	print(f"PSNR: {psnr:.2f} dB | mode={info['mode']} alpha={info['alpha']} repeat_blocks={info['repeat_blocks']}")


if __name__ == "__main__":
	main()
