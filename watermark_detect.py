import argparse
from typing import Tuple, List

import cv2
import numpy as np
import hashlib

from watermark_embed import _block_view, _dct2, _idct2


def _select_indices(block_size: int, mode: str):
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
				if freq_sum >= 8:
					indices.append((i, j))
			else:
				if 2 <= freq_sum <= 6:
					indices.append((i, j))
	return indices


def text_to_bits(text: str) -> np.ndarray:
	data = text.encode("utf-8")
	bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
	return bits.astype(np.int8)


def detect_watermark_bgr(image_bgr: np.ndarray, original_text: str, alpha: float = 12.0, seed: int = 12345, mode: str = "robust", check_integrity: bool = False) -> Tuple[float, dict]:
	"""Detect watermark by correlating PN-impacted DCT coefficients.

	Returns similarity score [0..1] and info dict.
	"""
	if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
		raise ValueError("Input must be a BGR color image")

	mode = mode.lower().strip()
	if mode not in ("robust", "fragile"):
		mode = "robust"

	# Convert to Y channel
	ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
	Y = ycrcb[:, :, 0].astype(np.float32)

	# Pad
	h, w = Y.shape
	pad_h = (8 - (h % 8)) % 8
	pad_w = (8 - (w % 8)) % 8
	Y_padded = cv2.copyMakeBorder(Y, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

	# Rebuild PN sequences
	bits = text_to_bits(original_text)
	if bits.size == 0:
		raise ValueError("Original watermark text encodes to empty bit array")

	indices = _select_indices(8, mode)
	rng = np.random.default_rng(seed)
	seq_len = 64
	pn_sequences = {}
	for idx in indices:
		pn = rng.choice([-1, 1], size=seq_len, replace=True).astype(np.int8)
		pn_sequences[idx] = pn

	blocks = _block_view(Y_padded, 8)
	bh, bw = blocks.shape[:2]
	n_total_blocks = bh * bw
	# Determine expected payload length (message bits + optional integrity bits)
	integrity_bits_len = 256 if check_integrity else 0
	payload_bits_len = bits.size + integrity_bits_len

	if mode == "fragile":
		repeat_factor = 1
	else:
		repeat_factor = max(1, n_total_blocks // max(1, payload_bits_len))

	# We'll reconstruct recovered bits per DCT-block then aggregate per repeated group
	recovered_block_bits = []
	block_idx = 0
	for by in range(bh):
		for bx in range(bw):
			block = blocks[by, bx].astype(np.float32)
			block_dct = _dct2(block)
			acc = 0.0
			for k, (i, j) in enumerate(indices):
				pn = pn_sequences[(i, j)]
				sign = pn[k % seq_len]
				coef = block_dct[i, j]
				acc += coef * sign
			recovered_block_bits.append(1 if acc >= 0 else 0)
			block_idx += 1

	recovered_block_bits = np.array(recovered_block_bits, dtype=np.int8)

	# Aggregate per original payload bit (taking majority over the repeat_factor)
	num_payload_bits = int(np.ceil(n_total_blocks / repeat_factor))
	recovered_payload = []
	for i in range(num_payload_bits):
		start = i * repeat_factor
		end = min(start + repeat_factor, recovered_block_bits.size)
		group = recovered_block_bits[start:end]
		# majority vote
		if group.size == 0:
			bit = 0
		else:
			bit = 1 if group.sum() >= (group.size / 2.0) else 0
		recovered_payload.append(bit)

	recovered_payload = np.array(recovered_payload, dtype=np.int8)

	# Now split payload into message bits and optional integrity bits
	msg_bits_len = bits.size
	if msg_bits_len > recovered_payload.size:
		# Not enough payload bits recovered to even contain the message
		message_match_ratio = 0.0
		recovered_message = np.array([], dtype=np.int8)
		recovered_integrity = np.array([], dtype=np.int8)
	else:
		recovered_message = recovered_payload[:msg_bits_len]
		recovered_integrity = recovered_payload[msg_bits_len:]
		# Compare recovered message to provided original_text
		# Convert recovered_message (bits) to bytes/text for a softer comparison
		# First pad to full bytes if needed
		usable = (recovered_message.size // 8) * 8
		recovered_message_bytes = np.packbits(recovered_message[:usable]).tobytes() if usable > 0 else b""
		try:
			recovered_text = recovered_message_bytes.decode("utf-8", errors="ignore")
		except Exception:
			recovered_text = ""
		# compute simple match ratio between bit arrays
		matches = (recovered_message == np.repeat(bits[:recovered_message.size], 1)[:recovered_message.size])
		message_match_ratio = float(matches.mean()) if matches.size > 0 else 0.0

	# Integrity check: if requested compare recovered integrity bits to recomputed fingerprint
	integrity_ok = None
	if check_integrity:
		# compute fingerprint bits of the (possibly attacked) image (use chroma channels to avoid Y-channel watermark)
		def _compute_fingerprint_bits(img_compact: np.ndarray, size: int = 64) -> np.ndarray:
			"""Compute SHA-256 bits from low-resolution chroma image (handles multi-channel)"""
			blur = cv2.GaussianBlur(img_compact.astype(np.uint8), (9, 9), sigmaX=3)
			small = cv2.resize(blur, (size, size), interpolation=cv2.INTER_AREA)
			hbytes = small.tobytes()
			digest = hashlib.sha256(hbytes).digest()
			bits_arr = np.unpackbits(np.frombuffer(digest, dtype=np.uint8)).astype(np.int8)
			return bits_arr

		# compute fingerprint on chroma channels (Cr,Cb)
		ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
		fp_bits = _compute_fingerprint_bits(ycrcb[:, :, 1:])
		# compare recovered_integrity (may be shorter) to the fp_bits prefix
		min_len = min(recovered_integrity.size, fp_bits.size)
		if min_len == 0:
			integrity_ok = False
		else:
			integrity_ok = bool((recovered_integrity[:min_len] == fp_bits[:min_len]).all())

	info = {"repeat_blocks": int(repeat_factor), "num_votes": int(recovered_block_bits.size), "mode": mode, "message_match_ratio": message_match_ratio, "integrity_ok": integrity_ok}

	# For backward compatibility return score as message_match_ratio
	score = message_match_ratio
	# If integrity checking is enabled and the integrity failed, treat as tampered
	if check_integrity and integrity_ok is False:
		score = 0.0

	return score, info


def main():
	parser = argparse.ArgumentParser(description="DCT-based invisible watermark detection")
	parser.add_argument("--input", required=True, help="Watermarked image path (JPG/PNG)")
	parser.add_argument("--text", required=True, help="Original watermark text message")
	parser.add_argument("--seed", type=int, default=12345, help="Random seed used during embedding")
	parser.add_argument("--mode", choices=["robust", "fragile"], default="robust", help="Detection mode (must match embedding)")
	args = parser.parse_args()

	img_bgr = cv2.imread(args.input, cv2.IMREAD_COLOR)
	if img_bgr is None:
		raise FileNotFoundError(f"Cannot read image: {args.input}")

	score, info = detect_watermark_bgr(img_bgr, args.text, seed=args.seed, mode=args.mode)
	print(f"Similarity score: {score*100:.2f}% | mode={info['mode']} repeat_blocks={info['repeat_blocks']} votes={info['num_votes']}")
	print("Detection:", "SUCCESS" if score >= 0.6 else "FAIL")


if __name__ == "__main__":
	main()
