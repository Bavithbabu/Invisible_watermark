import numpy as np
try:
	from skimage.metrics import structural_similarity as ssim
except Exception:
	# Fallback implementation if scikit-image isn't installed (used for quick local tests)
	def ssim(a, b, data_range=None):
		a = a.astype(np.float64).flatten()
		b = b.astype(np.float64).flatten()
		if a.size == 0 or b.size == 0:
			return 0.0
		denom = (np.std(a) * np.std(b))
		if denom == 0:
			return 0.0
		return float(np.corrcoef(a, b)[0, 1])


def compute_mse(img_a: np.ndarray, img_b: np.ndarray) -> float:
	if img_a.shape != img_b.shape:
		raise ValueError("Images must have the same shape for MSE")
	img_a_f = img_a.astype(np.float64)
	img_b_f = img_b.astype(np.float64)
	return float(np.mean((img_a_f - img_b_f) ** 2))


def compute_psnr(img_a: np.ndarray, img_b: np.ndarray, max_val: float = 255.0) -> float:
	mse = compute_mse(img_a, img_b)
	if mse == 0:
		return float('inf')
	return 20.0 * np.log10(max_val) - 10.0 * np.log10(mse)


def compute_ssim(img_a: np.ndarray, img_b: np.ndarray) -> float:
	# Convert to grayscale if needed
	if img_a.ndim == 3 and img_a.shape[2] == 3:
		img_a_gray = 0.299 * img_a[:, :, 2] + 0.587 * img_a[:, :, 1] + 0.114 * img_a[:, :, 0]
		img_b_gray = 0.299 * img_b[:, :, 2] + 0.587 * img_b[:, :, 1] + 0.114 * img_b[:, :, 0]
	else:
		img_a_gray = img_a
		img_b_gray = img_b
	img_a_gray = img_a_gray.astype(np.float64)
	img_b_gray = img_b_gray.astype(np.float64)
	return float(ssim(img_a_gray, img_b_gray, data_range=img_b_gray.max() - img_b_gray.min()))


def correlation_coefficient(a: np.ndarray, b: np.ndarray) -> float:
	if a.size != b.size:
		raise ValueError("Arrays must have same size for correlation coefficient")
	if a.ndim > 1:
		a = a.flatten()
	if b.ndim > 1:
		b = b.flatten()
	if np.std(a) == 0 or np.std(b) == 0:
		return 0.0
	return float(np.corrcoef(a, b)[0, 1])
