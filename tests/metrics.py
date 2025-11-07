import numpy as np

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
    # fallback simple placeholder using normalized cross-correlation
    a = img_a.astype(np.float64).flatten()
    b = img_b.astype(np.float64).flatten()
    if a.size == 0 or b.size == 0:
        return 0.0
    denom = (np.std(a) * np.std(b))
    if denom == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])
