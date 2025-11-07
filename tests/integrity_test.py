import os
import sys
import cv2
# Ensure project root on path
sys.path.insert(0, os.path.dirname(__file__))
# ensure tests directory is first so our lightweight metrics fallback is used during this quick test
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from watermark_embed import embed_watermark_bgr, save_image
from watermark_detect import detect_watermark_bgr

os.makedirs('watermarked', exist_ok=True)
img = cv2.imread('images/architecture.jpg', cv2.IMREAD_COLOR)
if img is None:
    raise SystemExit('Missing test image')

text = 'Test message 123'
wm, info = embed_watermark_bgr(img, text, embed_integrity=True)
print('Embed info:', info)
save_image('watermarked/test_wm.jpg', wm, quality=95)

# Detect on original watermarked image
score, info2 = detect_watermark_bgr(wm, text, check_integrity=True)
print('Detect on WM: score=', score, 'info=', info2)

# Directly compute chroma fingerprints to debug
import hashlib
import numpy as np
orig_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
wm_ycrcb = cv2.cvtColor(wm, cv2.COLOR_BGR2YCrCb)
def fp_bits_of(arr):
    small = cv2.resize(arr.astype('uint8'), (64,64), interpolation=cv2.INTER_AREA)
    d = hashlib.sha256(small.tobytes()).digest()
    return np.unpackbits(np.frombuffer(d, dtype=np.uint8)).astype(np.int8)
fp_orig = fp_bits_of(orig_ycrcb[:, :, 1:])
fp_wm = fp_bits_of(wm_ycrcb[:, :, 1:])
print('fp_orig == fp_wm?', (fp_orig == fp_wm).all(), 'len', fp_orig.size)
print('chroma max abs diff:', int(np.max(np.abs(orig_ycrcb[:, :, 1:].astype(int) - wm_ycrcb[:, :, 1:].astype(int)))))

# Attack: JPEG compress
ok, enc = cv2.imencode('.jpg', wm, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
attacked = cv2.imdecode(enc, cv2.IMREAD_COLOR)
cv2.imwrite('results/test_attacked.jpg', attacked)

score2, info3 = detect_watermark_bgr(attacked, text, check_integrity=True)
print('Detect on attacked: score=', score2, 'info=', info3)
