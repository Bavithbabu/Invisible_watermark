## Invisible Watermark System (Images Only)

Python-based DCT watermark embedding and detection with robustness tests, Streamlit GUI, and PDF reporting.

### Features

- DCT-based invisible watermark (text/signature) in frequency domain
- Robust against JPEG compression, mild resizing, small rotations, and Gaussian noise
- Streamlit GUI: image selection, watermark text input, alpha strength slider, embed/detect buttons
- Batch processing for multiple images
- Metrics: PSNR, MSE, SSIM, Correlation Coefficient
- Robustness testing script (compression, resize, rotation, noise)
- PDF report generation with results

### Project Structure

```
/images            # put your original images here
/watermarked       # generated watermarked images
/results           # evaluation outputs and reports
watermark_embed.py
watermark_detect.py
metrics.py
robustness_eval.py
report.py
app.py             # Streamlit GUI
requirements.txt
README.md
```

### Setup

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
. .venv/Scripts/activate        # Windows PowerShell
pip install -r requirements.txt
```

2. Prepare images:

- Place JPG/PNG images in `images/`

### Usage

#### Streamlit App

```bash
streamlit run app.py
```

- Browse and select one or multiple images
- Enter watermark text
- Adjust alpha strength (trade-off robustness vs imperceptibility)
- Click Embed to generate images into `watermarked/`
- Click Detect to verify watermark on selected file

#### CLI Embedding

```bash
python watermark_embed.py --input images/sample.jpg --text "Your © watermark" --alpha 12.0 --output watermarked/sample_wm.jpg
```

#### CLI Detection

```bash
python watermark_detect.py --input watermarked/sample_wm.jpg --text "Your © watermark"
```

#### Robustness Evaluation

```bash
python robustness_eval.py --input watermarked/sample_wm.jpg --text "Your © watermark" --out_dir results
```

### Notes

- PNG is lossless; robustness checks simulate JPEG compression and geometric/noise attacks.
- Small rotations and resizes are partially handled via redundancy and correlation; severe transforms will degrade detection.
- Tune `--alpha` (or UI slider) to balance invisibility (higher PSNR) and robustness.

### License

For educational use. Ensure you own rights to images you watermark.
