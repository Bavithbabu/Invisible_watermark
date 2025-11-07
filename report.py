from typing import Optional

import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.units import cm


def generate_pdf_report(df: pd.DataFrame, out_path: str, title: str = "Watermark Robustness Report") -> None:
	c = canvas.Canvas(out_path, pagesize=A4)
	width, height = A4

	# Title
	c.setFont("Helvetica-Bold", 18)
	c.drawString(2 * cm, height - 2 * cm, title)

	# Summary
	c.setFont("Helvetica", 11)
	c.drawString(2 * cm, height - 3 * cm, f"Total tests: {len(df)}")
	if not df.empty and {'Similarity', 'PSNR', 'SSIM'}.issubset(df.columns):
		avg_sim = df['Similarity'].mean() * 100.0
		avg_psnr = df['PSNR'].mean()
		avg_ssim = df['SSIM'].mean()
		c.drawString(2 * cm, height - 3.6 * cm, f"Average Similarity: {avg_sim:.2f}%  |  PSNR: {avg_psnr:.2f} dB  |  SSIM: {avg_ssim:.3f}")

	# Table header
	y = height - 5 * cm
	c.setFont("Helvetica-Bold", 11)
	headers = ["Attack", "Param", "PSNR", "SSIM", "Similarity"]
	xs = [2 * cm, 7 * cm, 11 * cm, 14 * cm, 17 * cm]
	for x, hname in zip(xs, headers):
		c.drawString(x, y, hname)
	c.setLineWidth(0.5)
	c.setStrokeColor(colors.black)
	c.line(2 * cm, y - 0.2 * cm, 19 * cm, y - 0.2 * cm)

	# Rows
	c.setFont("Helvetica", 10)
	y -= 0.8 * cm
	for _, row in df.iterrows():
		if y < 2 * cm:
			c.showPage()
			y = height - 2 * cm
			c.setFont("Helvetica", 10)
		attack = str(row.get('Attack', ''))
		param = str(row.get('Param', ''))
		psnr = f"{row.get('PSNR', 0):.2f}"
		ssim = f"{row.get('SSIM', 0):.3f}"
		sim = f"{row.get('Similarity', 0) * 100.0:.2f}%"
		vals = [attack, param, psnr, ssim, sim]
		for x, v in zip(xs, vals):
			c.drawString(x, y, v)
		y -= 0.6 * cm

	c.save()
