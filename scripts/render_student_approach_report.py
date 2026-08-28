# -*- coding: utf-8 -*-
"""Renders reports/student_approach/report.md AND report.pdf from the single
content source in student_approach_content.py, so the two documents cannot
drift apart. Run from the repo root.
"""
import os
import sys

sys.path.insert(0, "scripts")
from student_approach_content import TITLE, SUBTITLE, DATE, SECTIONS

OUT_DIR = "reports/student_approach"
os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------
md = [f"# {TITLE}\n", f"*{SUBTITLE}*  \n*{DATE}*\n"]
for heading, blocks in SECTIONS:
    md.append(f"\n## {heading}\n")
    for block in blocks:
        kind = block[0]
        if kind == "p":
            md.append(f"\n{block[1]}\n")
        elif kind == "bullets":
            md.append("")
            for item in block[1]:
                md.append(f"- {item}")
            md.append("")
        elif kind == "img":
            _, path, caption = block
            md.append(f"\n![{caption}]({path})\n*{caption}*\n")
with open(f"{OUT_DIR}/report.md", "w", encoding="utf-8") as f:
    f.write("\n".join(md))
print(f"wrote {OUT_DIR}/report.md")

# --------------------------------------------------------------------------
# PDF (fpdf2)
# --------------------------------------------------------------------------
from fpdf import FPDF

NAVY = (31, 58, 95)
GREY = (90, 98, 110)
INK = (26, 26, 26)

DEJAVU = "/usr/share/fonts/truetype/dejavu"


class Report(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("DejaVu", "I", 8)
        self.set_text_color(*GREY)
        self.cell(0, 8, TITLE, align="L")
        self.ln(12)

    def footer(self):
        self.set_y(-15)
        self.set_font("DejaVu", "I", 8)
        self.set_text_color(*GREY)
        self.cell(0, 10, f"{self.page_no()}", align="C")


pdf = Report(orientation="P", unit="mm", format="A4")
pdf.add_font("DejaVu", "", f"{DEJAVU}/DejaVuSans.ttf")
pdf.add_font("DejaVu", "B", f"{DEJAVU}/DejaVuSans-Bold.ttf")
pdf.add_font("DejaVu", "I", f"{DEJAVU}/DejaVuSans-Oblique.ttf")
pdf.add_font("DejaVu", "BI", f"{DEJAVU}/DejaVuSans-BoldOblique.ttf")
pdf.set_auto_page_break(auto=True, margin=20)
pdf.set_margins(20, 20, 20)
pdf.add_page()

# Title page content on page 1
pdf.set_font("DejaVu", "B", 22)
pdf.set_text_color(*NAVY)
pdf.ln(30)
pdf.multi_cell(0, 11, TITLE, align="C")
pdf.ln(4)
pdf.set_font("DejaVu", "", 13)
pdf.set_text_color(*GREY)
pdf.multi_cell(0, 8, SUBTITLE, align="C")
pdf.ln(2)
pdf.set_font("DejaVu", "I", 11)
pdf.cell(0, 8, DATE, align="C")
pdf.ln(20)
pdf.set_draw_color(*NAVY)
x = pdf.l_margin + 40
pdf.line(x, pdf.get_y(), pdf.w - x, pdf.get_y())

for heading, blocks in SECTIONS:
    pdf.add_page()
    pdf.set_font("DejaVu", "B", 15)
    pdf.set_text_color(*NAVY)
    pdf.multi_cell(0, 9, heading)
    pdf.ln(1)
    pdf.set_draw_color(220, 224, 230)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(4)

    for block in blocks:
        kind = block[0]
        if kind == "p":
            pdf.set_font("DejaVu", "", 10.5)
            pdf.set_text_color(*INK)
            pdf.multi_cell(0, 5.6, block[1])
            pdf.ln(2)
        elif kind == "bullets":
            pdf.set_font("DejaVu", "", 10.5)
            pdf.set_text_color(*INK)
            indent_w = pdf.w - pdf.r_margin - (pdf.l_margin + 4)
            for item in block[1]:
                pdf.set_x(pdf.l_margin + 4)
                pdf.multi_cell(indent_w, 5.6, f"-  {item}")
            pdf.ln(2)
        elif kind == "img":
            _, path, caption = block
            if pdf.get_y() > 200:
                pdf.add_page()
            avail_w = pdf.w - pdf.l_margin - pdf.r_margin
            pdf.image(f"{OUT_DIR}/{path}", x=pdf.l_margin, w=avail_w)
            pdf.ln(1)
            pdf.set_font("DejaVu", "I", 8.8)
            pdf.set_text_color(*GREY)
            pdf.multi_cell(0, 4.6, caption, align="C")
            pdf.ln(2)

pdf.output(f"{OUT_DIR}/report.pdf")
print(f"wrote {OUT_DIR}/report.pdf")
