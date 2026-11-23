"""Regenerate samples/medical_reimbursement_claim.pdf (a text-based PDF sample)."""
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

HERE = Path(__file__).parent
SRC = HERE / "medical_reimbursement_claim.txt"
OUT = HERE / "medical_reimbursement_claim.pdf"


def main() -> None:
    lines = SRC.read_text(encoding="utf-8").splitlines()
    c = canvas.Canvas(str(OUT), pagesize=A4)
    c.setTitle("Medical Reimbursement Claim (synthetic sample)")
    width, height = A4
    y = height - 20 * mm
    for line in lines:
        if y < 20 * mm:
            c.showPage()
            y = height - 20 * mm
        # reportlab's base fonts lack the rupee glyph; spell it out
        c.setFont("Helvetica", 10)
        c.drawString(18 * mm, y, line.replace("₹", "Rs."))
        y -= 5 * mm
    c.save()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
