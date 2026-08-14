"""Extract text from SHMQ paper PDF and save as text file."""
import fitz  # PyMuPDF
import os

PDF = "/home/z/my-project/shmq-ultimate/paper/shmq_paper.pdf"
OUT_TXT = "/home/z/my-project/shmq-ultimate/paper/shmq_paper.txt"

doc = fitz.open(PDF)
print(f"Pages: {len(doc)}")
text_parts = []
for i, page in enumerate(doc):
    txt = page.get_text("text")
    text_parts.append(f"\n========== PAGE {i+1} ==========\n{txt}")

full = "".join(text_parts)
with open(OUT_TXT, "w", encoding="utf-8") as f:
    f.write(full)

print(f"Total chars: {len(full)}")
print(f"Saved to: {OUT_TXT}")
print("\n--- First 3000 chars ---")
print(full[:3000])
