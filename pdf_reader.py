from pypdf import PdfReader

reader = PdfReader(r"C:\Users\ana_s\onboarding\nature.pdf") # adjust to your actual path
paper_text = "".join(page.extract_text() or "" for page in reader.pages)

print(len(paper_text), "characters extracted") # sanity check — should be a few tens of thousands