import fitz  # PyMuPDF

class PDFParser:

    def extract_text(self, file_path):
        text = ""

        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text()

        return text