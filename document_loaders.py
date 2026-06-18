import os 
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, pdf


CURRENT_PATH = Path(__file__).resolve().parent

load_dotenv()

def pdf_loader(file_path):
    loader = PyPDFLoader(file_path=file_path)
    pages = loader.load()
    print(f"Loaded {len(pages)} pages from the pdf")
    for i, doc in enumerate(pages):
        print(f"Document {i+1} Content Preview: {doc.page_content[:100]}")
        print(f"metadata {doc.metadata}")

if __name__ == "__main__":
    pdf_path = Path.joinpath(CURRENT_PATH, "docs", "pdfTest.pdf")
    pdf_loader(pdf_path)
