import os
import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_file_path):
    try:
        text = ''
        with fitz.open(pdf_file_path) as pdf_file:
            for page_num in range(pdf_file.page_count):
                page = pdf_file.load_page(page_num)
                text += page.get_text()
        return text

def process_pdf_files_in_directory(directory_path):
    pdf_text = {}
    try:
        for filename in os.listdir(directory_path):
            if filename.endswith('.pdf'):
                pdf_file_path = os.path.join(directory_path, filename)
                text = extract_text_from_pdf(pdf_file_path)
                pdf_text[filename] = text
                
    except Exception as e:
        print(f"Error processing PDF files in {directory_path}: {e}")
    return pdf_text

def export_pdf_text_to_txt(pdf_text, output_txt_path):
    with open(output_txt_path, 'w', encoding='utf-8') as txt_file:
        for text in pdf_text.items():
            txt_file.write(text)
    print(f"PDF text exported to '{output_txt_path}'.")

# Use the functions to extract words from the corpus to form a dict
pdf_directory = 'C:/Users/pangy/Downloads/Corpus Extend'
output_directory = 'C:/Users/pangy/Downloads' 
output_txt_path = 'extracted_text.txt'

pdf_text = process_pdf_files_in_directory(pdf_directory)
export_pdf_text_to_txt(pdf_text, os.path.join(output_directory, output_txt_path))

