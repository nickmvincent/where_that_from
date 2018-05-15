"""
Don't use this.
"""
import PyPDF2

def main():
    """Driver"""
    text = ""
    filename = 'sample_paper.pdf'
    with open(filename, 'rb') as f:
        read_pdf = PyPDF2.PdfFileReader(f)
        num_pages = read_pdf.getNumPages()
        for page_num in range(num_pages):
            page = read_pdf.getPage(page_num)
            page_content = page.extractText()
            text += page_content
    with open('pdfs_as_txt/' + filename.replace('.pdf', '.txt'), 'w') as f:
        f.write(text)

if __name__ == '__main__':
    main()