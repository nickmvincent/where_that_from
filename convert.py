import PyPDF2

def main():
    """Driver"""
    pdf_file = open('sample_paper.pdf')
    read_pdf = PyPDF2.PdfFileReader(pdf_file)
    num_pages = read_pdf.getNumPages()
    page = read_pdf.getPage(0)
    page_content = page.extractText()
    print(num_pages)
    print(page_content)

if __name__ == '__main__':
    main()