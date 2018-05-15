from subprocess import call
import glob

# first convert .pdf to .txt using the command line utility pdftotext from Xpdf
pdf_filenames = glob.glob("pdf_files/*.pdf")
for pdf_filename in pdf_filenames:
    call(["pdftotext", pdf_filename, "txt_files/{}".format(pdf_filename.lower().replace('.pdf', '.txt'))])


