from subprocess import call
import glob

from split_sentences import split_sentences

# first convert .pdf to .txt using the command line utility pdftotext from Xpdf
for folder in [
    'pdf_files/psa_research', 
    #'pdf_files/chi_2018_pdf_files'
]:
    pdf_filepaths = glob.glob("{}/*.pdf".format(folder))
    for pdf_filepath in pdf_filepaths:
        txt_filepath = pdf_filepath.lower().replace("pdf_files", "txt_files").replace('.pdf', '.txt')
        call(["pdftotext", pdf_filepath, txt_filepath])
        split_sentences(txt_filepath)
        




