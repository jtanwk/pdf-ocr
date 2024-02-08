# parse_table_obj.py
#
# Description:
# This script is an object-oriented approach to table extraction from PDFs.
# It relies heavily on the Document and Page objects defined in parser.py.

import datetime
import itertools
import logging
import os
import shutil

import pandas as pd

from parser import Document


def clear_contents(dir_path):
    """
    Deletes the contents of the given filepath. Useful for testing runs.
    """

    filelist = os.listdir(dir_path)
    if filelist:
        for f in filelist:
            if os.path.isdir(os.path.join(dir_path, f)):
                shutil.rmtree(os.path.join(dir_path, f))
            else:
                os.remove(os.path.join(dir_path, f))

    return None


def parse_document(data_dir, output_dir, doc_path):
    """
    This is a separate function to facilitate parallelization.
    Returns a dictionary in case of an error, else None.
    """
    pdf_doc = Document(doc_path, data_dir, output_dir)

    return pdf_doc.parse_doc()


def main(data_dir, output_dir):
    """
    Main control flow:
        1. Checks if required folders exist; if not, creates them
        2. Loops over each PDF file in data_path and calls parse_pdf().
        3. Output CSVs are written to output_path.
    """

    # Clear output folder
    clear_contents(output_dir)

    # Check if organizing folders exist
    for i in [data_dir, output_dir]:
        try:
            if i == data_dir and not os.path.exists(data_dir):
                raise Exception("Data folder is missing or not assigned.")
            else:
                os.mkdir(i)
        except FileExistsError:
            continue

    # Get list of pdfs to parse
    pdf_list = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
    logger.info(f"{len(pdf_list)} file(s) detected.")

    # Initialize pool and parallelize processing
    errors = []
    for pdf in pdf_list:
        error = parse_document(data_dir, output_dir, pdf)
        if error:
            errors.append(error)

    # Export errors to CSV if they exlist
    if len(errors) > 0:
        error_df = pd.DataFrame(list(itertools.chain(*errors)))
        error_path = os.path.join(output_dir, 'errors.csv')
        logger.info(f"Completed with {len(errors)} errors. Exporting to {error_path}")
        error_df.to_csv(error_path, index=False, columns=['document', 'page', 'error'])
    else:
        logger.info("Completed with no errors detected.")

    # Non-parallel version
    # Loop over PDF files, create Document objects, call Document.parse()
    # for i in sorted(pdf_list):
    #     # Parse pdf
    #     logger.info(f"Parsing file: {os.path.join(data_dir, i)}")
    #     pdf_doc = Document(i, data_dir, output_dir)
    #     pdf_doc.parse_doc()

    return None


if __name__ == "__main__":

    # Key paths and parameters
    DATA_DIR = "01_data"
    OUTPUT_DIR = "02_output"

    # Initialize logger
    if os.path.exists('parse_table.log'):
        os.remove('parse_table.log')
    logger = logging.getLogger('parse_table')
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    fh = logging.FileHandler('parse_table.log')
    logger.addHandler(ch)
    logger.addHandler(fh)

    # Run main control flow
    start = datetime.datetime.now()
    main(DATA_DIR, OUTPUT_DIR)
    duration = datetime.datetime.now() - start
    print(f"Time taken: {duration}")
