# Extracting tabular data from scanned images

Jonathan Tan

## Installation

### On OSX

First, install the `tesseract` OCR engine by running `brew install tesseract` in the command line.

Then:
1. Download this folder to your computer. I'll refer to it as `root`, but you can name the folder whatever you want.
2. Launch the command line and navigate to the `root` folder. For example, if you downloaded it to your desktop, run `cd ./Desktop/root`.
3. _(Optional)_ Create a virtual environment by running `python3 -m venv env`, then activate it by running `$ source env/bin/activate` in your terminal. This ensures that everything you install in Step 4 won't interfere with other projects.
4. Install all requirements by running `pip3 install -r requirements.txt`.

### On Windows

First, install the `tesseract` OCR engine by downloading the installer `.exe` file [here](https://github.com/UB-Mannheim/tesseract/wiki) and run it.

Then:
1. Download this folder to your computer. I'll refer to it as `root`, but you can name the folder whatever you want.
2. Launch the command line and navigate to the `root` folder. For example, if you downloaded it to your desktop, run `cd ./Desktop/root`.
3. Install all requirements by running `py -m pip install -r requirements.txt`.

## Usage

1. Add any PDFs you want to process to the `/01_data` folder.
2. On OSX, run `python3 parse_table.py`. On Windows, run `py parse_table.py`.

In summary (remember to replace `./Desktop/root` with the actual path to where you downloaded the `root` folder):

```
(on OSX)
$ cd ./Desktop/root
$ python3 -m venv env
$ source env/bin/activate
$ pip3 install -r requirements.txt
$ python3 parse_table.py

(on Windows)
$ cd ./Desktop/root
$ py -m pip install -r requirements.txt
# py parse_table.py
```

## Results

The program reads any PDF files in the `01_data` folder.

It will also create one folder per PDF file in the `02_output` folder. Each folder will contain one CSV and one image per page in the PDF.
- Each image shows the detected table cells in red.
- Each CSV file is the parsed information from the table.

For example, if you put a 3-page PDF called `sample.pdf` in the `01_data` folder, the program will create 3 CSVs (`sample_page001.csv`, `sample_page002.csv`, `sample_page003.csv`) and 3 similarly-titled PNG images in the `02_output/sample` folder.

Lastly, two files are output for debugging purposes:
- `parse_table.log` is, as the name suggests, a log of everything printed to the console while running `parse_table.py`.
- `errors.csv` is exported to `02_output` if an error occurs while parsing any PDF document. It contains the document name, the page number, and the error message.


## Known issues

1. Currently cannot read cells with single numbers in them. This is a known issue with the underlying OCR library (`tesseract`).
2. Has only been tested with horizontally-merged cells; behavior is unclear if tables have vertically-merged cells (i.e. merged across several rows rather than columns).
3. Works best with cleanly-segmented tables. Tables with broken or jagged boundary lines will only have some cells detected (and thus read).
4. Reads tables by bounding lines only. Cannot currently distinguish between several rows if there is no horizontal line between them.


## File structure

A quick explanation of what each file or folder is:

```
root
├── 01_data/ - holds input PDF to be parsed
├── 02_output/ - where output CSVs are saved (created by running parse_table.py)
    └── errors.csv - lists any errors occured while running parse_table.py
├── documents/ - holds additional documents about this project
├── test-code/ - holds incomplete test code that may still be useful
├── parse_table.ipynb - jupyter notebook that walks through the OCR process.
├── parse_table.py - the primary script for this project
├── parse_table.log - logs the console output while running parse_table.py
├── parser.py - holds utility code used by parse_table.py
└── requirements.txt - the list of requirements for the project.
```
