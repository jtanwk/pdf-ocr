# parser.py
#
# Description:
# This script defines the Document and Page classes that parse_table.py uses
# to streamline the flow of information through the script.

import itertools
import logging
import os
import statistics
from io import BytesIO

import cv2
import fitz
import jenkspy
import numpy as np
import pandas as pd
import pytesseract

# Initialize logger
logger = logging.getLogger('parse_table')
logger.setLevel(logging.DEBUG)


class Document:

    def __init__(self, doc_name, doc_dir, output_dir):

        # Initialize key attributes and filepaths
        self.doc_name = doc_name.replace('.pdf', '')
        self.doc_dir = doc_dir
        self.output_dir = output_dir
        self.pages = []
        self.page_dfs = []
        self.doc_data = None

        # Make folder for this document in /output_dir to hold output files
        os.mkdir(os.path.join(self.output_dir, self.doc_name))

    def split_pages(self):
        """
        1. Splits the input pdf into pages
        2. Writes a temporary image for each page to a byte buffer
        3. Loads the image as a numpy array using cv2.imread()
        4. Appends the page image/array to self.pages

        Notes:
        PyMuPDF's getPixmap() has a default output of 96dpi, while the desired
        resolution is 300dpi, hence the zoom factor of 300/96 = 3.125 ~ 3.
        """

        logger.debug("Splitting PDF into pages")

        doc = fitz.open(os.path.join(self.doc_dir, self.doc_name + ".pdf"))
        zoom_factor = 3
        for i in range(len(doc)):
            # Load page and get pixmap
            if doc.is_closed or doc.isEncrypted:
                logger.info("   PDF file is encrypted")
                return None
            page = doc.load_page(i)
            pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom_factor, zoom_factor))

            # Initialize bytes buffer and write PNG image to buffer
            buffer = BytesIO()
            buffer.write(pixmap.tobytes())
            buffer.seek(0)

            # Load image from buffer as array, append to self.pages, close buffer
            img_array = np.asarray(bytearray(buffer.read()), dtype=np.uint8)
            page_img = cv2.imdecode(img_array, 1)
            self.pages.append(page_img)
            buffer.close()

        return None

    def export_data(self, tables):
        """
        Exports each table in self.tables to separate CSV files in output_dir
        """

        logger.debug("Exporting data to CSV files...")

        for i, table_df in enumerate(tables):
            csv_filename = f"{self.doc_name}_table_{i + 1}.csv"
            table_df.reset_index(drop=True, inplace=True)
            # self.doc_data = pd.concat(self.page_dfs, ignore_index=True)
            table_df.to_csv(os.path.join(self.output_dir, self.doc_name, csv_filename), index=False)
            logger.info(f"    Exported table {i + 1} to {csv_filename}")

        logger.info(f"Completed exporting data from {self.doc_name} with no errors.")

    def merge_tables(self, page):
        tables = page.tables
        # Merge tables with the last table from self.page_dfs if any column matches
        if self.page_dfs:
            last_table = self.page_dfs[-1]
            for table_df in tables:
                if any(col in last_table.columns for col in table_df.columns):
                    merged_table = pd.concat([last_table, table_df], ignore_index=True)
                    self.page_dfs[-1] = merged_table
                else:
                    self.page_dfs.append(table_df)
        else:
            self.page_dfs.extend(tables)

    def parse_doc(self):

        # Split and convert pages to images
        self.split_pages()

        # Loop over images and parse each
        error_list = []
        for idx, i in enumerate(self.pages):

            logger.debug(f"Reading page {idx + 1} out of {len(self.pages)}")

            try:
                page = Page(i, idx + 1, self.doc_name, self.output_dir)
                if not page.check_for_table():
                    continue
                page.parse_page()

                self.merge_tables(page)

            except Exception as e:
                logger.info(f"    ERROR IN {self.doc_name}, page {idx + 1}: {str(e)}")

                # Append error dict to list that will be returned
                error_list.append({
                    'document': self.doc_name,
                    'page': str(idx + 1),
                    'error': str(e)
                })

        # Finally, concat all dataframes for the document and export one
        # consolidated CSV file
        if len(error_list) > 0:
            logger.info(f"    {self.doc_name} ran into errors while parsing.")
            return error_list
        elif not self.page_dfs:
            logger.info(f"    No table found in pdf")
            return None
        else:
            logger.info(f"    Completed parsing {self.doc_name} with no errors.")
            self.export_data(self.page_dfs)
            return None


class Page:

    def __init__(self, img, page_num, doc_name, output_dir):
        self.img = img
        self.page_num = page_num
        self.doc_name = doc_name
        self.output_dir = output_dir

        # Attributes to be assigned later
        self.img_gray = None
        self.table = None
        self.contours = []
        self.text_data = None

    def preprocess_image(self):
        """
        1. Converts input color image to grayscale
        2. Applies thresholding to self.img_gray
        3. Inverts the image
        4. Detects skew and auto-deskews self.img_gray
        """

        # Convert color image to grayscale
        img = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)

        # Binarize image using thresholding
        _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Invert image
        img = 255 - img

        # Deskew image
        # img_rot = cv2.dilate(img, np.ones((1, 3)), iterations=10)
        img_rot = cv2.erode(img, np.ones((1, 3)), iterations=50)
        cnt, _ = cv2.findContours(img_rot, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        angle_list = [x[-1] for x in list(map(cv2.minAreaRect, cnt)) \
                      if abs(x[-1]) != 0 and abs(x[-1]) != 90]

        try:
            angle = statistics.median(angle_list)
        except statistics.StatisticsError:
            angle = 0

        if angle < -45:
            angle += 90
        elif angle > 45:
            angle -= 90

        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img,
                             matrix,
                             (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)

        # Save results to object
        self.img_gray = img

        return self

    def detect_table(self):
        """
        1. Finds horizontal and vertical kernels in self.img_gray
        2. Combines them to get boxes
        """

        logger.debug("    Detecting lines on page")

        # Define kernels
        kernel_length = np.array(self.img_gray).shape[1] // 80
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
        sq_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # Detect horizontal and vertical lines from an image
        h_lines = cv2.erode(self.img_gray, h_kernel, iterations=3)
        h_lines = cv2.dilate(h_lines, h_kernel, iterations=4)
        v_lines = cv2.erode(self.img_gray, v_kernel, iterations=3)
        v_lines = cv2.dilate(v_lines, v_kernel, iterations=3)

        # Combine horizontal and vertical lines to form final detected table
        table = cv2.addWeighted(h_lines, 0.5, v_lines, 0.5, 0.0)
        table = cv2.erode(~table, sq_kernel, iterations=2)

        # Apply final thresholding
        _, table = cv2.threshold(table, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        self.table = table

        return self

    def get_contours(self):
        """
        1. Detects contours in the binary image stored in self.table
        2. Filters detected contours for quality (filters out lines, etc.)
        3. Appends final contours to self.contours
        """

        # Detect all contours
        contours, _ = cv2.findContours(self.table,
                                       cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # Filter out non-box contours
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w > 20 and h > 20 and w < (0.5 * self.img.shape[1]):
                self.contours.append(c)

        logger.debug(f"        {len(self.contours)} boxes detected")

        return self

    def check_for_table(self):
        """
        Checks if a table exists on the page.

        Returns:
        - True if a table exists, False otherwise.
        """

        # Preprocess image
        self.preprocess_image()

        # Detect table
        self.detect_table()
        self.get_contours()

        # If contours are found, consider it as a table
        if len(self.contours) > 0:
            return True
        else:
            return False

    def draw_contours(self):
        """
        Draws the detected contours onto the original image and exports it.
        """

        # Overlay contours on color image
        img_temp = cv2.cvtColor(self.img_gray, cv2.COLOR_GRAY2BGR)
        img_temp = 255 - img_temp
        cv2.drawContours(img_temp, self.contours, -1, (0, 0, 255), 3)

        # Save to file
        output_name = self.doc_name + "_page" + str(self.page_num) + ".png"
        cv2.imwrite(os.path.join(self.output_dir, self.doc_name, output_name),
                    img_temp)

        return self

    @staticmethod
    def read_text_from_coords(img, coords):
        """
        Takes the image in question, finds the bounding box given by the given
        coordinates, and returns text in the box.
        """
        x, y, w, h = coords

        # Crop image, then add 20px white border
        img_to_read = img[y:y + h, x:x + w]
        # img_to_read = np.pad(img_to_read, pad_width=20, mode='constant', constant_values=0)

        # Read text
        text = pytesseract.image_to_string(img_to_read, config='--psm 6')
        print(text, "text")
        return text

    def read_cells(self):
        """
        1. Constructs a dataframe from self.contours
        2. Extract coordinates of bounding rectangle for each contour
        3. Reads text within the given coordinates in self.img_gray
        """

        logger.debug("    Reading detected cells in table")

        # Get dataframe and coordinates of bounding boxes
        # df = pd.DataFrame(self.contours).rename({0: 'contours'}, axis=1)

        contours_array = [np.array(c, dtype=np.int32) for c in self.contours]

        # Calculate bounding rectangles for each contour
        bounding_rects = [cv2.boundingRect(contour) for contour in contours_array]

        # Create DataFrame with bounding rectangles and contours
        df = pd.DataFrame({'contours': contours_array, 'xywh': bounding_rects})

        # Read cells
        self.img_gray = 255 - self.img_gray
        df['text'] = df.apply(
            lambda x: Page.read_text_from_coords(self.img_gray, x['xywh']), axis=1
        )
        df = df.drop('contours', axis=1)

        # Save to object
        self.text_data = df

        return self

    @staticmethod
    def calculate_dimensions(table):
        """
        Given a dataframe with (x, y, w, h) data, uses large gaps in coordinates
        to estimate the number of rows and columns in the data.
        """

        # Add the last table

        # Calculate number of rows
        num_rows = 1
        table = table.sort_values(by=['y', 'x'])
        for i in range(len(table)):
            if i == 0:
                continue
            elif table.iloc[i]['y'] > (table.iloc[i - 1]['y'] + 0.8 * table.iloc[i - 1]['h']):
                num_rows += 1

        # Calculate number of columns
        num_cols = 1
        table = table.sort_values(by=['x', 'y'])
        for i in range(len(table)):
            if i == 0:
                continue
            elif table.iloc[i]['x'] > (table.iloc[i - 1]['x'] + 0.8 * table.iloc[i - 1]['w']):
                num_cols += 1
        return num_rows, num_cols

    @staticmethod
    def split_into_tables(df):
        # Find gaps between tables
        table_gaps = []
        df = df.sort_values(by=['y'])
        for i in range(len(df)):
            if i == 0:
                continue
            elif df.iloc[i]['y'] > (df.iloc[i - 1]['y'] + 1.5 * df.iloc[i - 1]['h']):
                table_gaps.append(i)

        tables = []
        start = 0
        for end in table_gaps:
            tables.append(df.iloc[start:end])
            start = end
        tables.append(df.iloc[start:])
        return tables

    def reconstruct_table(self):
        """
        1. Calculates number of rows, cols
        2. Use Jenks optimization to assign row and col numbers
        3. If necessary, adjusts column assignment if number of rows, cols
            doesn't match number of detected cells
        4. If there are merged cells, fill merged cells with 0 to complete table
        5. Pivor table into final format
        6. Export results to csv
        """

        logger.debug("    Parsing data into table...")
        tables = []

        # Split xywh into x, y, w, h
        df = self.text_data
        df[['x', 'y', 'w', 'h']] = pd.DataFrame(df['xywh'].tolist(), index=df.index)
        df = df.drop('xywh', axis=1)

        # Estimate number of rows and columns for each table
        tables_df = Page.split_into_tables(df)
        for index, table_df in enumerate(tables_df, 1):

            num_rows, num_cols = Page.calculate_dimensions(table_df)
            logger.debug(f"{num_rows} rows and {num_cols} columns detected.")

            # Use Jenks optimization to get natural breaks in y-coordinates, then
            # assign row number based on natural breaks
            row_breaks = jenkspy.jenks_breaks(table_df['y'], num_rows)
            calculate_row_num = lambda y: sum(list(map(lambda x: 1 if x < y else 0, row_breaks[1:])))
            table_df['row_num'] = table_df['y'].apply(calculate_row_num)

            # Repeat for columns
            col_breaks = jenkspy.jenks_breaks(table_df['x'], num_cols)
            calculate_col_num = lambda y: sum(list(map(lambda x: 1 if x < y else 0, col_breaks[1:])))
            table_df['col_num'] = table_df['x'].apply(calculate_col_num)

            # Recalculate column assignments if there are overlapping cell coords
            if num_rows * num_cols < len(table_df):
                while num_rows * num_cols < len(table_df):
                    num_cols += 1
                logger.debug(f"Adjusting estimate to {num_rows} rows and {num_cols} columns.")
                col_breaks = jenkspy.jenks_breaks(table_df['x'], num_cols)
                calculate_col_num = lambda y: sum(list(map(lambda x: 1 if x < y else 0, col_breaks[1:])))
                table_df['col_num'] = table_df['x'].apply(calculate_col_num)

            # Fill in missing columns if there are merged cells
            if num_rows * num_cols > len(table_df):
                table_df = table_df \
                    .set_index(['row_num', 'col_num']) \
                    .reindex(pd.MultiIndex.from_tuples(
                    set(itertools.product(table_df['row_num'], table_df['col_num'])))
                ) \
                    .reset_index() \
                    .rename({'level_0': 'row_num', 'level_1': 'col_num'}, axis=1) \
                    .sort_values(by=['row_num', 'col_num']) \
                    .fillna(0)
            else:
                table_df = table_df.sort_values(by=['row_num', 'col_num'])

            # Reshape using row and column assignments
            table_df = table_df.pivot(index='row_num', columns='col_num', values='text')
            table_df.columns = table_df.iloc[0]
            table_df = table_df.iloc[1:].reset_index(drop=True)

            tables.append(table_df)

        # Save tables to object
        self.tables = tables

        return self

    def export_data(self):
        """
        Exports the fully-shaped data in self.text_data to output_dir
        """

        output_name = self.doc_name + "_page" + str(self.page_num) + ".csv"
        logger.debug(f"    Writing table to file: {output_name}")

        self.text_data.to_csv(os.path.join(
            self.output_dir, self.doc_name, output_name
        ), index=False)

        return self

    def get_data(self):
        return self.tables

    def parse_page(self):

        # self.preprocess_image() \
        #     .detect_table() \
        #     .get_contours() \
        #     .draw_contours() \
        #     .read_cells() \
        #     .reconstruct_table()

        self.draw_contours() \
            .read_cells() \
            .reconstruct_table()
        # .export_data()

        return None
