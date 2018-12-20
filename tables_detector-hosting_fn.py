import copy
import subprocess
import sys
import traceback

subprocess.call([sys.executable, '-m', 'pip', 'install', '-U', 'opencv-python'])
subprocess.call([sys.executable, '-m', 'pip', 'install', '-U', 'pdftabextract'])
subprocess.call([sys.executable, '-m', 'pip', 'install', '-U', 'tabula-py'])

import json
import os

import numpy as np

from pdftabextract.clustering import calc_cluster_centers_1d
from pdftabextract.clustering import find_clusters_1d_break_dist
from pdftabextract.clustering import zip_clusters_and_values
from pdftabextract.common import DIRECTION_VERTICAL
from pdftabextract.common import parse_pages
from pdftabextract.extract import fit_texts_into_grid, datatable_to_dataframe
from pdftabextract.extract import make_grid_from_positions
from pdftabextract.textboxes import border_positions_from_texts
from tables_detector.TableDetector import TableDetector

import boto3
import pickle
import gzip


# ------------------------------------------------------------ #
# Hosting methods                                              #
# ------------------------------------------------------------ #

def model_fn(model_dir):
    return None


def transform_fn(none_model, data, input_content_type, output_content_type):

    parsed = json.loads(data)
    print(type(parsed))
    bucket = parsed['bucket']
    file_name = parsed['file_name']

    s3 = boto3.resource('s3', aws_access_key_id='', aws_secret_access_key='')
    obj = s3.Object(bucket_name=bucket, key=file_name)

    with open("temp.stream.zip", 'wb') as f:
        f.write(obj.get()["Body"].read())

    with gzip.open("temp.stream.zip", 'rb') as f:
        img_orig, xmlroot = pickle.load(f)

    pdf_file_name = 'Accident-Handwritten_V3.pdf'
    base_file_name = os.path.splitext(pdf_file_name)[0]
    page_num = 3
    xml_file = base_file_name + '.xml'

    # file_name = base_file_name + '.tiff'
    # img_orig = cv2.imread(os.path.join(dat_path, file_name), 0)
    # img_orig = cv2.imread(file_name, 0) from pickle
    img = (255 - copy.copy(img_orig)) / 255
    img = img[:, :-100]

    print("image space(when PDFs are converted to images with given resolution)", img.shape)

    # _, xmlroot = read_xml(xml_file) from pickle

    pages = parse_pages(xmlroot)
    print("Text space from(PDF):", (pages[1]["height"], pages[1]["width"]))

    print("Scaling")
    page_scaling_hor = img.shape[1] / pages[1]['width']  # pages[1] : page text boxes coordinate system dimensions
    page_scaling_vert = img.shape[0] / pages[1]['height']  # pages[1] : page text boxes coordinate system dimensions

    print("page to image scaling", "VER: ", page_scaling_vert, "HOR:", page_scaling_hor)

    df_tabula = None  # If tables with lines, better use tabula

    if (df_tabula == None):
        verbose = False
        # 3508x2379 ~ 90linesx25words ~ pixels/word = 96, pixels/line=40--> strip_height > 40 (*2 for header usually > 2 lines) (100) w_max_pool < 96 (50)
        tables_detector = TableDetector(verbose, strip_height=50, w_max_pool=75, min_col_width=250, ratio_clip_max=0.25)
        tables = tables_detector.detect_tables(img)
        # tables_detector.visualize_tables(img_orig, file_name)
        tables_detector.fit_text_to_tables(xmlroot)  # <<< pdf coordinates are compared to image coordinates (scaling needed)
        list_of_json_tables = tables_detector.get_tables_in_json()

    else:
        df_res = df_tabula
        list_of_json_tables = [json.loads(df_res.to_json())]

    response_body = json.dumps({"data": list_of_json_tables})
    return response_body, output_content_type


print(sys.version)
