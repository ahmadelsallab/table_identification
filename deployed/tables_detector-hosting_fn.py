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
    xml_file_name = parsed['xml_file']
    image_file_name = parsed['image_file']

    # xml_file_name = "Accident-Handwritten_V3.xml"
    # image_file_name = "Accident-Handwritten_V3.tiff"

    # getting data
    s3 = boto3.resource('s3', aws_access_key_id='', aws_secret_access_key='')  # new key
    xml_file_stream = s3.Object(bucket_name=bucket, key=xml_file_name)
    image_file_stream = s3.Object(bucket_name=bucket, key=image_file_name)

    # write from binary stream
    #xml_file_name = xml_file_name.split('/')[-1]
    xml_file_name = 'xml_file.xml'
    with open(xml_file_name, 'wb') as f:
        f.write(xml_file_stream.get()["Body"].read())

    # write from binary stream
    #image_file_name = image_file_name.split('/')[-1]
    image_file_name = 'image_file_name.tif'
    with open(image_file_name, 'wb') as f:
        f.write(image_file_stream.get()["Body"].read())
    
    #
    # # read from binary stream using gzip of pickle >> very high compression ratio
    # with gzip.open("temp.stream.zip", 'rb') as f:
    #     img_orig, xmlroot = pickle.load(f)  # here is the data

    
    img_orig = cv2.imread(image_file_name, 0)
    
    
    #img_orig = cv2.imread(image_file_stream.get()["Body"].read(), 0)
    #xmltree, xmlroot = read_xml(xml_file_stream.get()["Body"].read())

    # file_name = base_file_name + '.tiff'
    # img_orig = cv2.imread(os.path.join(dat_path, file_name), 0)
    # img_orig = cv2.imread(file_name, 0) from pickle
    img = (255 - copy.copy(img_orig)) / 255
    img = img[:, :-100]


    df_tabula = None  # If tables with lines, better use tabula

    if (df_tabula == None):
        verbose = False
        # 3508x2379 ~ 90linesx25words ~ pixels/word = 96, pixels/line=40--> strip_height > 40 (*2 for header usually > 2 lines) (100) w_max_pool < 96 (50)
        tables_detector = TableDetector(verbose, strip_height=50, w_max_pool=75, min_col_width=250, ratio_clip_max=0.25)
        tables = tables_detector.detect_tables(img)
        # tables_detector.visualize_tables(img_orig, file_name)
        tables_detector.fit_text_to_tables(xml_file_name)  # <<< pdf coordinates are compared to image coordinates (scaling needed)
        list_of_json_tables = tables_detector.get_tables_in_json()

    else:
        df_res = df_tabula
        list_of_json_tables = [json.loads(df_res.to_json())]
    # COORD    
    coordinates = []
    for table in tables:
        #json.loads(table.to_json().encode('utf-8').decode('utf-8'))
        coordinates.append({'x1': int(table['col_positions'][0]), 'y1':int(table['table_start']) , 'x2': int(table['col_positions'][-1]),'y2': int(table['table_end'])})
    
    response_body = json.dumps({"data": list_of_json_tables, 'coordinates' : coordinates})
    return response_body, output_content_type


print(sys.version)
