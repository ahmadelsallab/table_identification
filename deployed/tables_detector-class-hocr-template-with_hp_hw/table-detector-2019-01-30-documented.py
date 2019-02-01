import subprocess
import sys

subprocess.call([sys.executable, '-m', 'pip', 'install', 'sagemaker==1.13.0'])
subprocess.call([sys.executable, '-m', 'pip', 'install', '-U', 'opencv-python'])
subprocess.call([sys.executable, '-m', 'pip', 'install', '-U', 'pdftabextract'])
subprocess.call([sys.executable, '-m', 'pip', 'install', '-U', 'tabula-py'])
subprocess.call([sys.executable, '-m', 'pip', 'install', '-U', 'lxml'])
subprocess.call([sys.executable, '-m', 'pip', 'install', '-U', 'pillow'])

subprocess.call([sys.executable, '-m', 'pip', 'install', '-U', 'intervaltree==2.1.0'])

subprocess.call([sys.executable, '-m', 'pip', 'install', '-U', 'xmltodict'])
import threading
import copy
import json
import os
import pickle
import re
import traceback
from collections import OrderedDict, defaultdict
from itertools import groupby
from string import Template

import boto3
import cv2
import numpy as np
import pandas as pd
import sagemaker
import xmltodict
from intervaltree import IntervalTree
from sagemaker import RealTimePredictor
from sagemaker.mxnet import MXNetPredictor
from sagemaker.predictor import json_serializer, json_deserializer

from pdftabextract.extract import make_grid_from_positions
from pdftabextract.geom import rectarea


def get_bbox(title):  # x1, y1, x2, y2
    # get the bounding box from the hocr string
    return [int(v) for v in re.findall(r'(?<=bbox )(.+?)(?=;|$)', title)[0].replace("bbox", "").strip().split()]


def make_sure_list(datastr):
    # list*ify* things :D
    if type(datastr) != list:
        datastr = [datastr]

    return datastr


class HocrDocument(object):
    # this is the class used to parse the HOCR file into the nested datastructure I use
    # the hierarchy is blocks>paragraphs>lines>words.. i use blocks for borderless detection
    # tesseract represents this hierarchy in blocks>paragraphs>lines>words with addition information like bounding box, text(for words)..etc
    # words are either empty or non empty string, for empty words they are used as vertical or horizontal lines according to the aspect ratio,
    # some words maybe empty but neither horizontal or even vertical.. horizontal and vertical lines are crucial for bordered detector which modifies tables

    # I represent the HOCR file in 2 formats (hierarchical or linear as ordered dictionaries)
    # hierarchical used for borderless
    # linear used for bordered

    def __init__(self, hocr_path):
        self.hocr_path = hocr_path
        self.parse_data = None
        self.hierarchy = None

    def parse(self):
        if self.parse_data is None:
            h_pages = self.hierarchical_parse  # uses the hierarchical_parse and then transforms it into an ordered dictionary
            linear_pages = OrderedDict()  # linear pages
            for pid, page in h_pages.items():
                sentences = []
                lines = []
                words = []
                vert = []
                hors = []
                empty = []
                # loop on blocks >> paragraphs >> lines >> words
                for block in page["blocks"].values():  # loop on blocks
                    for parag in block["paragraphs"].values():  # loop on paragraphs
                        for line in parag["lines"].values():  # loop on lines

                            line_sentences = []
                            for word_id, word in line["words"].items():
                                x1_i, y1_i, x2_i, y2_i = word["bbox"]
                                if x2_i - x1_i < .5 * page["width"] and y2_i - y1_i < .5 * page["height"]:
                                    # add dictionary for each word
                                    words.append({"width": x2_i - x1_i,
                                                  "height": y2_i - y1_i,
                                                  "value": word["value"],
                                                  "top": y1_i,
                                                  "left": x1_i, "bottom": y2_i,
                                                  "right": x2_i,
                                                  "topleft": np.array([x1_i, y1_i]),
                                                  "bottomleft": np.array([x1_i, y2_i]),
                                                  "topright": np.array([x2_i, y1_i]),
                                                  "bottomright": np.array([x2_i, y2_i]),
                                                  "line_code": (int(word_id.split("_")[1]), int(word_id.split("_")[2])),
                                                  "type": "tesser"
                                                  })

                            # words are [non words,horizontal lines,vertical lines,sentences,words]
                            # empty words
                            for non in line["non words"].values():
                                x1_i, y1_i, x2_i, y2_i = non["bbox"]
                                empty.append({"height": y2_i - y1_i, "width": x2_i - x1_i, "top": y1_i, "left": x1_i, "bottom": y2_i, "right": x2_i})

                            # horizontal separators
                            for hor in line["horizontals"].values():
                                x1_i, y1_i, x2_i, y2_i = hor["bbox"]
                                hors.append({"height": y2_i - y1_i, "width": x2_i - x1_i, "top": y1_i, "left": x1_i, "bottom": y2_i, "right": x2_i})

                            # verticals separators
                            for ver in line["verticals"].values():
                                x1_i, y1_i, x2_i, y2_i = ver["bbox"]
                                vert.append({"height": y2_i - y1_i, "width": x2_i - x1_i, "top": y1_i, "left": x1_i, "bottom": y2_i, "right": x2_i})

                            # sentences
                            for sent in line["sentences"].values():
                                x1_i, y1_i, x2_i, y2_i = sent["bbox"]
                                dict_sent = {"width": x2_i - x1_i,
                                             "height": y2_i - y1_i,
                                             "value": sent["value"],
                                             "top": y1_i,
                                             "left": x1_i, "bottom": y2_i,
                                             "right": x2_i,
                                             "topleft": np.array([x1_i, y1_i]),
                                             "bottomleft": np.array([x1_i, y2_i]),
                                             "topright": np.array([x2_i, y1_i]),
                                             "bottomright": np.array([x2_i, y2_i])
                                             }

                                sentences.append(dict_sent)
                                line_sentences.append(dict_sent)
                            lines.append(line_sentences)

                # add the entire page
                linear_pages[pid] = {'id': pid,
                                     'height': page["height"],
                                     'width': page["width"],
                                     'image': page["image"],

                                     "sentences": sentences,
                                     'words': words,
                                     'hor_lines': hors,
                                     'vert_lines': vert,
                                     'empty': empty,
                                     'lines': lines
                                     }
            self.parse_data = linear_pages

        return self.parse_data

    @property
    def hierarchical_parse(self):
        if self.hierarchy is None:
            def end_the_sentences(sent_temp_list, sent_h, word_h, line_area, line_bbox):
                sent_left, sent_top, sent_right, sent_bottom = np.inf, np.inf, -np.inf, -np.inf  # sentence defaults
                is_not_set = True
                sent_text = []
                for sent_word in sent_temp_list:
                    sent_word_bbox = word_h[sent_word]["bbox"]
                    sent_word_area = (sent_word_bbox[2] - sent_word_bbox[0]) * (sent_word_bbox[3] - sent_word_bbox[1])
                    # extending the bounding box, sometimes tesseract produces size 0 !!
                    if sent_word_area <= line_area:
                        is_not_set = False
                        sent_left, sent_top, sent_right, sent_bottom = min(sent_left, sent_word_bbox[0]), min(sent_top, sent_word_bbox[1]), max(sent_right, sent_word_bbox[2]), max(sent_bottom, sent_word_bbox[3])
                    sent_text.append(word_h[sent_word]["value"])

                if (sent_right - sent_left) > img_orig.shape[1] * .003 and (sent_bottom - sent_top) > img_orig.shape[0] * .0035:
                    # noinspection PyTypeChecker
                    sent_h[sent_id] = {"bbox": line_bbox if is_not_set else [sent_left, sent_top, sent_right, sent_bottom], "value": " ".join(sent_text)}

            pages_h = OrderedDict()  # pages_h means hierarchical pages representation
            for current_page in make_sure_list(xmltodict.parse(open(self.hocr_path, encoding="utf-8").read())["html"]["body"]["div"]):  # looping on pages

                # getting page information
                page_id = current_page["@id"]
                image_path = re.findall(r'(?<=image )(.+?)(?=;|$)', current_page["@title"])[0].replace("\"", "").replace("\'", "")  # finding image path from the parsed text
                image_size = get_bbox(current_page["@title"])[2:]  # in the processed files it should be.. [6198, 8770]

                blocks_h = OrderedDict()
                for block in make_sure_list(current_page["div"]):  # looping on blocks
                    # getting block information
                    block_id = block["@id"]
                    block_bbox = get_bbox(block["@title"])

                    parag_h = OrderedDict()

                    for parag in make_sure_list(block["p"]):
                        # getting paragraph information
                        parag_id = parag["@id"]
                        parag_bbox = get_bbox(parag["@title"])

                        line_h = OrderedDict()

                        for line in make_sure_list(parag["span"]):
                            # getting line information
                            line_id = line["@id"]
                            line_bbox = get_bbox(line["@title"])  # x1, y1, x2, y2
                            line_area = (line_bbox[2] - line_bbox[0]) * (line_bbox[3] - line_bbox[1])
                            y1, y2 = line_bbox[1], line_bbox[3]
                            word_h = OrderedDict()
                            e_word_h = OrderedDict()  # empty words.. has no text
                            vert_h = OrderedDict()  # has no text but it has h>>>>>w
                            hor_h = OrderedDict()  # has no text but it has w>>>>>h

                            sent_h = OrderedDict()
                            sent_temp_list = []
                            sent_id = 0
                            for word in make_sure_list(line["span"]):
                                # getting word information
                                word_id = word["@id"]

                                word_bbox = get_bbox(word["@title"])  # x1, y1, x2, y2
                                current_word_left = word_bbox[0]
                                w_i, h_i = word_bbox[2] - word_bbox[0], word_bbox[3] - word_bbox[1]

                                # finding text, maybe empty
                                if "#text" in word:
                                    word_text = word["#text"]
                                else:
                                    if type(word.get("strong")) != str and (word.get("strong") is None or word["strong"].get("em") is None):
                                        word_text = None
                                    else:
                                        if type(word.get("strong")) == str:
                                            word_text = word["strong"]
                                        else:
                                            word_text = word["strong"]["em"]

                                if word_text is None:  # none word
                                    # empty word maybe horizontal line,vertical line,empty bbox
                                    if w_i > h_i * 3:  # row separator.. horizontal line
                                        if h_i < 0.0085 * img_orig.shape[0] and w_i > 0.04 * img_orig.shape[1]:  # thresholds
                                            hor_h[word_id] = {"bbox": word_bbox}
                                        else:
                                            e_word_h[word_id] = {"bbox": word_bbox}
                                    elif h_i > w_i * 3:  # column separator ..vertical line
                                        vert_h[word_id] = {"bbox": word_bbox}
                                else:
                                    # those are actual words.. which will be fit into a grid to form sentences
                                    word_h[word_id] = {"bbox": word_bbox, "value": word_text}

                                    # this is construction of sentences, words are assumed to be sentences if the horizontal distance is as small as 1.5* height(the font used)
                                    if len(sent_temp_list) == 0:
                                        sent_temp_list.append(word_id)
                                    else:
                                        last_word_right = word_h[sent_temp_list[-1]]["bbox"][2]
                                        # noinspection PyTypeChecker
                                        if current_word_left - last_word_right > 1.5 * (y2 - y1):  # here the sentences being split
                                            # end of sentence
                                            end_the_sentences(sent_temp_list, sent_h, word_h, line_area, line_bbox)

                                            # reset for the new one
                                            sent_id += 1
                                            sent_temp_list = []

                                        sent_temp_list.append(word_id)

                            if sent_temp_list:
                                # end of last sentence.. to be added
                                end_the_sentences(sent_temp_list, sent_h, word_h, line_area, line_bbox)

                            # save this line
                            line_h[line_id] = {"bbox": line_bbox, "words": word_h, "non words": e_word_h, "horizontals": hor_h, "verticals": vert_h, "sentences": sent_h}

                        # save this paragraph
                        parag_h[parag_id] = {"bbox": parag_bbox, "lines": line_h}

                    # save this block
                    blocks_h[block_id] = {"bbox": block_bbox, "paragraphs": parag_h}

                # save this page
                pages_h[page_id] = {"image": os.path.split(image_path)[-1].replace("\"", ""), "blocks": blocks_h, "height": image_size[1], "width": image_size[0]}

                self.hierarchy = pages_h
        return self.hierarchy

    def add_from_deployed(self, page_id, json_input, predictor, is_new_api=False):
        # map tesseract space into localizer space
        self.parse()
        inputs_indexes_for_predictor = []  # here I save reference to words I update from predictor
        inputs_for_predictor = []

        is_HP = os.path.split(json_input['file_name'])[1].split(".")[0] == 'hand_printed'
        print("Downloading original pickle from localizer...")
        download_file(s3, json_input["bucket"], os.path.split(json_input['file_name'])[1], json_input["file_name"])

        localization_output = unpickle(os.path.split(json_input['file_name'])[1])  #
        localization_bboxes = list(map(lambda item: item[1], localization_output))

        for i, (x1_loc_i, y1_loc_i, w_loc_i, h_loc_i) in enumerate(localization_bboxes):
            x2_loc_i, y2_loc_i = x1_loc_i + w_loc_i, y1_loc_i + h_loc_i
            x1_loc_i, y1_loc_i, x2_loc_i, y2_loc_i = float(x1_loc_i), float(y1_loc_i), float(x2_loc_i), float(y2_loc_i)

            loc_bbox = np.array([[x1_loc_i, y1_loc_i], [x2_loc_i, y2_loc_i]])
            localization_found = False
            for words_index, word in enumerate(self.parse_data[page_id]["words"]):
                tesseract_bbox = np.array(
                    [[get_in_hor_loc_space(word["left"]), get_in_vert_loc_space(word["top"])],
                     [get_in_hor_loc_space(word["right"]), get_in_vert_loc_space(word["bottom"])]])

                if rectintersect(loc_bbox, tesseract_bbox) > .8 and word["height"] > 50 and word["width"] > 50:  # mostly, localization boxes are larger than needed
                    word["type"] = "HP" if is_HP else "HW"  # add it
                    inputs_indexes_for_predictor.append(words_index)
                    localization_found = True

            if not localization_found:
                # this must be in tesseract word space !!!!!!!!!!
                x1_loc_in_tess_i, y1_loc_in_tess_i, x2_loc_in_tess_i, y2_loc_in_tess_i = \
                    get_in_hor_tess_space(x1_loc_i), get_in_vert_tess_space(y1_loc_i), get_in_hor_tess_space(x2_loc_i), get_in_vert_tess_space(y2_loc_i)

                word = {"width": x2_loc_in_tess_i - x1_loc_in_tess_i,
                        "height": y2_loc_in_tess_i - y1_loc_in_tess_i,
                        "value": "tttt",
                        "top": y1_loc_in_tess_i,
                        "left": x1_loc_in_tess_i,
                        "bottom": y2_loc_in_tess_i,
                        "right": x2_loc_in_tess_i,
                        "topleft": np.array([x1_loc_in_tess_i, y1_loc_in_tess_i]),
                        "bottomleft": np.array([x1_loc_in_tess_i, y2_loc_in_tess_i]),
                        "topright": np.array([x2_loc_in_tess_i, y1_loc_in_tess_i]),
                        "bottomright": np.array([x2_loc_in_tess_i, y2_loc_in_tess_i]),
                        "line_code": (0, 0),
                        "type": "HP" if is_HP else "HW",
                        "confidence": .3
                        }
                if word["height"] > 50 and word["width"] > 50:
                    self.parse_data[page_id]["words"].append(word)
                    inputs_indexes_for_predictor.append(len(self.parse_data[page_id]["words"]) - 1)

        inputs_indexes_for_predictor = list(set(inputs_indexes_for_predictor))
        for input_index_for_predictor in inputs_indexes_for_predictor:
            # this must be in localization space !!!!!!!!!!
            input_words_for_predictor = self.parse_data[page_id]["words"][input_index_for_predictor]
            x_1, y_1, x_2, y_2 = input_words_for_predictor["left"], input_words_for_predictor["top"], input_words_for_predictor["right"], input_words_for_predictor["bottom"]
            x_1, y_1, x_2, y_2 = get_in_hor_loc_space(x_1), get_in_vert_loc_space(y_1), get_in_hor_loc_space(x_2), get_in_vert_loc_space(y_2)
            x_1, y_1, x_2, y_2 = int(x_1), int(y_1), int(x_2), int(y_2)

            inputs_for_predictor.append((img_for_predictors[y_1:y_2, x_1:x_2].tolist(), (x_1, y_1, x_2 - x_1, y_2 - y_1)))

        if is_new_api:
            new_inputs_for_predictor = []
            for image, bb in inputs_for_predictor:
                new_inputs_for_predictor.append({"bbox": {"left": bb[0], "top": bb[1], "height": bb[3], "width": bb[2]}, "lines": [image]})
            inputs_for_predictor = new_inputs_for_predictor

        print("Uploading refined pickle pickle from localizer to {}..".format(json_input["file_name"]))
        upload_file(s3, json_input["bucket"], pickle.dumps(inputs_for_predictor, protocol=2), json_input["file_name"])

        print("Calling the predictor..")
        try:
            json_predictions = predictor.predict(json_input)
        except Exception as ex:
            raise Exception()
        json_predictions = json_predictions["result"]
        print(json_predictions)

        if is_new_api:
            new_json_predictions = []
            for json_prediction in json_predictions:
                print("helloo")
                new_json_predictions.append({'text': json_prediction["lines"][0]["text"], 'score': json_prediction["lines"][0]["score"], 'type of text': json_prediction["lines"][0]["type of text"], 'y': json_prediction["bbox"]["top"], 'w': json_prediction["bbox"]['width'], 'x': json_prediction["bbox"]["left"], 'h': json_prediction["bbox"]["height"]})
            json_predictions = new_json_predictions
        for i, json_prediction in enumerate(json_predictions):
            word_index = inputs_indexes_for_predictor[i]
            x1_i, y1_i, x2_i, y2_i = get_in_hor_tess_space(json_prediction["x"]), get_in_vert_tess_space(json_prediction["y"]), \
                                     get_in_hor_tess_space(json_prediction["x"] + json_prediction["w"]), get_in_vert_tess_space(json_prediction["y"] + json_prediction["h"])

            x1_i, y1_i, x2_i, y2_i = float(x1_i), float(y1_i), float(x2_i), float(y2_i)
            # if json_prediction["score"] > .3:
            self.parse_data[page_id]["words"][word_index] = {"width": x2_i - x1_i,
                                                             "height": y2_i - y1_i,
                                                             "value": json_prediction["text"],
                                                             "top": y1_i,
                                                             "left": x1_i,
                                                             "bottom": y2_i,
                                                             "right": x2_i,
                                                             "topleft": np.array([x1_i, y1_i]),
                                                             "bottomleft": np.array([x1_i, y2_i]),
                                                             "topright": np.array([x2_i, y1_i]),
                                                             "bottomright": np.array([x2_i, y2_i]),
                                                             "line_code": (0, 0),
                                                             "type": json_prediction["type of text"],
                                                             "confidence": json_prediction["score"]
                                                             }
        print("Done:Added data from predictor..")

    def write_equivalent_xml(self):
        # this writes the xml of the given hocr(not yet needed),
        HEADER = """<?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE pdf2xml SYSTEM "pdf2xml.dtd">
        <pdf2xml producer="poppler" version="0.62.0">
        {PAGES}
        </pdf2xml>"""
        PAGE = Template(
            """<page number="1" top="0" left="0" height="${height}" width="${width}">
            <image top="0" left="0" width="${width}" height="${height}" src="${image}"/>
            ${TEXTS}
            </page>""")

        TEXT = """<text top="{top}" left="{left}" width="{width}" height="{height}" >{value}</text>"""
        data = self.parse()
        pages_words = []
        pages_sents = []

        for pid in data:
            page = data[pid]

            sents = []
            words = []
            for text in page["sentences"]:
                sents.append(TEXT.format(top=text["top"], left=text["left"], width=text["width"], height=text["height"], value=text["value"]))

            for text in page["words"]:
                words.append(TEXT.format(top=text["top"], left=text["left"], width=text["width"], height=text["height"], value=text["value"]))

            pages_sents.append(PAGE.safe_substitute({"width": page["width"], "height": page["height"], "image": page["image"], "TEXTS": "\n".join(sents)}))
            pages_words.append(PAGE.safe_substitute({"width": page["width"], "height": page["height"], "image": page["image"], "TEXTS": "\n".join(words)}))

        xml_words = HEADER.format(PAGES="\n".join(pages_words))
        open("test.xml", "w").write(xml_words)


page_scaling_vert = None
page_scaling_hor = None

xmlroot = None


def get_vertical_separators(line):
    # given a line with list of bounding boxes in the form W WW WWWW W WW W
    # this will get the the separators W|WW|WWWW|W|WW|W locations
    # used by borderless detector to find tables forming a grid
    vertical_separators = []
    for index in range(len(line) + 1):

        if index == 0:
            separator = line[index]["left"] / 2
        elif index == len(line):
            separator = (line[index - 1]["right"] + img_orig.shape[1]) / 2
        else:
            separator = (line[index - 1]["left"] + line[index]["right"]) / 2
        vertical_separators.append(separator)

    return vertical_separators


def is_alignment_suitable(line, current_separators):
    # this compares the alignment of two lines for example
    ########################################################
    # a suitable alignment: maybe good candidate
    ########################################################
    #
    #    WW W WWW W WW W
    #    WW W WWW W WW W
    ########################################################
    # a bad alignment can't be in the same table
    ########################################################
    #
    #    WW W WWW W WW W
    #    WWWWWW WWWWWWWW
    ########################################################
    for index in range(len(line) + 1):
        if index == 0:  # left
            if current_separators[index] >= line[index]["left"]:
                return False
        elif index == len(line):  # right
            if line[index - 1]["right"] >= current_separators[index]:
                return False
        else:  # mid
            if not line[index - 1]["right"] < current_separators[index] < line[index]["left"]:
                return False

    return True


class TableDetector:
    def __init__(self, verbose=False, strip_height=50, w_max_pool=50, min_col_width=50, ratio_clip_max=0.8):
        self.verbose = verbose
        self.state = 'Table_Search'
        self.state_machine = {'Table_Search': self.table_search,
                              'Candidate_Table': self.candidate_table,
                              'Table_Registered': self.table_registered,
                              'Confirm_Table_End': self.confirm_table_end}
        self.tables = []
        self.tables_df = []
        self.reset_table_info()
        self.strip_height = strip_height
        self.w_max_pool = w_max_pool
        self.min_col_width = min_col_width
        self.ratio_clip_max = ratio_clip_max

    def reset_table_info(self):
        self.table_info = {}
        self.col_positions = []

    def table_search(self):
        '''
        if(self.verbose):
            print('Table_Search')
        '''
        if len(self.col_positions) > 0:
            self.state = 'Candidate_Table'
            self.table_info['table_start'] = self.start
            self.table_info['col_positions'] = self.col_positions
            self.n_cols = len(self.col_positions)

    def candidate_table(self):
        '''
        if(self.verbose):
            print('Candidate_Table')
        '''
        if len(self.col_positions) > 0:  # and len(self.col_positions) == self.n_cols:
            self.state = 'Table_Registered'
        else:
            self.state = 'Table_Search'
            self.reset_table_info()

    def table_registered(self):
        '''
        if(self.verbose):
            print('Table_Registered')
        '''
        if len(self.col_positions) == 0:  # or len(self.col_positions) != self.n_cols:
            # if len(self.col_positions) == 0:
            self.table_info['table_end'] = self.start
            self.state = 'Confirm_Table_End'

    def confirm_table_end(self):
        '''
        if(self.verbose):
            print('Confirm_Table_End')
        '''
        if len(self.col_positions) == 0:  # or len(self.col_positions) != self.n_cols:
            # if len(self.col_positions) != self.n_cols:
            self.state = 'Table_Search'

            self.tables.append(self.table_info)
            self.reset_table_info()
        else:
            self.state = 'Table_Registered'

    def remove_false_cols(self, grads):
        # self.min_col_width = 50
        # Get all posititions of col starts

        col_starts = np.squeeze(np.argwhere(grads == 1))
        # If dist between 2 1's < min_col_width--> set all to zeros until next 1 pos
        prev_pos = col_starts[0]
        for idx, pos in enumerate(col_starts):
            if idx > 0:
                dist = pos - prev_pos

                if dist < self.min_col_width:
                    grads[prev_pos] = 0
                # else:
                prev_pos = pos

        return grads

    def remove_outliers(self, arr):
        arr[np.abs(arr - np.mean(arr)) > (3 * np.std(arr))] = 0
        return arr

    def remove_consecutive(self, input):
        # Get 1's pos
        ones = np.squeeze(np.argwhere(input == 1))
        # Get -1's pos
        neg_ones = np.squeeze(np.argwhere(input == -1))
        # Alternate from 1's and -1's. Always start by 1's
        result = []
        positive = True
        next_pos = 0
        for i in range(len(ones)):
            if positive:
                result.append(ones[next_pos])
                positive = False
                curr_pos = next_pos

                next_pos = np.squeeze(np.argwhere(neg_ones > ones[curr_pos])).tolist()  # Alternate to the pos in neg_ones > curr_pos value in ones
                if isinstance(next_pos, list):
                    if len(next_pos) > 0:
                        next_pos = next_pos[0]
                    else:
                        break
                else:
                    next_pos = next_pos
            else:
                result.append(neg_ones[next_pos])
                positive = True
                curr_pos = next_pos

                next_pos = np.squeeze(np.argwhere(ones > neg_ones[curr_pos])).tolist()
                if isinstance(next_pos, list):
                    if len(next_pos) > 0:
                        next_pos = next_pos[0]
                    else:
                        break
                else:
                    next_pos = next_pos

        mask = np.zeros(len(input), dtype=int)
        if (len(result) > 0):
            mask[np.array(result)] = input[np.array(result)]

        return mask

    def maxpool1D(self, h, w):
        max_thresh = 3
        n_w = int(len(h) / w)
        h_maxes = np.zeros(len(h))
        for i in range(n_w):

            # h_maxes[i*w:(i+1)*w] = np.mean(h[i*w:(i+1)*w])
            # h_maxes[i*w:(i+1)*w] = max(h[i*w:(i+1)*w])

            local_max = max(h[i * w:(i + 1) * w])
            if local_max > max_thresh:
                h_maxes[i * w:(i + 1) * w] = local_max
            else:
                h_maxes[i * w:(i + 1) * w] = 0

        return h_maxes

    def zero_crossings(self, h_maxes):
        res = 0 * h_maxes

        pos = h_maxes > 0
        pos2neg_positions = (~pos[:-1] & pos[1:]).nonzero()[0]
        neg2pos_positions = (pos[:-1] & ~pos[1:]).nonzero()[0]
        res[pos2neg_positions] = 1
        res[neg2pos_positions] = -1
        return res

    def clean_grads(self, grads):
        # TODO: outliers removal
        # grads[np.abs(grads-np.mean(grads)) > (3*np.std(grads))] = 0
        # grads = self.remove_outliers(grads)

        # Adaptive threshold = max*ratio
        thresh = np.max(grads) * self.ratio_clip_max
        global_thresh = 2

        # grads[grads < global_thresh] = 0

        # Filter pos values
        filter_pos_idx = np.squeeze(np.argwhere(np.logical_and((grads > 0), (grads <= thresh))))
        grads[filter_pos_idx] = 0

        # Filter neg values
        filter_neg_idx = np.squeeze(np.argwhere(np.logical_and((grads < 0), (grads >= -thresh))))
        grads[filter_neg_idx] = 0

        # Normalize thr grads to 1/-1
        grads[grads < 0] = -1
        grads[grads > 0] = 1

        # Remove consecutive 1's or -1's
        grads = self.remove_consecutive(grads)

        return grads

    def preprocess_row(self, img_strip):

        # 1. Histo projection on columns
        h_all = np.sum(img_strip, axis=0)
        h_all = self.remove_outliers(h_all)
        # 2. Maxpool1D
        h_maxes = self.maxpool1D(h_all, w=self.w_max_pool)
        # Clip small maxes
        h_maxes[h_maxes <= max(h_maxes) * self.ratio_clip_max] = 0

        # h_maxes = self.remove_outliers(h_maxes)
        # 3. Gradients
        '''
        g = np.gradient(h_maxes)
        # 4. Clean grads
        g_clean = self.clean_grads(g.copy())   
        '''
        g_clean = self.zero_crossings(copy.copy(h_maxes))
        # g_clean = g
        if self.verbose:
            self.h = copy.copy(h_maxes)
            # self.g = g_clean.copy()
            self.g = copy.copy(g_clean)

        return g_clean

    def check_row_pattern(self, row_grads):
        # self.g = 0*row_grads.copy()
        if len(np.argwhere(row_grads == 1)) >= 2:  # +v crossing

            # If dist between 2 1's < min_col_width--> set all to zeros
            row_grads = self.remove_false_cols(row_grads)
            # self.g = row_grads.copy()
            # Count 1's => 2
            # col_positions = np.squeeze(np.argwhere(row_grads == 1))  # row_grads are now clean
            col_positions = list(np.argwhere(row_grads == 1)[:, 0])
        else:
            col_positions = []

        return col_positions

    def row_pattern_detect(self, img_strip):
        g_clean = self.preprocess_row(img_strip)
        col_positions = self.check_row_pattern(g_clean)  # clear noisy columns

        self.col_positions = copy.copy(col_positions)

        return col_positions

    def adjust_tables_boundaries(self, img):
        H, W = img.shape

        for table in self.tables:
            # table['table_start'] = table['table_start'] - self.strip_height
            # table['table_end'] = table['table_end'] - self.strip_height
            last_col = W - table['col_positions'][0]
            table['col_positions'] = np.append(table['col_positions'], last_col)

        return self.tables

    def detect_tables(self, img):

        H, W = img.shape
        n_strips = int(np.floor(H / self.strip_height))
        # overlap = 0.5

        for i in range(n_strips):
            self.start = i * self.strip_height
            self.end = self.start + self.strip_height
            img_strip = img[self.start:self.end, :]
            col_positions = self.row_pattern_detect(img_strip)
            self.state_machine[self.state]()

        self.tables = self.adjust_tables_boundaries(img)

        return self.tables

    def layout_based_borderless_detection(self):
        """
        i generate a mask of the bounding boxes then looping on lines.. as long as lines are consistent into a table they are grouped into a borderless table
        """

        # make the mask
        mask = np.zeros_like(img_orig)
        all_tables = set()

        # add bounding boxes to interval trees for faster search
        sentences_tree = interval_format(page["sentences"])
        hor_tree = interval_format(page["hor_lines"])
        vert_tree = interval_format(page["vert_lines"])
        empt_tree = interval_format(page["empty"])

        # find candidate blocks
        for block in page_h["blocks"].values():  # loop on blocks(I assume a table is grouped into a block, it works well for clear tables but there are some false ones)
            ignore_block = True

            # blocks are bad if they have one sentence-lines, one sentence can't be used as table, ex
            #
            #   WWWWWWWWWWWWWWWWWWWWWWW
            #   WWWWWWWWWWWWWWWWWWWWWWW
            #     WWWWWWWWWWWWWWWWWWWWWWW
            # WWWWWWWWWWWWWWWWWWWWWWW

            for par in block["paragraphs"].values():
                for line in par["lines"].values():
                    if len(line["sentences"]) > 1:
                        ignore_block = False

            # not filtered in the multi-sentence check
            if not ignore_block:
                x1, y1, x2, y2 = block["bbox"]

                x1, y1, x2, y2 = .95 * x1, .95 * y1, 1.05 * x2, 1.05 * y2
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # search for the sentences in the block bounding box using interval search
                candidate_statements = list(filter(lambda candidate: page["sentences"][candidate]["width"] < .65 * (x2 - x1), _2d_search(x1, x2, y1, y2, tree=sentences_tree)))

                # also find the non words in the same block bounding box
                candidate_hors = _2d_search(x1, x2, y1, y2, tree=hor_tree, is_strict=False)
                candidate_vert = _2d_search(x1, x2, y1, y2, tree=vert_tree, is_strict=False)
                candidate_empt = _2d_search(x1, x2, y1, y2, tree=empt_tree, is_strict=False)

                for candidate in candidate_statements:
                    sent = page["sentences"][candidate]
                    mask[sent["top"]:sent["bottom"], sent["left"]:sent["right"]] = 1.0

                    # remove non words fully inside sentences
                    candidate_hors -= _2d_search(sent["left"], sent["right"], sent["top"], sent["bottom"], tree=hor_tree)
                    candidate_vert -= _2d_search(sent["left"], sent["right"], sent["top"], sent["bottom"], tree=vert_tree)
                    candidate_empt -= _2d_search(sent["left"], sent["right"], sent["top"], sent["bottom"], tree=empt_tree)

                # get columns and rows using the projection of the mask, bounding boxes ranges will be detected
                cols = intervalize(np.sum(mask[y1:y2, x1:x2], axis=0), img_orig.shape[1] * .01, x1)
                cols = [x1] + cols + [x2]
                rows = intervalize(np.sum(mask[y1:y2, x1:x2], axis=1), img_orig.shape[0] * .008, y1)
                rows = [y1] + rows + [y2]

                # make grid to fit words
                cells = make_grid_from_positions(cols, rows)

                reject = False
                table_structure = []
                number_of_texts = 0
                for row, cells_row in enumerate(cells):
                    table_structure.append([])
                    for col, cell_bbox in enumerate(cells_row):
                        left, top, right, bottom = cell_bbox[0, 0], cell_bbox[0, 1], cell_bbox[1, 0], cell_bbox[1, 1]
                        cell_text_list = [page["sentences"][candidate] for candidate in _2d_search(left, right, top, bottom, tree=sentences_tree)]  # get words in the cell

                        lefts = [text["left"] for text in cell_text_list]  # get the left point
                        if lefts:
                            shift = min(lefts)
                        else:
                            shift = 0
                        groups = defaultdict(list)
                        for key, group in groupby(cell_text_list, lambda x: int(round((x["left"] - shift) / 25))):  # group things in the same column
                            for thing in group:
                                groups[key].append(thing)

                        if len(groups) > 2 or len(candidate_hors) + len(candidate_vert) + len(candidate_empt) > .15 * len(candidate_statements):  # we expect more than 2 columns or at least good distribution of words
                            reject = True
                        number_of_texts += len(groups)
                        table_structure[row].append("".join([text["value"] for text in cell_text_list]))

                for candidate in candidate_hors:
                    if page["hor_lines"][candidate]["width"] > .2 * (x2 - x1):  # this maybe bordered table, we expect no separating lines in borderless tables so we reject this
                        reject = True
                        break

                for candidate in candidate_vert:
                    if page["vert_lines"][candidate]["height"] > .2 * (y2 - y1):  # this maybe bordered table, we expect no separating lines in borderless tables so we reject this
                        reject = True
                        break

                if not reject:
                    # clean noisy columns/rows
                    remove = True
                    for col in table_structure[0]:  # empty row? remove it
                        if col != "":
                            remove = False

                    if remove:
                        # remove bad row
                        rows = [rows[0]] + rows[2:]
                        table_structure = table_structure[1:]

                    remove = True
                    for col in table_structure[-1]:  # empty row? remove it
                        if col != "":
                            remove = False

                    if remove:
                        # remove bad row
                        rows = rows[:-2] + [rows[-1]]
                        table_structure = table_structure[:-1]

                    remove = True
                    for row in table_structure:  # empty column? remove it
                        if row[0] != "":
                            remove = False

                    if remove:
                        # remove bad column
                        cols = cols[:-2] + [cols[-1]]
                        for i in range(len(table_structure)):
                            table_structure[i] = table_structure[i][0:]

                    remove = True
                    for row in table_structure:  # empty column? remove it
                        if row[-1] != "":
                            remove = False

                    if remove:
                        # remove bad column
                        cols = cols[:-2] + [cols[-1]]
                        for i in range(len(table_structure)):
                            table_structure[i] = table_structure[i][:-1]

                    # make the data frame
                    pd_table = pd.DataFrame(table_structure)

                    # reject (1,1) table, reject tables with large number of words per cell,reject tables with small number of words per cell,and check the width to be as large as the half of the page
                    reject = pd_table.shape == (1, 1) or \
                             number_of_texts > 2 * pd_table.shape[0] * pd_table.shape[1] or \
                             2 * number_of_texts < pd_table.shape[0] * pd_table.shape[1] or \
                             cols[-1] - cols[0] < .5 * page["width"]

                    if not reject:
                        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                            #print(len(candidate_hors) + len(candidate_vert) + len(candidate_empt), len(candidate_statements))
                            #print(pd_table, pd_table.shape)
                            if str(pd_table) not in all_tables:
                                self.tables_df.append(pd_table)
                                self.tables.append({"col_positions": cols, "table_start": rows[0], "table_end": rows[-1], "modified_left": cols[0], "modified_top": rows[0], "modified_right": cols[-1], "modified_bottom": rows[-1]})
                                all_tables.add(str(pd_table))

    def fit_bordered_tables(self):
        """
        this fits text to the candidate tables from the state machine of the table detection
        """
        self.tables_df = []
        tables_with_text = []
        all_tables = set()

        for table_index, table in enumerate(self.tables):

            # they may not be sorted
            table['col_positions'] = sorted(table['col_positions'])

            try:

                initial_cols, initial_rows, words = [], [], []

                table_top, table_bottom, table_left, table_right = table["table_start"], table["table_end"], table["col_positions"][0], table["col_positions"][-1]  # get the bounding box

                #################### GRAPHICAL DEBUGGING ###################
                # this shows the initial tables from the state machine
                # print(table_index)
                # cv2.namedWindow("original", cv2.WINDOW_NORMAL)
                # cv2.imshow("original", img_orig_words[table_top:table_bottom, table_left:table_right])
                # cv2.waitKey()
                # ##################################

                table_height, table_width = table_bottom - table_top, table_right - table_left
                table_bbox = np.array([[table_left, table_top], [table_right, table_bottom]])  # make 2d point

                # find horizontal lines in the table
                for hor in page['hor_lines']:
                    left_i, right_i, top_i, bottom_i, width_i, height_i = int(hor["left"]), int(hor["right"]), int(hor["top"]), int(hor["bottom"]), hor["width"], hor["height"]
                    hor_bbox = np.array([[left_i, top_i], [right_i, bottom_i]])

                    if rectintersect(table_bbox, hor_bbox) > .3 and width_i > .25 * page["width"]:
                        initial_rows.append((left_i, right_i, (bottom_i + top_i) / 2))

                # find vertical lines in the table
                for vert in page["vert_lines"]:
                    left_i, right_i, top_i, bottom_i, width_i, height_i = int(vert["left"]), int(vert["right"]), int(vert["top"]), int(vert["bottom"]), vert["width"], vert["height"]
                    vert_bbox = np.array([[left_i, top_i], [right_i, bottom_i]])

                    if rectintersect(table_bbox, vert_bbox) > .31 and height_i > .05 * page["height"]:
                        initial_cols.append((top_i, bottom_i, (left_i + right_i) / 2))


                if initial_cols and initial_rows:
                    # we have some initiall rows and columns, let's extend them and produce the modified table bounding box using the maximum width and height of those lines as reference
                    table["modified_top"] = modified_top = int(np.min(list(map(lambda item: item[0], initial_cols))))
                    table["modified_bottom"] = modified_bottom = int(np.max(list(map(lambda item: item[1], initial_cols))))
                    table["modified_left"] = modified_left = int(np.min(list(map(lambda item: item[0], initial_rows))))
                    table["modified_right"] = modified_right = int(np.max(list(map(lambda item: item[1], initial_rows))))
                    table_bbox = np.array([[modified_left, modified_top], [modified_right, modified_bottom]])

                    cols, rows = [], []
                    # find the new horizontal lines in the table after the modification
                    for hor in page['hor_lines']:
                        left_i, right_i, top_i, bottom_i, width_i, height_i = int(hor["left"]), int(hor["right"]), int(hor["top"]), int(hor["bottom"]), hor["width"], hor["height"]
                        hor_bbox = np.array([[left_i, top_i], [right_i, bottom_i]])

                        if rectintersect(table_bbox, hor_bbox) > .3 and width_i > .25 * page["width"]:
                            rows.append((left_i, right_i, (bottom_i + top_i) / 2))

                    # find the new vertical lines in the table after the modification
                    for vert in page["vert_lines"]:
                        left_i, right_i, top_i, bottom_i, width_i, height_i = int(vert["left"]), int(vert["right"]), int(vert["top"]), int(vert["bottom"]), vert["width"], vert["height"]
                        vert_bbox = np.array([[left_i, top_i], [right_i, bottom_i]])

                        if rectintersect(table_bbox, vert_bbox) > .31 and height_i > .05 * page["height"]:
                            cols.append((top_i, bottom_i, (left_i + right_i) / 2))

                    # find the new words in the table after the modification
                    for word in page["words"]:
                        left_i, right_i, top_i, bottom_i = int(word["left"]), int(word["right"]), int(word["top"]), int(word["bottom"])
                        word_bbox = np.array([[left_i, top_i], [right_i, bottom_i]])
                        if rectintersect(table_bbox, word_bbox) > .5:
                            words.append(word)

                    # sorting them
                    rows = sorted(list(map(lambda item: int(item[2]), rows)))
                    cols = sorted(list(map(lambda item: int(item[2]), cols)))

                    # some times we may have duplicate columns this will cluster them on 1D
                    cols_new = []
                    for i in range(len(cols)):
                        if not cols_new or abs(cols[i] - cols[i - 1]) > 40:
                            cols_new.append(cols[i])
                        else:
                            cols_new[-1] = (cols[i] + cols[i - 1]) // 2
                    cols = cols_new

                    # some times we may have duplicate rows this will cluster them on 1D
                    rows_new = []
                    for i in range(len(rows)):
                        if not rows_new or abs(rows[i] - rows[i - 1]) > 40:
                            rows_new.append(rows[i])
                        else:
                            rows_new[-1] = (rows[i] + rows[i - 1]) // 2
                    rows = rows_new


                    # add the very left,right,bottom,top of the modified table if needed (if you remove this you may miss the first or last row/column :) )
                    if modified_top < min(rows) and min(rows) - modified_top > table_height * .1:
                        rows = [modified_top] + rows
                    if max(rows) < modified_bottom and modified_bottom - max(rows) > table_height * .1:
                        rows = rows + [modified_bottom]

                    if modified_left < min(cols) and min(cols) - modified_left > table_width * .1:
                        cols = [modified_left] + cols
                    if max(cols) < modified_right and modified_right - max(cols) > table_width * .1:
                        cols = cols + [modified_right]

                    if len(cols) > 2 and len(rows) > 2:
                        # for more than 2,2 columns/rows
                        cells = make_grid_from_positions(cols, rows)

                        # place the words in the cells according to the percentage of intersection
                        # initialize the best locations
                        best_word_locations = []
                        for _ in words:
                            best_word_locations.append((-1, -1, 0))  # row, col, score

                        # this will hold the table, where every word is placed in the cell with maximum intersection(this will handle words contained in more than one cell)
                        table_structure = []
                        for row, cells_row in enumerate(cells):
                            table_structure.append([])
                            for col, cell_bbox in enumerate(cells_row):
                                table_structure[row].append([])

                                for word_index, word in enumerate(words):
                                    word_bbox = np.array([[word["left"], word["top"]], [word["right"], word["bottom"]]])
                                    inter = rectintersect(cell_bbox, word_bbox)  # amount of intersection
                                    best_score = best_word_locations[word_index][2]  # best previous score
                                    if inter > best_score:
                                        best_word_locations[word_index] = (row, col, inter)  # assign the new best location

                        # place the words into the grid cell, then sort them according to the codes given by tesseract, these codes are already sorted on(x,y) plane as we read english
                        for word_index, word in enumerate(words):
                            row, col, _ = best_word_locations[word_index]
                            if row != -1:
                                table_structure[row][col].append(words[word_index])  # append to the list of words in the i,j cell

                        for row, cells_row in enumerate(cells):
                            for col, cell_bbox in enumerate(cells_row):
                                # sort the words according to the code, the (x,y) location and concatenate the string
                                table_structure[row][col] = " ".join(map(lambda item: item["value"], sorted(table_structure[row][col], key=lambda item: (item["line_code"][0], item["line_code"][1], item["left"], item["right"]))))

                        df = pd.DataFrame(table_structure)  # make the dataframe

                        # Rename columns of the dataframe
                        new_names = []
                        for j in range(len(df.columns)):
                            new_names.append(str(j))

                        df.columns = new_names

                        if self.post_process_tables(df.copy()) and str(df) not in all_tables:
                            # this will only accept new tables(not repeated in all_tables), the table has to have one of the templates defined in post_process_tables
                            self.tables_df.append(df)
                            tables_with_text.append(table)
                            all_tables.add(str(df))  # register the table to the set of tables
                            print("Table detected:", table_index, "with shape", df.shape)

                    # print(df, "\n", self.post_process_tables(df.copy()))
                    # print("> page: grid with %d rows, %d columns" % (n_rows, n_cols))
            except Exception as e:

                traceback.print_tb(e.__traceback__)
                print(table_index)

        # print('=' * 100)
        self.tables = tables_with_text  # COORD

    def post_process_tables(self, table_df):
        """
        this checks if the given table dataframe follow the templates we define in template_header_col, you can add more templates and repeat the same loop below
        """
        template_header_col = {'0': ['Dates', 'Service', 'Confinement'],
                               '1': ['Diagnosis', 'Code', 'ICD', '(ICD)'],
                               '2': ['Diagnosis', 'Description'],
                               '3': ['Procedure', 'Code'],
                               '4': ['Procedure', 'Description'],
                               }
        header_limit = 5

        # check if it follows the template
        result = False
        for col in table_df.columns[:5]:
            # col_data = table_df[col]
            template_header = template_header_col[col]
            for row in range(min(header_limit, len(table_df[col]))):
                for header in template_header:
                    if header in table_df[col][row]:
                        result = True
        return result

    def get_json_response(self):
        """
        get the required json format, which includes the table represented as i,j format, with the bounding box information and the modified bounding box information
        """
        tables_structure = []
        for i, table_df in enumerate(self.tables_df):
            table_structure = []
            for row_index, row_series in table_df.iterrows():
                for col_index, data_item in enumerate(row_series):
                    table_structure.append({"r": row_index, "c": col_index, "value": data_item})
            tables_structure.append({"table": i,
                                     # COORD
                                     "coordinates": [{'x1': int(self.tables[i]['col_positions'][0]), 'y1': int(self.tables[i]['table_start']), 'x2': int(self.tables[i]['col_positions'][-1]), 'y2': int(self.tables[i]['table_end'])}]
                                        , "modified coordinates": [{'x1': int(self.tables[i]["modified_left"]), 'y1': int(self.tables[i]["modified_top"]), 'x2': int(self.tables[i]["modified_right"]),
                                                                    'y2': int(self.tables[i]["modified_bottom"])}]
                                        , "row_count": table_df.shape[0], "col_count": table_df.shape[1], "data": table_structure})

        response_body = json.dumps(tables_structure)

        return response_body


#########################################################################################################################################
# we need to transform between spaces by scaling, they have different spaces,(localization space = 3508*2479) (tesseract space= 7948*6198)
#########################################################################################################################################
def get_in_hor_tess_space(value_from_loc_hor):
    return value_from_loc_hor * tess_scaling_hor


def get_in_vert_tess_space(value_from_loc_vert):
    return value_from_loc_vert * tess_scaling_vert


def get_in_hor_loc_space(value_from_tess_hor):
    return value_from_tess_hor / tess_scaling_hor


def get_in_vert_loc_space(value_from_tess_vert):
    return value_from_tess_vert / tess_scaling_vert


#########################################################################################################################################

def rectintersect(a, b):
    # computes percentage of intersection between two bounding boxes, normalizes with respect to the second
    # ex: if b is contained in a you will get 1

    a_left, a_top, a_right, a_bottom = a[0, 0], a[0, 1], a[1, 0], a[1, 1]
    b_left, b_top, b_right, b_bottom = b[0, 0], b[0, 1], b[1, 0], b[1, 1]

    # disjoint
    if a_right <= b_left or b_right <= a_left or b_bottom <= a_top or a_bottom <= b_top:
        return 0

    intersection_box = np.array([[max(a[0, 0], b[0, 0]), max(a[0, 1], b[0, 1])], [min(a[1, 0], b[1, 0]), min(a[1, 1], b[1, 1])]])

    return rectarea(intersection_box) / rectarea(b)


def interval_format(list_of_dict):
    # make list of things in interval tree format, logarithmic 2d search :)
    tree_hor = IntervalTree()
    tree_ver = IntervalTree()

    for i, bbox in enumerate(list_of_dict):
        tree_hor[bbox["left"]:bbox["right"]] = i
        tree_ver[bbox["top"]:bbox["bottom"]] = i

    return tree_hor, tree_ver


def upload_file(s3, bucket, bin_data, s3_path_to_file):
    # uploads files to s3
    s3.Object(bucket_name=bucket, key=s3_path_to_file).put(Body=bin_data)  ## Body = some binary data protocol=2 in pickles


def download_file(s3, bucket, local_path_to_file, s3_path_to_file):
    # downloads files from s3
    file_binary_stream = s3.Object(bucket_name=bucket, key=s3_path_to_file)
    with open(local_path_to_file, 'wb') as f:
        f.write(file_binary_stream.get()["Body"].read())


def unpickle(path_to_file):
    # reads pickle file and returns the internal data structures
    with open(path_to_file, 'rb') as f:
        data = pickle.load(f)
    return data


def _2d_search(x1, x2, y1, y2, tree, is_strict=True):
    # 2d search using inteval trees
    # strict=true means totally contained will be returned
    # strict=false means just intersection
    tree_hor, tree_ver = tree
    hor_candidates = set(map(lambda item: item.data, tree_hor.search(x1, x2, is_strict)))
    ver_candidates = set(map(lambda item: item.data, tree_ver.search(y1, y2, is_strict)))
    return set.intersection(hor_candidates, ver_candidates)


def intervalize(sum_on_dim, thresh, shift):
    # like histogram projection which projects point to get intervals start and end
    # WWW WW W > (0,3),(4,6),(7,8)
    projection = np.reshape(np.argwhere(sum_on_dim == 0), [-1])
    intervals = []
    if len(projection) > 0:
        last_start = projection[0]
        for index in range(1, projection.shape[0]):
            if projection[index] != projection[index - 1] + 1:
                intervals.append((last_start, projection[index - 1]))
                last_start = projection[index]
        intervals.append((last_start, projection[-1]))

    return list(map(lambda item: shift + (item[1] + item[0]) // 2, filter(lambda item: (item[1] - item[0]) > thresh, intervals)))


import string
import random


def id_generator(size=12, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


# Initialize handwriting model
class JSONPredictor(RealTimePredictor):
    def __init__(self, endpoint_name, sagemaker_session):
        super(JSONPredictor, self).__init__(endpoint_name, sagemaker_session, json_serializer, json_deserializer)


# ------------------------------------------------------------ #
# Hosting methods                                              #
# ------------------------------------------------------------ #

def model_fn(model_dir):
    # here you should load your model (downloaded by sagemaker from s3 path you give to the endpoint-model path-)
    return None


def transform_fn(none_model, data, input_content_type, output_content_type):
    global page_scaling_vert, tess_scaling_vert, img_orig, page_scaling_hor, tess_scaling_hor, page, page_h, s3, img_for_predictors
    print("Getting Your files...")
    ######################################################
    # getting the data from s3
    parsed = json.loads(data)
    bucket = parsed['bucket']
    hocr_file_name = parsed['hocr_file']
    image_file_name_s3 = parsed['image_file']

    # endpoints

    loc_endpoint = parsed.get("loc_endpoint", "localization-model-2019-01-29")

    hw_endpoint = parsed.get("hw_endpoint", "pytorch-handwriting-ocr-2019-01-29-02-06-44-538")
    hp_endpoint = parsed.get("hp_endpoint", "hand-printed-model-2019-01-29-1")

    hw_endpoint_model = parsed.get("hw_endpoint_model", 'new')
    hp_endpoint_model = parsed.get("hp_endpoint_model", 'new')

    hw_endpoint_new_api = parsed.get("hw_endpoint_new_api", True)
    hp_endpoint_new_api = parsed.get("hp_endpoint_new_api", False)

    is_new_localizer = parsed.get("is_new_localizer", True)

    # access keys
    aws_access_key_id = parsed.get("aws_access_key_id", None)
    aws_secret_access_key = parsed.get("aws_secret_access_key", None)

    # getting data

    s3 = boto3.resource('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)  # new key
    hocr_file_stream = s3.Object(bucket_name=bucket, key=hocr_file_name)
    image_file_stream = s3.Object(bucket_name=bucket, key=image_file_name_s3)

    temp_hocr_file_name = id_generator() + ".hocr"
    with open(temp_hocr_file_name, 'wb') as f:  #
        f.write(hocr_file_stream.get()["Body"].read())

    temp_image_file_name = os.path.split(image_file_name_s3)[1]
    with open(temp_image_file_name, 'wb') as f:  #
        f.write(image_file_stream.get()["Body"].read())

        # communicating with endpoints
    s3 = boto3.resource('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
    session = boto3.Session(region_name='us-west-2', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
    sagemaker_session = sagemaker.Session(boto_session=session)
    loc_predictor = MXNetPredictor(loc_endpoint, sagemaker_session)

    hw_predictor = JSONPredictor(hw_endpoint, sagemaker_session)
    hp_predictor = MXNetPredictor(hp_endpoint, sagemaker_session)

    print("Uploading small image..")
    img_orig = cv2.imread(temp_image_file_name, 0)  # 7948*6198
    img_for_predictors = cv2.resize(img_orig.copy(), (2479, 3508))  # will be 3508*2479

    tess_scaling_vert = img_orig.shape[0] / img_for_predictors.shape[0]
    tess_scaling_hor = img_orig.shape[1] / img_for_predictors.shape[1]
    print("loc to tess scaling", (tess_scaling_vert, tess_scaling_hor))

    # ex: a/b/c.tiff
    directory_for_image = os.path.split(image_file_name_s3)[0]  # dir  a/b
    input_image_name_without_ext = os.path.split(image_file_name_s3)[1].split(".")[0]  # c image name
    image_extension = os.path.split(image_file_name_s3)[1].split(".")[1]  # tiff
    small_image_name = input_image_name_without_ext + "-small." + image_extension  # c-small.tiff
    small_image_path_s3 = os.path.join(directory_for_image, input_image_name_without_ext, small_image_name)

    cv2.imwrite(small_image_name, img_for_predictors)

    upload_file(s3, bucket, open(small_image_name, "rb"), small_image_path_s3)  #

    print("Calling localizer..")

    print("sending to localizer:" + small_image_path_s3)

    if is_new_localizer:
        loc_data = {"url": "s3://{}/{}".format(bucket, small_image_path_s3)}
    else:
        loc_data = {'bucket': bucket, 'file_name': small_image_path_s3}
    # Modify here
    tb = None
    try:
        loc_out = loc_predictor.predict(loc_data)

    except Exception as ex:
        tb = traceback.format_exc()

    if tb is not None:
        print("ERROR: {0}".format(tb))
        response_body = json.dumps({"status": "ERROR", "traceback": tb, "data": data})
        return response_body, output_content_type

    loc_out = loc_out["result"]
    # loc_out = loc_predictor.predict(loc_data)

    hw_data = {"bucket": loc_out["bucket_name"], "file_name": loc_out["hw_key"], "model": hw_endpoint_model}
    hp_data = {"bucket": loc_out["bucket_name"], "file_name": loc_out["hp_key"], "model": hp_endpoint_model}

    img = (255 - copy.copy(img_orig)) / 255
    img = img[:, :-100]

    print("Parsing the document..")
    repository = HocrDocument(temp_hocr_file_name)
    # repository.write_equivalent_xml()
    pages = repository.parse()
    pages_h = repository.hierarchical_parse()

    page = pages[list(pages.keys())[0]]
    page_h = pages_h[list(pages_h.keys())[0]]

    ##########################################################
    print("Calling Handwriting OCR...")

    try:
        repository.add_from_deployed(list(pages.keys())[0], hw_data, hw_predictor, is_new_api=hw_endpoint_new_api)
    except Exception as ex:
        tb = traceback.format_exc()
    if tb is not None:
        print("ERROR: {0}".format(tb))
        response_body = json.dumps({"status": "ERROR", "traceback": tb, "hw_data": hw_data})
        return response_body, output_content_type
    print("Calling Handprinting OCR...")
    try:
        repository.add_from_deployed(list(pages.keys())[0], hp_data, hp_predictor, is_new_api=hp_endpoint_new_api)
    except Exception as ex:
        tb = traceback.format_exc()
    if tb is not None:
        print("ERROR: {0}".format(tb))
        response_body = json.dumps({"status": "ERROR", "traceback": tb, "hp_data": hw_data})
        return response_body, output_content_type

    #########################################################

    def clean_files():
        os.remove(small_image_name)
        os.remove(temp_hocr_file_name)
        os.remove(temp_image_file_name)

    hw_thread = threading.Thread(target=clean_files)
    hw_thread.start()

    # import threading
    # print("Calling Handwriting OCR...")
    # hw_thread = threading.Thread(target=lambda: repository.add_from_deployed(list(pages.keys())[0], hw_data, hw_predictor))
    # hw_thread.start()
    #
    # print("Calling Handprinting OCR...")
    # hp_thread = threading.Thread(target=lambda: repository.add_from_deployed(list(pages.keys())[0], hp_data, hp_predictor))
    # hp_thread.start()
    #
    # hw_thread.join()
    # hp_thread.join()
    #########################################################

    page_scaling_hor = img.shape[1] / page['width']  # pages[1] : page text boxes coordinate system dimensions
    page_scaling_vert = img.shape[0] / page['height']  # pages[1] : page text boxes coordinate system dimensions
    print("Expected image:" + page["image"], ".......Given:" + image_file_name_s3)

    print("Text space from(PDF):", (page["height"], page["width"]))
    print("Image space:", img.shape)
    df_tabula = None  # If tables with lines, better use tabula

    if (df_tabula == None):
        verbose = False
        # 3508x2379 ~ 90linesx25words ~ pixels/word = 96, pixels/line=40--> strip_height > 40 (*2 for header usually > 2 lines) (100) w_max_pool < 96 (50)
        tables_detector = TableDetector(verbose, strip_height=50, w_max_pool=75, min_col_width=250, ratio_clip_max=0.25)
        print("Detecting Tables..")
        tables = tables_detector.detect_tables(img)
        # tables_detector.visualize_tables(img_orig)
        print("Fitting Text..")
        tables_detector.fit_bordered_tables()  # <<< pdf coordinates are compared to image coordinates (scaling needed)
        tables_detector.layout_based_borderless_detection()
        response_body = tables_detector.get_json_response()
    else:
        df_res = df_tabula
        list_of_json_tables = [json.loads(df_res.to_json())]
        response_body = json.dumps({"data": list_of_json_tables})

    print(os.popen("df . -m").read())
    return response_body, output_content_type
