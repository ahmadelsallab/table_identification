import copy
import subprocess
import sys
import traceback

import json
import os

import numpy as np

from pdftabextract.common import read_xml, parse_pages
from pdftabextract.clustering import calc_cluster_centers_1d
from pdftabextract.clustering import find_clusters_1d_break_dist
from pdftabextract.clustering import zip_clusters_and_values
from pdftabextract.common import DIRECTION_VERTICAL
from pdftabextract.common import parse_pages
from pdftabextract.extract import fit_texts_into_grid, datatable_to_dataframe
from pdftabextract.extract import make_grid_from_positions
from pdftabextract.textboxes import border_positions_from_texts
import matplotlib.pyplot as plt

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage.interpolation import shift
import seaborn as sns
import pandas as pd
import json


class TableDetector:

    def __init__(self, verbose=False, strip_height=50, w_max_pool=50, min_col_width=50, ratio_clip_max=0.8):
        self.verbose = verbose
        self.state = 'Table_Search'
        self.state_machine = {'Table_Search': self.table_search,
                              'Candidate_Table': self.candidate_table,
                              'Table_Registered': self.table_registered,
                              'Confirm_Table_End': self.confirm_table_end}
        self.tables = []
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

        return

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
        return

    def table_registered(self):
        '''
        if(self.verbose):
            print('Table_Registered')
        '''
        if len(self.col_positions) == 0:  # or len(self.col_positions) != self.n_cols:
            # if len(self.col_positions) == 0:
            self.table_info['table_end'] = self.start
            self.state = 'Confirm_Table_End'
        return

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
        return

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
        max_thresh = 7
        n_w = int(len(h)/w)
        h_maxes = np.zeros(len(h))
        for i in range(n_w):

            #h_maxes[i*w:(i+1)*w] = np.mean(h[i*w:(i+1)*w])
            #h_maxes[i*w:(i+1)*w] = max(h[i*w:(i+1)*w]) 
            
            local_max = max(h[i*w:(i+1)*w])
            if(local_max > max_thresh):
                h_maxes[i*w:(i+1)*w] = max(h[i*w:(i+1)*w]) 
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
        g_clean = self.zero_crossings(h_maxes.copy())
        #g_clean = g
        if self.verbose:
            self.h = h_maxes.copy()
            #self.g = g_clean.copy()
            self.g = g_clean.copy()
            
        
        return g_clean

    def check_row_pattern(self, row_grads):
        # self.g = 0*row_grads.copy()
        if len(np.argwhere(row_grads == 1)) >= 2:  # +v crossing

            # If dist between 2 1's < min_col_width--> set all to zeros
            row_grads = self.remove_false_cols(row_grads)
            # self.g = row_grads.copy()
            # Count 1's => 2
            if len(np.argwhere(row_grads==1)) >= 2:
                col_positions = np.squeeze(np.argwhere(row_grads==1))                   

            else:
                col_positions = []
        else:
            col_positions = []

        return col_positions

    def row_pattern_detect(self, img_strip):
        g_clean = self.preprocess_row(img_strip)
        col_positions = self.check_row_pattern(g_clean)  # clear noisy columns

        self.col_positions = col_positions.copy()

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
        self.img_shape = img.shape
        H, W = img.shape
        n_strips = int(np.floor(H / self.strip_height))
        # overlap = 0.5

        for i in range(n_strips):
            self.start = i * self.strip_height
            self.end = self.start + self.strip_height
            img_strip = img[self.start:self.end, :]
            col_positions = self.row_pattern_detect(img_strip)
            self.state_machine[self.state]()
            if self.verbose:
                fig = plt.figure()    
                # Show the orig image with the strip on col 1
                ax = fig.add_axes([0,0,1,1])
                rect = patches.Rectangle((0,self.start), W, self.strip_height, linewidth=1, edgecolor='r', facecolor='none')
                
                
                ax.add_patch(rect)
                
                ax.imshow(img, cmap='gray')
                fig.savefig('images/' + file_name + '_image_stip_' + str(i) + '_table.jpg')
                #fig.clf()
                
                fig = plt.figure()    
                ax = fig.add_subplot(211)
                n_cols = len(col_positions)
                ax.text(10,0,self.state + ', n_cols: ' + str(col_positions))#str(n_cols))
                #ax.text(10,100,str(col_positions))
                ax.plot(self.h)    
                # Show next the cleaned up grad or histogram
                ax = fig.add_subplot(212)
                ax.plot(self.g)    
                #plt.imsave(file_name + '_' + str(i) + '_table.jpg', fig, cmap='gray')
                fig.savefig('plots/' + file_name + '_plots_' + str(i) + '_table.jpg')
                #fig.savefig(file_name + '_' + str(i) + '_table.jpg')
                #fig.clf()
        
        self.tables = self.adjust_tables_boundaries(img)

        return self.tables

    def visualize_tables(self, img_orig, file_name):
        tables = self.tables
        tables_img = img_orig
        line_width = 10
        fig = plt.figure()
        for table in tables:
            # table_start = table['table_start'] - self.strip_height
            # table_end = table['table_end'] - self.strip_height
            table_start = table['table_start']
            table_end = table['table_end']
            # Last colomn boundary = min(img_boundary, last_col_start + spacing of the previous colomn)
            #last_col = min(tables_img.shape[1], table['col_positions'][-1] + (table['col_positions'][-1] - table['col_positions'][-2]))
            #last_col = tables_img.shape[1] - table['col_positions'][0]
            last_col = table['col_positions'][-1]
            # Draw table boundaries
            tables_img[table_start - line_width:table_start + line_width, table['col_positions'][0]: last_col] = 0
            tables_img[table_end - line_width:table_end + line_width, table['col_positions'][0]: last_col] = 0
            # Draw cols
            for col in table['col_positions']:
                tables_img[table_start:table_end, col - line_width:col + line_width] = 0
            # Left boundary
            tables_img[table_start:table_end, last_col - line_width:last_col + line_width] = 0
        plt.imshow(tables_img, cmap='gray')
        plt.imsave(file_name + '_table.jpg', tables_img, cmap='gray')
        plt.show()
        return

    def fit_text_to_tables(self, xmlroot):

        # Load the XML that was generated with pdftohtml
        #xmltree, xmlroot = read_xml(xml_file)
        # parse it and generate a dict of pages
        pages = parse_pages(xmlroot)
        page_1 = pages[1]
        self.tables_df = []
        self.page_scaling_hor = self.img_shape[1] / page_1['width']  # pages[1] : page text boxes coordinate system dimensions
        self.page_scaling_vert = self.img_shape[0] / page_1['height']  # pages[1] : page text boxes coordinate system dimensions        
        for table in self.tables:
            print('=' * 100)
            """
            I refactored variables to end with _is = image space(tiff)
            or _ps = page space(pdf)
            """

            page_colpos_is = table['col_positions']
            # right border of the last column
            last_rightborder_is = page_colpos_is[-1]

            # calculate median text box height
            median_text_height_ps = np.median([t_ps['height'] for t_ps in page_1['texts']])

            # get all texts in all the with a "usual" textbox height
            # we will only use these text boxes in order to determine the line positions

            text_height_deviation_thresh_ps = median_text_height_ps / 2
            texts_cols_ps = [t_ps for t_ps in page_1['texts']
                             if self.get_in_hor_image_space(t_ps['right']) <= last_rightborder_is  # <<<<<<<<<< scaling to image space
                             and abs(t_ps['height'] - median_text_height_ps) <= text_height_deviation_thresh_ps]
            # Next we get the text boxes' top and bottom border positions, cluster them, and calculate the cluster centers.

            # get all textboxes' top and bottom border positions
            borders_y_ps = border_positions_from_texts(texts_cols_ps, DIRECTION_VERTICAL)

            # break into clusters using half of the median text height as break distance
            clusters_y_ps = find_clusters_1d_break_dist(borders_y_ps, dist_thresh=median_text_height_ps / 2)
            clusters_w_vals_ps = zip_clusters_and_values(clusters_y_ps, borders_y_ps)

            # for each cluster, calculate the median as center
            pos_y_ps = calc_cluster_centers_1d(clusters_w_vals_ps)
            pos_y_ps.append(page_1['height'])

            top_y_is = table['table_start']

            bottom_y_is = table['table_end']

            # finally filter the line positions so that only the lines between the table top and bottom are left
            page_rowpos_ps = [y_ps for y_ps in pos_y_ps if top_y_is <= self.get_in_vert_image_space(y_ps) <= bottom_y_is]

            ## 7. Create a grid of columns and lines

            # From the column and row positions that we detected, we can now generate a "page grid" which should resemble the table layout as close as possible. We then save the grid information as JSON file so that we can display it in pdf2xml-viewer.

            try:
                grid_ps = make_grid_from_positions(self.get_in_hor_page_space(page_colpos_is), page_rowpos_ps)

                # if len(grid) != 0:

                n_rows = len(grid_ps)
                n_cols = len(grid_ps[0])
                print("> Grid with %d rows, %d columns" % (n_rows, n_cols))

                ## 8. Match the text boxes into the grid and hence extract the tabular data in order to export it as Excel and CSV file

                # We can use `fit_texts_into_grid` to fit the text boxes into the grid and then transform it to a [pandas](http://pandas.pydata.org/) *DataFrame*.

                datatable = fit_texts_into_grid(page_1['texts'], grid_ps)

                df = datatable_to_dataframe(datatable)

                # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                #     print(df.encode('utf-8'))

                self.tables_df.append(df)

            except Exception as e:
                print(e)
                print('Empty table\n')

        print('=' * 100)

    def get_tables_in_json(self):
        tables_json = []
        for i, table in enumerate(self.tables_df):
            tables_json.append(json.loads(table.to_json().encode('utf-8').decode('utf-8')))  # file_name + '_table_' + str(i) + '.json'

        return tables_json





    def get_in_hor_image_space(self, value_from_xml_hor):
        return value_from_xml_hor * self.page_scaling_hor


    def get_in_vert_image_space(self, value_from_xml_vert):
        return value_from_xml_vert * self.page_scaling_vert


    def get_in_hor_page_space(self, value_from_image_hor):
        return value_from_image_hor / self.page_scaling_hor


    def get_in_vert_page_space(self, value_from_image_vert):
        return value_from_image_vert / self.page_scaling_vert
