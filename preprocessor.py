#!/usr/bin/env python
# Copyright 2019 Augusto Cunha and Axelle Pochet
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this code and 
# associated documentation files, to deal in the code without restriction, 
# including without limitation the rights to use, copy, modify, merge, publish, distribute, 
# sublicense, and/or sell copies of the code, and to permit persons to whom the code is 
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or 
# substantial portions of the code.
#
# THE CODE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT 
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND 
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, 
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
# OUT OF OR IN CONNECTION WITH THE CODE OR THE USE OR OTHER DEALINGS IN THE CODE.
__license__ = "MIT"
__author__ = "Augusto Cunha, Axelle Pochet"
__email__ = "acunha@tecgraf.puc-rio.br, axelle@tecgraf.puc-rio.br"
__credits__ = ["Augusto Cunha", "Axelle Pochet", "Helio Lopes", "Marcelo Gattass"]

import gc
import cv2, os, numpy, sys
import pandas as pd
import multiprocessing
import time
from joblib import Parallel, delayed

from tqdm import tqdm
from itertools import product

numpy.random.seed(1337)

def processPatches(data, patch_size, pixel_step, resize, nb_channels):
    # get data
    if isinstance(data, pd.DataFrame):
        section_mat = data.values
    else:
        section_mat = data
    half_patch = int(patch_size/2)
    
    # get image info
    nb_rows = data.shape[0] 
    nb_cols = data.shape[1]
    print("[Processing Patches] data.shape[0] =", nb_rows)
    print("[Processing Patches] data.shape[1] =", nb_cols)
    
    count_patches = 0
    patch_name_list = []
    patch_list = []
    total = len(list(range(half_patch, nb_rows - half_patch, pixel_step)))**2
    i_range = range(half_patch, nb_rows - half_patch, pixel_step)
    j_range = range(half_patch, nb_cols - half_patch, pixel_step)
    
    with tqdm(total=total) as pbar:
        for i, j in product(i_range, j_range):
            # create patch
            start_row = i - half_patch
            start_col = j - half_patch
            end_row = start_row + patch_size
            end_col = start_col + patch_size
            # patch = numpy.zeros((patch_size,patch_size)) # 1 empty patch
            # for x in range(patch_size):
            #     for y in range(patch_size):
            #         patch[x][y] = section_mat[start_row + x][start_col + y]
            patch = numpy.array(
                section_mat[start_row:end_row, start_col:end_col], 
                copy=True)

            # resize, clip
            patch = cv2.resize(patch, dsize=(resize, resize), interpolation=cv2.INTER_CUBIC)
            patch = numpy.clip(patch, -1., 1.)

            # append to global list
            patch_list.append(patch)
            patch_name = 'patch_p_' + str(i) + '_' + str(j) + '.csv'
            patch_name_list.append(patch_name)
            # count
            count_patches +=1
            pbar.update(1)
    
    return patch_list, patch_name_list
