import numpy as np
import logging
import sys
from aicsimageio import AICSImage
from aicsimageprocessing import resize
import os
from scipy import ndimage as ndi
from scipy import stats
import argparse
from glob import glob

import yaml
import re

def load_filenames(data_config):

    all_stacks = []
    all_timelapse = []
    for data_item in data_config:
        if data_item['item']=='folder':
            
            if data_item['search'] == '*':
                filenames = glob(data_item['dir']+os.sep+'/*')
            else:
                reg = re.compile(data_item['search'])
                filenames = [data_item['dir'] + os.sep + f for f in os.listdir(data_item['dir']) if reg.search(f)]
            filenames.sort()
            #filenames = sorted(filenames, reverse=True)
            if 'data_range' in data_item:
                if data_item['data_range'][1] == -1:
                    all_stacks.extend(filenames[data_item['data_range'][0]:])
                else:
                    all_stacks.extend(filenames[data_item['data_range'][0]:data_item['data_range'][1]])
            else:
                all_stacks.extend(filenames)
        elif data_item['item']=='zstack':
            all_stacks.extend([data_item['dir']])
        elif data_item['item']=='timelapse':
            all_timelapse.extend(str(data_item['dir']))
        elif data_item['item']=='csv':
            import pandas as pd
            df = pd.read_csv(data_item['file'])
            col = str(data_item['column'])
            for _index, row in df.iterrows():
                fn = row[col]
                assert os.path.exists(fn)
                all_stacks.extend([fn])

    if len(all_stacks)>0 and len(all_timelapse)>0:
        print('we cannot handle timelapse data and non-timelapse data in one shot')
        print('please do this separately')
        quit()
    elif len(all_stacks)==0 and len(all_timelapse)==0:
        print('not file is found')
        quit()

    if len(all_stacks)>0:
        return all_stacks, False 
    else: 
        print('timelapse is not supported yet')
        #return all_timelapse, True