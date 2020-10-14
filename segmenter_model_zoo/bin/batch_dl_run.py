"""
This sample script will get deployed in the bin directory of the
users' virtualenv when the parent module is installed using pip.
"""

import argparse
import logging
import sys
import os
import traceback
import pathlib

import numpy as np
import yaml
from aicsimageio import AICSImage, omeTifWriter
from model_zoo_3d_segmentation.zoo import SegModel, SuperModel
from model_zoo_3d_segmentation.utils import load_filenames
#from aicsmlsegment.utils import load_config

###############################################################################

log = logging.getLogger()
# Note: basicConfig should only be called in bin scripts (CLIs).
# https://docs.python.org/3/library/logging.html#logging.basicConfig
# "This function does nothing if the root logger already has handlers configured for it."
# As such, it should only be called once, and at the highest level (the CLIs in this case).
# It should NEVER be called in library code!
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s - %(name)s - %(lineno)3d][%(levelname)s] %(message)s')

###############################################################################


class Args(argparse.Namespace):

    def __init__(self):
        # Arguments that could be passed in through the command line
        self.debug = False
        self.__parse()

    def __parse(self):
        p = argparse.ArgumentParser(prog='run dl segmentation',
                                    description='the entry point for running dl segmentation')
        p.add_argument('--config', required=True, help='the path to the configuration file')
        p.add_argument('-d', '--debug', action='store_true', dest='debug', help=argparse.SUPPRESS)
        p.parse_args(namespace=self)

class Seg3DStacks(object):

    def __init__(self, args):
        self.model = SuperModel(args['model'],args)
        self.config = args

    def execute(self, filenames):

        if not os.path.exists(self.config['output_path']):
            os.mkdir(self.config['output_path'])
            
        ##########################################################################
        for fi, fn in enumerate(filenames):
            print(fn)
            if self.config['model'] == 'LMNB1_morphological_production_alpha':
                seg = self.model.apply_on_single_zstack(filename=fn,inputCh=self.config['input_channel'])
                seg = seg.astype(np.uint8)
                seg[seg>0] = 255
                
                out_fn = self.config['output_path']+ os.sep + pathlib.PurePosixPath(fn).stem + '_struct_segmentation.tiff'
                writer = omeTifWriter.OmeTifWriter(out_fn)
                writer.save(seg)

            elif self.config['model'] == 'LMNB1_morphological_RnD':
                seg, seg_name = self.model.apply_on_single_zstack(filename=fn,inputCh=self.config['input_channel'])
                for li, seg_n in enumerate(seg_name):
                    out_fn = self.config['output_path']+ os.sep + pathlib.PurePosixPath(fn).stem + seg_n + 'segmentation.tiff'
                    writer = omeTifWriter.OmeTifWriter(out_fn)
                    writer.save(seg[li].astype(np.uint8))

            elif self.config['model'] == 'LMNB1_morphological_with_labelfree':
                seg, seg_name = self.model.apply_on_single_zstack(filename=fn,inputCh=self.config['input_channel'])
                
                for li, seg_n in enumerate(seg_name):
                    out_fn = self.config['output_path']+ os.sep + pathlib.PurePosixPath(fn).stem + seg_n + 'segmentation.tiff'
                    writer = omeTifWriter.OmeTifWriter(out_fn)
                    writer.save(seg[li].astype(np.uint8))

            elif self.config['model'] == 'LMNB1_fill_instance_100x_hipsc':

                #print(self.config['output_path']+ os.sep + pathlib.PurePosixPath(fn).stem + '_fill_segmentation.tiff')
                if os.path.exists(self.config['output_path']+ os.sep + pathlib.PurePosixPath(fn).stem + '_fill_segmentation.tiff'):
                    continue

                seg, seg_name = self.model.apply_on_single_zstack(filename=fn,inputCh=self.config['input_channel'])
                
                for li, seg_n in enumerate(seg_name):
                    out_fn = self.config['output_path']+ os.sep + pathlib.PurePosixPath(fn).stem + seg_n + 'segmentation.tiff'
                    writer = omeTifWriter.OmeTifWriter(out_fn)
                    writer.save(seg[li].astype(np.uint16))

            elif self.config['model'] == 'DNA_MEM_instance_production_alpha':
                seg_mem, seg_dna, seed_label = self.model.apply_on_single_zstack(filename=fn,inputCh=self.config['input_channel'])
                out_mem = self.config['output_path']+ os.sep + pathlib.PurePosixPath(fn).stem + '_mem_segmentation.tiff'
                writer = omeTifWriter.OmeTifWriter(out_mem)
                writer.save(seg_mem.astype(np.uint8))

                out_dna = self.config['output_path']+ os.sep + pathlib.PurePosixPath(fn).stem + '_dna_segmentation.tiff'
                writer = omeTifWriter.OmeTifWriter(out_dna)
                writer.save(seg_dna.astype(np.uint8))

                out_seed = self.config['output_path']+ os.sep + pathlib.PurePosixPath(fn).stem + '_seed.tiff'
                writer = omeTifWriter.OmeTifWriter(out_seed)
                writer.save(seed_label.astype(np.uint8))
            elif self.config['model'] == 'LF_DNA_instance_alpha':
                seg_dna = self.model.apply_on_single_zstack(filename=fn,inputCh=self.config['input_channel'])

                out_dna = self.config['output_path']+ os.sep + pathlib.PurePosixPath(fn).stem + '_dna_segmentation.tiff'
                writer = omeTifWriter.OmeTifWriter(out_dna)
                writer.save(seg_dna.astype(np.uint8))

            elif self.config['model'] =='DNA_MEM_instance_LF_integration' or self.config['model'] == 'DNA_MEM_instance_LF_integration_two_camera':
                out_fn = self.config['output_path']+ os.sep + pathlib.PurePosixPath(fn).stem + '_segmentation.tiff'
                if os.path.exists(out_fn):
                    print(f'skipping {fn}, results already exist')
                    continue
                seg_combined = self.model.apply_on_single_zstack(filename=fn,inputCh=self.config['input_channel'])
                if seg_combined is None:
                    print(f'skipping {fn}, due to failed segmentation')
                    continue
                
                writer = omeTifWriter.OmeTifWriter(out_fn, overwrite_file=True)
                writer.save(seg_combined)

            elif self.config['model'] =='DNA_MEM_instance_CAAX_with_BF':
                out_fn = self.config['output_path']+ os.sep + pathlib.PurePosixPath(fn).stem + '_segmentation.tiff'
                if os.path.exists(out_fn):
                    print(f'skipping {fn}, results already exist')
                    continue
                seg_combined = self.model.apply_on_single_zstack(filename=fn,inputCh=self.config['input_channel'])
                if seg_combined is None:
                    print(f'skipping {fn}, due to failed segmentation')
                    continue
                
                writer = omeTifWriter.OmeTifWriter(out_fn, overwrite_file=True)
                writer.save(seg_combined)

            elif self.config['model'] == 'structure_H2B_production':
                seg = self.model.apply_on_single_zstack(filename=fn, inputCh=self.config['input_channel'])
                seg = seg.astype(np.uint8)
                seg[seg>0] = 255
                
                out_fn = self.config['output_path']+ os.sep + pathlib.PurePosixPath(fn).stem + '_struct_segmentation.tiff'
                writer = omeTifWriter.OmeTifWriter(out_fn)
                writer.save(seg)

            elif self.config['model'] == 'structure_AAVS1_production':
                seg = self.model.apply_on_single_zstack(filename=fn, inputCh=self.config['input_channel'])
                seg = seg.astype(np.uint8)
                seg[seg>0] = 255
                
                out_fn = self.config['output_path']+ os.sep + pathlib.PurePosixPath(fn).stem + '_struct_segmentation.tiff'
                writer = omeTifWriter.OmeTifWriter(out_fn)
                writer.save(seg)

            elif self.config['model'] == 'DNA_instance_LF_integration_two_camera':
                out_mem = self.config['output_path']+ os.sep + pathlib.PurePosixPath(fn).stem + '_mem_segmentation.tiff'
                out_dna = self.config['output_path']+ os.sep + pathlib.PurePosixPath(fn).stem + '_dna_segmentation.tiff'

                if os.path.exists(out_mem) and os.path.exists(out_dna):
                    print(f'skipping {fn}, results already exist')
                    continue
                
                seg_mem, seg_dna = self.model.apply_on_single_zstack(filename=fn,inputCh=self.config['input_channel'])
                
                writer = omeTifWriter.OmeTifWriter(out_mem, overwrite_file=True)
                writer.save(seg_mem.astype(np.uint8))
                
                writer = omeTifWriter.OmeTifWriter(out_dna, overwrite_file=True)
                writer.save(seg_dna.astype(np.uint8))
            else:
                out_fn = self.config['output_path']+ os.sep + pathlib.PurePosixPath(fn).stem + '_segmentation.tiff'
                if os.path.exists(out_fn):
                    print(f'skipping {fn}, results already exist')
                    continue

                seg = self.model.apply_on_single_zstack(filename=fn,inputCh=self.config['input_channel'])
                
                writer = omeTifWriter.OmeTifWriter(out_fn)
                writer.save(seg.astype(np.uint8))

        print('all files are done')
###############################################################################

def main():
    try:
        args = Args()
        dbg = args.debug
        print(args.config)
        config = yaml.load(open(args.config, 'r'))
        all_files, timelapse_flag = load_filenames(config['Data'])
        print(all_files)

        # Do your work here - preferably in a class or function,
        # passing in your args. E.g.
        if timelapse_flag:
            print('not support yet')
            quit()
            #exe = Seg3DTimelapse(args)
        else:
            exe = Seg3DStacks(config)
        exe.execute(all_files)

    except Exception as e:
        log.error("=============================================")
        if dbg:
            log.error("\n\n" + traceback.format_exc())
            log.error("=============================================")
        log.error("\n\n" + str(e) + "\n")
        log.error("=============================================")
        sys.exit(1)


###############################################################################
# Allow caller to directly run this module (usually in development scenarios)

if __name__ == '__main__':
    main()
