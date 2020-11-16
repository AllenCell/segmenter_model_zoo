"""
This sample script will get deployed in the bin directory of the
users' virtualenv when the parent module is installed using pip.
"""
import os
import argparse
import logging
import sys
import traceback
from pathlib import Path, PurePosixPath
from typing import List
from tqdm import tqdm
import yaml
import numpy as np

from segmenter_model_zoo.zoo import SuperModel
from segmenter_model_zoo.utils import load_filenames, save_as_uint

###############################################################################

log = logging.getLogger()
# Note: basicConfig should only be called in bin scripts (CLIs).
# https://docs.python.org/3/library/logging.html#logging.basicConfig
# "This function does nothing if the root logger already has handlers configured for
# it." As such, it should only be called once, and at the highest level (the CLIs in
# this case). It should NEVER be called in library code!
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s - %(name)s - %(lineno)3d][%(levelname)s] %(message)s",
)

###############################################################################


class Args(argparse.Namespace):
    def __init__(self):
        # Arguments that could be passed in through the command line
        self.debug = False
        self.__parse()

    def __parse(self):
        p = argparse.ArgumentParser(
            prog="run dl segmentation",
            description="the entry point for running segmentation using ML model zoo",
        )
        p.add_argument(
            "--config", required=True, help="the path to the configuration file"
        )
        p.add_argument(
            "-d", "--debug", action="store_true", dest="debug", help=argparse.SUPPRESS
        )
        p.add_argument(
            "--overwrite",
            action="store_true",
            help="whether to overwrite existing results",
        )
        p.add_argument(
            "--search_tag",
            type=str,
            help="a string used to match filename when checking existence",
        )
        p.parse_args(namespace=self)


class Seg3DStacks(object):
    def __init__(self, config, args):
        self.model = SuperModel(config["model"], config)
        self.overwrite = args.overwrite
        self.tag = args.search_tag
        self.debug = args.debug
        self.output_path = config["output_path"]
        self.input_channel = config["input_channel"]

    def execute(self, filenames: List):
        # print out all filenames, if debug
        if self.debug:
            print(filenames)

        save_path = Path(self.output_path)
        for fi, fn in tqdm(enumerate(filenames)):
            fn_core = PurePosixPath(os.path.basename(fn)).stem
            if not self.overwrite:
                if self.tag is None:
                    # check if similar segmentation exists
                    existing_results = list(save_path.glob(fn_core + "*."))
                    if len(existing_results) > 0:
                        print("the following files already exists, please check.")
                        print(existing_results)
                        continue
                else:
                    # do exact match
                    check_name = save_path / f"{fn_core}_{self.search_tag}.tiff"
                    if check_name.exists():
                        print(f"{check_name} exists, skipping")
                        continue

            # run segmentation
            seg = self.model.apply_on_single_zstack(
                filename=fn, inputCh=self.input_channel
            )

            ##############################################
            # check return type.
            # two options: np.ndarray | List
            #   - case 1: np.ndarray (simply one segmentation)
            #   - case 2: List (two sub-lists: seg and seg_name)
            ##############################################
            if isinstance(seg, np.ndarray):
                save_as_uint(seg, save_path, fn_core, self.overwrite)
            elif isinstance(seg, List):
                seg_output = zip(seg[0], seg[1])
                for seg_img, seg_tag in seg_output:
                    save_as_uint(seg_img, save_path, fn_core, seg_tag, self.overwrite)

        print("all files are done")


###############################################################################


def main():
    try:
        args = Args()
        dbg = args.debug
        print(args.config)
        config = yaml.load(open(args.config, "r"))
        all_files, timelapse_flag = load_filenames(config["Data"])
        print(all_files)

        # Do your work here - preferably in a class or function,
        # passing in your args. E.g.
        if timelapse_flag:
            print("timelapse is not supported yet")
            sys.exit(0)
        else:
            exe = Seg3DStacks(config, args)
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

if __name__ == "__main__":
    main()
