#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import quilt3
from pathlib import Path
from typing import List, Union
import pandas as pd


class QuiltModelZoo():
    def __init__(self):
        """connect to model zoo on quilt3"""

        # connect to quilt
        self.pkg = quilt3.Package.browse("aics/segmenter_model_zoo", registry="s3://allencell")
        self.meta = self.pkg['metadata.csv']()

    def peak_all_models(self) -> List:
        """print out names of existing models in the model zoo"""
        models = list(self.meta.name)
        print(models)
        return models

    def download_model(self, model_name: str, save_path: Union(str, Path) = './zoo/model.pth'):
        """
        download the model "model_name" to "out_path"

        Paremeters
        ---
        model_name: str
            the name of the model to be downloaded
        save_path: Union(str, Path)
            the path to save the model, default is './zoo/model.pth'
        """
        # check if the model name is valide
        assert model_name in self.meta.name, f"requested model {model_name} does not exist"

        # check if save_path already has the model
        save_dir = os.path.dirname(save_dir)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        else:
            assert os.path.exists(save_path), f"the save_path {save_path} is already used"

        # fetch the model file
        model_id = self.meta[self.meta['name']==model_name]['models'].iloc[0]
        model_path = model_id.split('/')
        self.pkg[model_path[0]][model_path[1]].fetch(save_path)