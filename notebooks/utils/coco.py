import numpy as np
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
from pathlib import Path


class COCOLoader(COCO):
    def __init__(self, data_dir):
        """
        Parameters
        ----------
        data_dir
        str: location of COCO data files
        """
        self.data_dir = Path(data_dir)
        self.annotations_dir = self.data_dir.joinpath('annotations/instances_train2017.json')
        self.train_dir = self.data_dir.joinpath('train2017')
        super(COCOLoader, self).__init__(self.annotations_dir.as_posix())

coco_loader = COCOLoader('/Users/jbsnyder/PycharmProjects/mrcnn-notebooks')