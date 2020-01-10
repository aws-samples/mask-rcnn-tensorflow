import numpy as np
import os
import shutil
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
from pathlib import Path
import json

class COCOSubsetter(object):
    """
    Tools for subsetting COCO data. Can be used to create a smaller subset of
    data either randomly, or by using in conjunction with pycocotools
    to subset by category. Can also be used to duplicate a dataset. For example,
    if the user wants to train on a single image for testing, that image can be
    duplicated multiple times.
    """
    def __init__(self, data_dir):
        """

        Parameters
        ----------
        data_dir : str
            Filepath location of of COCO data. Expects to find
            subdirectories for train2017 and annotations
        """
        self.data_dir = Path(data_dir)
        self.instance_file = self.data_dir.joinpath('annotations/instances_train2017.json')
        self.train_dir = self.data_dir.joinpath('train2017')
        self.images = list(self.train_dir.glob('*.jpg'))
        self.images = {int(os.path.splitext(os.path.basename(i.as_posix()))[0]): \
                           i for i in self.images}
        self.load_annotations()
        return

    def random_subset(self, count):
        """

        Parameters
        ----------
        count : int
            the number of random images to select

        Returns
        -------
        dict
            dictionary of {image_id: Path(image)}
        """
        images = np.random.choice(list(self.images.keys()), size=count)
        return {i:self.images[i] for i in images}

    def load_annotations(self):
        """
        Load annotations for COCO data

        Returns
        -------

        """
        with open(self.instance_file) as infile:
            self.instances = json.load(infile)

    def _create_new_annotations(self, annotations, images):
        """
        Used for generating a new set of annotations
        info licenses and categories are the same for
        the subset, so just copy them. For annotations
        and images, take a new set to combine.
        Parameters
        ----------
        annotations : list[dict]
            a list of dictionaries of annotations
        images : list[dict]
            a list of dictionaries of image information

        Returns
        -------
        dict
            A dictionary mirroring the annotations format
        """
        new_annotations = dict()
        new_annotations['info'] = self.instances['info']
        new_annotations['licenses'] = self.instances['licenses']
        new_annotations['categories'] = self.instances['categories']
        new_annotations['annotations'] = annotations
        new_annotations['images'] = images
        return new_annotations

    def filter_annotations(self, images):
        """
        Given a set of image ids, subset the annotations and images
        and combine with other fields of the annotations file
        Parameters
        ----------
        images : list[int]
            a list of image ids

        Returns
        -------
        dict
            A dictionary of new annotations
        """
        annotations = [i for i in self.instances['annotations'] if i['image_id'] in images]
        images = [i for i in self.instances['images'] if i['id'] in images]
        return self._create_new_annotations(annotations, images)

    def duplicate_annotations(self, count):
        """
        Create duplicates of the annotations by incrementing ids and filenames.
        Given a count, apply the range of (0,count) to the end of the image ids
        and filenames.

        Parameters
        ----------
        count : int
            The number of time to duplicate the annotations

        Returns
        -------
        dict
            A dictionary of new annotations
        """
        new_annotations = []
        new_images = []
        for num in range(count):
            for anno in self.instances['annotations']:
                anno_copy = anno.copy()
                anno_copy['image_id'] = int("{}{}".format(anno['image_id'], str(num)))
                new_annotations.append(anno_copy)
            for image in self.instances['images']:
                image_copy = image.copy()
                filename = os.path.splitext(image_copy['file_name'])
                image_copy['file_name'] = "{}{}{}".format(filename[0],
                                                          str(num),
                                                          filename[1])
                image_id = int("{}{}".format(image_copy['id'], str(num)))
                image_copy['id']=image_id
                new_images.append(image_copy)
        return self._create_new_annotations(new_annotations, new_images)

    def create_subset_dir(self, dir):
        """
        Checks if the output directory exists and creates it if it doesn't
        Parameters
        ----------
        dir : str
            filepath for output

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If directory already exists, return error
        """
        assert not dir.exists(), "directory {} exists".format(dir.as_posix())
        dir.mkdir()
        dir.joinpath('annotations').mkdir()
        dir.joinpath('train2017').mkdir()

    def create_subset(self, images, dir):
        """
        Create a new dataset based on a list of images

        Parameters
        ----------
        images : list[int]
            A list of image ids
        dir : str
            path for output

        Returns
        -------
        None
        """
        dir = Path(dir)
        self.create_subset_dir(dir)
        for image in images:
            shutil.copy(self.images[image],
                        dir.joinpath('train2017').joinpath(os.path.basename(self.images[image])))
        with open(dir.joinpath('annotations').joinpath('instances_train2017.json'), 'w') as anno_file:
            anno_file.write(json.dumps(self.filter_annotations(images)))

    def duplicate_dataset(self, count, dir):
        """
        Create a new dataset with duplicated images

        Parameters
        ----------
        count : int
            Number of duplicates to generate
        dir : str
            output directory

        Returns
        -------
        None
        """
        dir = Path(dir)
        self.create_subset_dir(dir)
        new_annotations = self.duplicate_annotations(count)
        for image in self.images.values():
            basename, _ = os.path.splitext(os.path.basename(image))
            for num in range(count):
                new_file = basename + str(num) + '.jpg'
                shutil.copy(image,
                        dir.joinpath('train2017').joinpath(new_file))
        with open(dir.joinpath('annotations').joinpath('instances_train2017.json'), 'w') as outfile:
            outfile.write(json.dumps(new_annotations))

