# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
# File: eval.py

import itertools
import sys
import os
import numpy as np
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import cv2
import pycocotools.mask as cocomask
import tqdm
from scipy import interpolate

from tensorpack.callbacks import Callback
from tensorpack.utils import logger
from tensorpack.utils.utils import get_tqdm

from common import CustomResize, clip_boxes
from data import get_eval_dataflow, get_batched_eval_dataflow
from dataset import DetectionDataset
from config import config as cfg

try:
    import horovod.tensorflow as hvd
except ImportError:
    pass


DetectionResult = namedtuple(
    'DetectionResult',
    ['box', 'score', 'class_id', 'mask'])
"""
box: 4 float
score: float
class_id: int, 1~NUM_CLASS
mask: None, or a binary image of the original image shape
"""

def _scale_box(box, scale):
    w_half = (box[2] - box[0]) * 0.5
    h_half = (box[3] - box[1]) * 0.5
    x_c = (box[2] + box[0]) * 0.5
    y_c = (box[3] + box[1]) * 0.5

    w_half *= scale
    h_half *= scale

    scaled_box = np.zeros_like(box)
    scaled_box[0] = x_c - w_half
    scaled_box[2] = x_c + w_half
    scaled_box[1] = y_c - h_half
    scaled_box[3] = y_c + h_half
    return scaled_box

def _paste_mask(box, mask, shape):
    """
    Args:
        box: 4 float
        mask: MxM floats
        shape: h,w
    Returns:
        A uint8 binary image of hxw.
    """
    
    mask = np.pad(mask, [(1, 1), (1, 1)], mode='constant')
    box = _scale_box(box, float(mask.shape[0]) / (mask.shape[0] - 2))

    mask_pixels = np.arange(0.0, mask.shape[0]) + 0.5
    mask_continuous = interpolate.interp2d(mask_pixels, mask_pixels, mask, fill_value=0.0)
    h, w = shape
    ys = np.arange(0.0, h) + 0.5
    xs = np.arange(0.0, w) + 0.5
    ys = (ys - box[1]) / (box[3] - box[1]) * mask.shape[0]
    xs = (xs - box[0]) / (box[2] - box[0]) * mask.shape[1]
    res = mask_continuous(xs, ys)
    return (res >= 0.5).astype('uint8')


def predict_image(img, model_func):
    """
    Run detection on one image, using the TF callable.
    This function should handle the preprocessing internally.

    Args:
        img: an image
        model_func: a callable from the TF model.
            It takes image and returns (boxes, probs, labels, [masks])

    Returns:
        [DetectionResult]
    """

    orig_shape = img.shape[:2]
    resizer = CustomResize(cfg.PREPROC.TEST_SHORT_EDGE_SIZE, cfg.PREPROC.MAX_SIZE)
    resized_img = resizer.augment(img)
    scale = np.sqrt(resized_img.shape[0] * 1.0 / img.shape[0] * resized_img.shape[1] / img.shape[1])
    boxes, probs, labels, *masks = model_func(resized_img)
    boxes = boxes / scale
    # boxes are already clipped inside the graph, but after the floating point scaling, this may not be true any more.
    boxes = clip_boxes(boxes, orig_shape)

    if masks:
        # has mask
        full_masks = [_paste_mask(box, mask, orig_shape)
                      for box, mask in zip(boxes, masks[0])]
        masks = full_masks
    else:
        # fill with none
        masks = [None] * len(boxes)

    results = [DetectionResult(*args) for args in zip(boxes, probs, labels, masks)]
    return results

def predict_image_batch(img_batch, model_func, resized_sizes, scales, orig_sizes):
    """
    Run detection on one image, using the TF callable.
    This function should handle the preprocessing internally.

    Args:
        img: an image
        model_func: a callable from the TF model.
            It takes image and returns (boxes, probs, labels, [masks])

    Returns:
        [DetectionResult]
    """

    resized_sizes = np.stack(resized_sizes)
    resized_sizes_in = np.concatenate((resized_sizes, 3*np.ones((resized_sizes.shape[0], 1))), axis=1)

    indices, boxes, probs, labels, *masks = model_func(img_batch, resized_sizes_in)

    results = []
    for i in range(len(scales)):
        ind = np.where(indices.astype(np.int32) == i)[0]

        if len(ind) > 0:
            boxes[ind, :] = boxes[ind, :]/scales[i]

            # boxes are already clipped inside the graph, but after the floating point scaling, this may not be true any more.
            boxes[ind, :] = clip_boxes(boxes[ind, :], orig_sizes[i])

        if masks and len(ind) > 0:
           # has mask
           full_masks = [_paste_mask(box, mask, orig_sizes[i])
                      for box, mask in zip(boxes[ind,:], masks[0][ind,:])]
           masks = full_masks
        else:
           # fill with none
           masks = [None] * len(boxes[ind,:])

    results.append([DetectionResult(*args) for args in zip(boxes, probs, labels, masks)])
    return results


def predict_dataflow(df, model_func, tqdm_bar=None):
    """
    Args:
        df: a DataFlow which produces (image, image_id)
        model_func: a callable from the TF model.
            It takes image and returns (boxes, probs, labels, [masks])
        tqdm_bar: a tqdm object to be shared among multiple evaluation instances. If None,
            will create a new one.
    Returns:
        list of dict, in the format used by
        `DetectionDataset.eval_or_save_inference_results`
    """
    df.reset_state()
    all_results = []
    with ExitStack() as stack:
        # tqdm is not quite thread-safe: https://github.com/tqdm/tqdm/issues/323
        if tqdm_bar is None:
            tqdm_bar = stack.enter_context(get_tqdm(total=df.size()))
        for img, img_id in df:
            results = predict_image(img, model_func)
            for r in results:

                img_id = int(img_id)
                class_id = int(r.class_id)
                bbox = list([float(b) for b in r.box])
                score = round(float(r.score), 4)

                res = {
                    'image_id': img_id,
                    'category_id': class_id,
                    'bbox': bbox,
                    'score': score,
                }

                # also append segmentation to results
                if r.mask is not None:
                    rle = cocomask.encode(
                        np.array(r.mask[:, :, None], order='F'))[0]
                    rle['counts'] = rle['counts'].decode('ascii')
                    res['segmentation'] = rle
                all_results.append(res)
            tqdm_bar.update(1)
    return all_results



def predict_dataflow_batch(df, model_func, tqdm_bar=None):
    """
    Args:
        df: a DataFlow which produces (image, image_id)
        model_func: a callable from the TF model.
            It takes image and returns (boxes, probs, labels, [masks])
        tqdm_bar: a tqdm object to be shared among multiple evaluation instances. If None,
            will create a new one.

    Returns:
        list of dict, in the format used by
        `DetectionDataset.eval_or_save_inference_results`
    """
    df.reset_state()
    all_results = []
    with ExitStack() as stack:
        # tqdm is not quite thread-safe: https://github.com/tqdm/tqdm/issues/323
        if tqdm_bar is None:
            tqdm_bar = stack.enter_context(get_tqdm(total=df.size()))
        for imgs, img_ids, resized_sizes, scales, orig_sizes in df:
            results = predict_image_batch(imgs, model_func, resized_sizes, scales, orig_sizes)
            batch_id = 0
            for img_results in results:
                for r in img_results:

                    img_id = int(img_ids[batch_id])
                    class_id = int(r.class_id)
                    bbox = list([float(b) for b in r.box])
                    score = round(float(r.score), 4)

                    res = {
                        'image_id': img_id,
                        'category_id': class_id,
                        'bbox': bbox,
                        'score': score,
                    }

                    # also append segmentation to results
                    if r.mask is not None:
                        rle = cocomask.encode(
                            np.array(r.mask[:, :, None], order='F'))[0]
                        rle['counts'] = rle['counts'].decode('ascii')
                        res['segmentation'] = rle
                    all_results.append(res)
                batch_id += 1
            tqdm_bar.update(1)
    return all_results


def multithread_predict_dataflow(dataflows, model_funcs, batched=True):
    """
    Running multiple `predict_dataflow` in multiple threads, and aggregate the results.

    Args:
        dataflows: a list of DataFlow to be used in :func:`predict_dataflow`
        model_funcs: a list of callable to be used in :func:`predict_dataflow`

    Returns:
        list of dict, in the format used by
        `DetectionDataset.eval_or_save_inference_results`
    """
    num_worker = len(model_funcs)
    assert len(dataflows) == num_worker
    if num_worker == 1:
        if batched:
            local_results = predict_dataflow_batch(dataflows[0], model_funcs[0])
        else:
            local_results = predict_dataflow(dataflows[0], model_funcs[0])
        return local_results
    kwargs = {'thread_name_prefix': 'EvalWorker'} if sys.version_info.minor >= 6 else {}
    with ThreadPoolExecutor(max_workers=num_worker, **kwargs) as executor, \
            tqdm.tqdm(total=sum([df.size() for df in dataflows])) as pbar:
        futures = []
        for dataflow, pred in zip(dataflows, model_funcs):
            if batched:
                futures.append(executor.submit(predict_dataflow_batch, dataflow, pred, pbar))
            else:
                futures.append(executor.submit(predict_dataflow, dataflow, pred, pbar))
        all_results = list(itertools.chain(*[fut.result() for fut in futures]))
        return all_results

def gather_result_from_all_processes(local_results, root=0):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    res = comm.gather(local_results,root=root)
    return res


class AsyncEvaluator():
    '''
    An async evaluator used to submit coco evaluation job to a background thread

    Usage:
    1. create the worker with: worker = AsyncEvaluator()
    2. submit the job: work.submit_task(tag, background_task_fn, fn_inputs)
    '''
    def __init__(self, num_threads=1, device=None):
        self.num_threads = num_threads
        self.pool = ThreadPoolExecutor(num_threads)
        self.events = {}

    def submit_task(self, tag, fn, *args, **kwargs):
        e = self.pool.submit(fn, *args, **kwargs)
        self.events[tag] = e

    def task_done(self, tag):
        if tag in self.events.keys():
            return self.events[tag].done()
        else:
            return False


class EvalCallback(Callback):
    """
    A callback that runs evaluation once a while.
    It supports multi-gpu evaluation.
    """

    _chief_only = False

    def __init__(self, eval_dataset, in_names, out_names, output_dir, batch_size):
        self._eval_dataset = eval_dataset
        self._in_names, self._out_names = in_names, out_names
        self._output_dir = output_dir
        self.batched = batch_size > 0
        self.batch_size = batch_size

    def _setup_graph(self):
        num_gpu = cfg.TRAIN.NUM_GPUS
        if cfg.TRAINER == 'replicated':
            self.num_predictor = num_gpu
            self.predictors = [self._build_predictor(k % num_gpu) for k in range(self.num_predictor)]
            if self.batched:
                self.dataflows = [get_batched_eval_dataflow(self._eval_dataset,
                                                shard=k, num_shards=self.num_predictor, 
                                                batch_size=self.batch_size)
                              for k in range(self.num_predictor)]
            else:
                self.dataflows = [get_eval_dataflow(self._eval_dataset,
                                                shard=k, num_shards=self.num_predictor)
                              for k in range(self.num_predictor)]
        else:
            # Eval on all ranks and use gather
            self.predictor = self._build_predictor(0)

            if self.batched:
                self.dataflow = get_batched_eval_dataflow(self._eval_dataset,
                                              shard=hvd.rank(), num_shards=hvd.size(), batch_size=self.batch_size)
            else:
                self.dataflow = get_eval_dataflow(self._eval_dataset,
                                              shard=hvd.rank(), num_shards=hvd.size())


    def _build_predictor(self, idx):
        return self.trainer.get_predictor(self._in_names, self._out_names, device=idx)

    def _before_train(self):
        eval_period = cfg.TRAIN.EVAL_PERIOD
        self.epochs_to_eval = set()
        for k in itertools.count(1):
            if k * eval_period > self.trainer.max_epoch:
                break
            self.epochs_to_eval.add(k * eval_period)
        self.epochs_to_eval.add(self.trainer.max_epoch)
        logger.info("[EvalCallback] Will evaluate every {} epochs".format(eval_period))

    def _eval(self):
        logdir = self._output_dir
        if cfg.TRAINER == 'replicated':
            all_results = multithread_predict_dataflow(self.dataflows, self.predictors, self.batched)
        else:
            if self.batched:
                local_results = predict_dataflow_batch(self.dataflow, self.predictor)
            else:
                local_results = predict_dataflow(self.dataflow, self.predictor)

            results = gather_result_from_all_processes(local_results)
            if hvd.rank() > 0:
                return
            all_results = []
            for item in results:
                if item is not None:
                    all_results.extend(item)

        output_file = os.path.join(
            logdir, '{}-outputs{}'.format(self._eval_dataset, self.global_step))

        scores = DetectionDataset().eval_or_save_inference_results(
            all_results, self._eval_dataset, output_file)
        for k, v in scores.items():
            self.trainer.monitors.put_scalar(k, v)

    def _trigger_epoch(self):
        if self.epoch_num in self.epochs_to_eval:
            logger.info("Running evaluation ...")
            self._eval()


class AsyncEvalCallback(Callback):
    """
    A callback that runs evaluation once a while.
    It supports multi-gpu evaluation.

    Supoort the async evaluation:
    1. Running the graph on all gpus and gather the result on the master node
    2. Running a background thread to do the coco evaluation
    """

    _chief_only = False

    def __init__(self, eval_dataset, in_names, out_names, output_dir, batch_size):
        self._eval_dataset = eval_dataset
        self._in_names, self._out_names = in_names, out_names
        self._output_dir = output_dir
        self.batched = batch_size > 0
        self.batch_size = batch_size

    def _setup_graph(self):
        num_gpu = cfg.TRAIN.NUM_GPUS
        if cfg.TRAINER == 'replicated':
            self.num_predictor = num_gpu
            self.predictors = [self._build_predictor(k % num_gpu) for k in range(self.num_predictor)]
            self.dataflows = [get_eval_dataflow(self._eval_dataset,
                                                shard=k, num_shards=self.num_predictor)
                              for k in range(self.num_predictor)]
        else:
            # Eval on all ranks and use gather
            self.predictor = self._build_predictor(0)

            if self.batched:
                self.dataflow = get_batched_eval_dataflow(self._eval_dataset,
                                              shard=hvd.rank(), num_shards=hvd.size(), batch_size=self.batch_size)
            else:
                self.dataflow = get_eval_dataflow(self._eval_dataset,
                                              shard=hvd.rank(), num_shards=hvd.size())


    def _build_predictor(self, idx):
        return self.trainer.get_predictor(self._in_names, self._out_names, device=idx)

    def _before_train(self):
        if hvd.rank() == 0:
            self.worker = AsyncEvaluator()
        eval_period = cfg.TRAIN.EVAL_PERIOD
        self.epochs_to_eval = set()
        for k in itertools.count(1):
            if k * eval_period > self.trainer.max_epoch:
                break
            self.epochs_to_eval.add(k * eval_period)
        self.epochs_to_eval.add(self.trainer.max_epoch)
        logger.info("[EvalCallback] Will evaluate every {} epochs".format(eval_period))

    def _eval(self):
        logdir = self._output_dir
        if cfg.TRAINER == 'replicated':
            all_results = multithread_predict_dataflow(self.dataflows, self.predictors, self.batched)
        else:
            if self.batched:
                local_results = predict_dataflow_batch(self.dataflow, self.predictor)
            else:
                local_results = predict_dataflow(self.dataflow, self.predictor)

            results = gather_result_from_all_processes(local_results)
            if hvd.rank() > 0:
                return
            all_results = []
            for item in results:
                if item is not None:
                    all_results.extend(item)

        def background_coco(all_results):
            output_file = os.path.join(
                logdir, '{}-outputs{}'.format(self._eval_dataset, self.global_step))
            scores = DetectionDataset().eval_or_save_inference_results(
                all_results, self._eval_dataset, output_file)
            cfg.TRAIN.SHOULD_STOP = scores['mAP(bbox)/IoU=0.5:0.95'] >= cfg.TEST.BOX_TARGET and scores['mAP(segm)/IoU=0.5:0.95'] >= cfg.TEST.MASK_TARGET
            for k, v in scores.items():
                self.trainer.monitors.put_scalar(k, v)
            return

        self.worker.submit_task(f"eval_{self.epoch_num}", background_coco, all_results)

    def _trigger_epoch(self):
        if self.epoch_num in self.epochs_to_eval:
            logger.info("Running evaluation ...")
            self._eval()
