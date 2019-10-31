# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: train.py

import argparse
import itertools
import numpy as np
import shutil
import cv2
import six
assert six.PY3, "FasterRCNN requires Python 3!"
import tensorflow as tf
import tqdm
import time
import subprocess

import tensorpack.utils.viz as tpviz
from tensorpack import *
from tensorpack.tfutils.common import get_tf_version_tuple


from dataset import DetectionDataset
from config import finalize_configs, config as cfg
from data import get_eval_dataflow, get_train_dataflow, get_batch_train_dataflow
from eval import DetectionResult, predict_image, multithread_predict_dataflow, EvalCallback, AsyncEvalCallback
from viz import draw_annotation, draw_final_outputs, draw_predictions, draw_proposal_recall
from performance import ThroughputTracker, humanize_float
from model.generalized_rcnn import ResNetFPNModel
from tensorpack.utils import fix_rng_seed


try:
    import horovod.tensorflow as hvd
except ImportError:
    pass

def do_visualize(model, model_path, nr_visualize=100, output_dir='output'):
    """
    Visualize some intermediate results (proposals, raw predictions) inside the pipeline.
    """
    df = get_train_dataflow()   # we don't visualize mask stuff
    df.reset_state()

    pred = OfflinePredictor(PredictConfig(
        model=model,
        session_init=get_model_loader(model_path),
        input_names=['images', 'gt_boxes', 'gt_labels'],
        output_names=[
            'generate_{}_proposals/boxes'.format('fpn' if cfg.MODE_FPN else 'rpn'),
            'generate_{}_proposals/scores'.format('fpn' if cfg.MODE_FPN else 'rpn'),
            'fastrcnn_all_scores',
            'output/boxes',
            'output/scores',
            'output/labels',
        ]))

    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    utils.fs.mkdir_p(output_dir)
    with tqdm.tqdm(total=nr_visualize) as pbar:
        for idx, dp in itertools.islice(enumerate(df), nr_visualize):
            img, gt_boxes, gt_labels = dp['images'], dp['gt_boxes'], dp['gt_labels']

            rpn_boxes, rpn_scores, all_scores, \
                final_boxes, final_scores, final_labels = pred(img, gt_boxes, gt_labels)

            # draw groundtruth boxes
            gt_viz = draw_annotation(img, gt_boxes, gt_labels)
            # draw best proposals for each groundtruth, to show recall
            proposal_viz, good_proposals_ind = draw_proposal_recall(img, rpn_boxes, rpn_scores, gt_boxes)
            # draw the scores for the above proposals
            score_viz = draw_predictions(img, rpn_boxes[good_proposals_ind], all_scores[good_proposals_ind])

            results = [DetectionResult(*args) for args in
                       zip(final_boxes, final_scores, final_labels,
                           [None] * len(final_labels))]
            final_viz = draw_final_outputs(img, results)

            viz = tpviz.stack_patches([
                gt_viz, proposal_viz,
                score_viz, final_viz], 2, 2)

            if os.environ.get('DISPLAY', None):
                tpviz.interactive_imshow(viz)
            cv2.imwrite("{}/{:03d}.png".format(output_dir, idx), viz)
            pbar.update()


def do_evaluate(pred_config, output_file):
    num_gpu = cfg.TRAIN.NUM_GPUS
    graph_funcs = MultiTowerOfflinePredictor(
        pred_config, list(range(num_gpu))).get_predictors()

    for dataset in cfg.DATA.VAL:
        logger.info("Evaluating {} ...".format(dataset))
        dataflows = [
            get_eval_dataflow(dataset, shard=k, num_shards=num_gpu)
            for k in range(num_gpu)]
        all_results = multithread_predict_dataflow(dataflows, graph_funcs)
        output = output_file + '-' + dataset
        DetectionDataset().eval_or_save_inference_results(all_results, dataset, output)


def do_predict(pred_func, input_file):
    img = cv2.imread(input_file, cv2.IMREAD_COLOR)
    results = predict_image(img, pred_func)
    final = draw_final_outputs(img, results)
    viz = np.concatenate((img, final), axis=1)
    cv2.imwrite("output.png", viz)
    logger.info("Inference output written to output.png")
    tpviz.interactive_imshow(viz)





def log_launch_config(log_full_git_diff):
    def check_and_log(cmd):
        logger.info(cmd)
        logger.info(subprocess.check_output(cmd, shell=True).decode("utf-8"))

    check_and_log('git status') # branch and changes
    check_and_log('git rev-parse HEAD') # commit
    if log_full_git_diff:
        check_and_log('git diff')

    check_and_log('env')
    check_and_log('ps -elf | grep mpirun')

def call_only_once(func):
    """
    Decorate a method or property of a class, so that this method can only
    be called once for every instance.
    Calling it more than once will result in exception.
    """
    import inspect
    if six.PY2:
        import functools32 as functools
    else:
        import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        self = args[0]
        # cannot use hasattr here, because hasattr tries to getattr, which
        # fails if func is a property
        assert func.__name__ in dir(self), "call_only_once can only be used on method or property!"

        if not hasattr(self, '_CALL_ONLY_ONCE_CACHE'):
            cache = self._CALL_ONLY_ONCE_CACHE = set()
        else:
            cache = self._CALL_ONLY_ONCE_CACHE

        cls = type(self)
        # cannot use ismethod(), because decorated method becomes a function
        is_method = inspect.isfunction(getattr(cls, func.__name__))
        assert func not in cache, \
            "{} {}.{} can only be called once per object!".format(
                'Method' if is_method else 'Property',
                cls.__name__, func.__name__)
        cache.add(func)

        return func(*args, **kwargs)

    return wrapper

class AsyncHorovodTrainer(HorovodTrainer):
    '''
    A wrapper of the HorovodTrainer, will stop the training once the target accuracy is reached.
    '''
    def __init__(self, average=True, compression=None):
        super(AsyncHorovodTrainer, self).__init__(average=average, compression=compression)

    @call_only_once
    def main_loop(self, steps_per_epoch, starting_epoch, max_epoch):
        """
        Run the main training loop.

        Args:
            steps_per_epoch, starting_epoch, max_epoch (int):
        """
        with self.sess.as_default():
            self.loop.config(steps_per_epoch, starting_epoch, max_epoch)
            self.loop.update_global_step()
            try:
                self._callbacks.before_train()
                # refresh global step (might have changed by callbacks) TODO ugly
                # what if gs is changed later?
                self.loop.update_global_step()
                for self.loop._epoch_num in range(
                        self.loop.starting_epoch, self.loop.max_epoch + 1):
                    logger.info("Start Epoch {} ...".format(self.loop.epoch_num))
                    self._callbacks.before_epoch()
                    start_time = time.time()
                    for self.loop._local_step in range(self.loop.steps_per_epoch):
                        if self.hooked_sess.should_stop():
                            return
                        if cfg.TRAIN.SHOULD_STOP:
                            logger.info("Target accuracy has been reached, stop.....")
                            return
                        self.run_step()  # implemented by subclass
                        self._callbacks.trigger_step()
                    self._callbacks.after_epoch()
                    logger.info("Epoch {} (global_step {}) finished, time:{}.".format(
                        self.loop.epoch_num, self.loop.global_step, humanize_time_delta(time.time() - start_time)))

                    # trigger epoch outside the timing region.
                    self._callbacks.trigger_epoch()
                logger.info("Training has finished!")
            except (StopTraining, tf.errors.OutOfRangeError) as e:
                logger.info("Training was stopped by exception {}.".format(str(e)))
            except KeyboardInterrupt:
                logger.info("Detected Ctrl-C and exiting main loop.")
                raise
            finally:
                self._callbacks.after_train()
                self.hooked_sess.close()




if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', help='load a model for evaluation or training. Can overwrite BACKBONE.WEIGHTS')
    parser.add_argument('--logdir', help='log directory', default='train_log/maskrcnn')
    parser.add_argument('--visualize', action='store_true', help='visualize intermediate results')
    parser.add_argument('--evaluate', help="Run evaluation. "
                                           "This argument is the path to the output json evaluation file")
    parser.add_argument('--predict', help="Run prediction on a given image. "
                                          "This argument is the path to the input image file")
    parser.add_argument('--config', help="A list of KEY=VALUE to overwrite those defined in config.py",
                        nargs='+')
    parser.add_argument('--fp16', help="Train in FP16", action="store_true")
    parser.add_argument('--async_eval', help="Enable async evaluation, for mlperf. Evaluation will run every epoch and training will be stopped once the target is hit", action="store_true")
    #################################################################################################################
    # Performance investigation arguments
    parser.add_argument('--throughput_log_freq', help="In perf investigation mode, code will print throughput after every throughput_log_freq steps as well as after every epoch", type=int, default=100)
    parser.add_argument('--images_per_epoch', help="Number of images in an epoch. = images_per_steps * steps_per_epoch (differs slightly from the total number of images).", type=int, default=120000)

    parser.add_argument('--tfprof', help="Enable tf profiler", action="store_true")
    parser.add_argument('--tfprof_start_step', help="Step to enable tf profiling", type=int, default=15005)
    parser.add_argument('--tfprof_end_step', help="Step after which tf profiling will be disabled", type=int, default=15010)

    parser.add_argument('--log_full_git_diff', help="Log the full git diff", action="store_false")


    #################################################################################################################




    if get_tf_version_tuple() < (1, 6):
        # https://github.com/tensorflow/tensorflow/issues/14657
        logger.warn("TF<1.6 has a bug which may lead to crash in FasterRCNN if you're unlucky.")

    args = parser.parse_args()
    if args.config:
        cfg.update_args(args.config)

    MODEL = ResNetFPNModel(args.fp16)
    DetectionDataset()  # initialize the config with information from our dataset



    if args.visualize or args.evaluate or args.predict:
        assert tf.test.is_gpu_available()
        assert args.load
        finalize_configs(is_training=False)

        if args.predict or args.visualize:
            cfg.TEST.RESULT_SCORE_THRESH = cfg.TEST.RESULT_SCORE_THRESH_VIS

        if args.visualize:
            do_visualize(MODEL, args.load)
        else:
            predcfg = PredictConfig(
                model=MODEL,
                session_init=get_model_loader(args.load),
                input_names=MODEL.get_inference_tensor_names()[0],
                output_names=MODEL.get_inference_tensor_names()[1])
            if args.predict:
                do_predict(OfflinePredictor(predcfg), args.predict)
            elif args.evaluate:
                assert args.evaluate.endswith('.json'), args.evaluate
                do_evaluate(predcfg, args.evaluate)



    else:

        is_horovod = cfg.TRAINER == 'horovod'
        if args.async_eval:
            assert is_horovod, "Async evaluation only support Horovod based trainer"

        if is_horovod:
            hvd.init()
            logger.info("Horovod Rank={}, Size={}".format(hvd.rank(), hvd.size()))

        if not is_horovod or hvd.rank() == 0:
            logger.set_logger_dir(args.logdir, 'd')
            log_launch_config(args.log_full_git_diff)

        finalize_configs(is_training=True)

        # Set evaluation every epoch for async eval mode
        if args.async_eval:
            cfg.TRAIN.EVAL_PERIOD = 1

        if cfg.TRAIN.SEED:
            tf.set_random_seed(cfg.TRAIN.SEED)
            fix_rng_seed(cfg.TRAIN.SEED*hvd.rank())
            np.random.seed(cfg.TRAIN.SEED)

        images_per_step = cfg.TRAIN.NUM_GPUS * cfg.TRAIN.BATCH_SIZE_PER_GPU
        steps_per_epoch = args.images_per_epoch // images_per_step
        batch_size_lr_factor = images_per_step # The LR is defined for bs=1 and then scaled linearly with the batch size
        base_lr_adjusted_for_bs = cfg.TRAIN.BASE_LR * batch_size_lr_factor

        # Warmup LR schedule is step based
        warmup_start_step = 0
        warmup_end_step = cfg.TRAIN.WARMUP_STEPS
        warmup_start_lr = cfg.TRAIN.WARMUP_INIT_LR*8
        warmup_end_lr = base_lr_adjusted_for_bs
        warmup_schedule = [(warmup_start_step, warmup_start_lr), (warmup_end_step, warmup_end_lr)]


        # Training LR schedule is epoch based
        warmup_end_epoch = cfg.TRAIN.WARMUP_STEPS * 1. / steps_per_epoch
        training_start_epoch = int(warmup_end_epoch + 0.5)
        lr_schedule = [(training_start_epoch, base_lr_adjusted_for_bs)]


        max_epoch = None
        for epoch, scheduled_lr_multiplier in cfg.TRAIN.LR_EPOCH_SCHEDULE:
            if scheduled_lr_multiplier is None:
                max_epoch = epoch # Training end is indicated by a lr_multiplier of None
                break

            absolute_lr = base_lr_adjusted_for_bs * scheduled_lr_multiplier
            lr_schedule.append((epoch, absolute_lr))


        logger.info("Warm Up Schedule (steps, value): " + str(warmup_schedule))
        logger.info("LR Schedule (epochs, value): " + str(lr_schedule))


        train_dataflow = get_batch_train_dataflow(cfg.TRAIN.BATCH_SIZE_PER_GPU)


        callbacks = [
            PeriodicCallback(
                ModelSaver(max_to_keep=10, keep_checkpoint_every_n_hours=1),
                every_k_epochs=20),
            # linear warmup
            ScheduledHyperParamSetter(
                'learning_rate', warmup_schedule, interp='linear', step_based=True),
            ScheduledHyperParamSetter('learning_rate', lr_schedule),
            PeakMemoryTracker(),
            EstimatedTimeLeft(median=True),
            SessionRunTimeout(60000).set_chief_only(True),   # 1 minute timeout
        ]

        if args.async_eval:
            callbacks.extend([
                AsyncEvalCallback(dataset, *MODEL.get_inference_tensor_names(), args.logdir, 1) #cfg.TRAIN.BATCH_SIZE_PER_GPU)
                for dataset in cfg.DATA.VAL
            ])
        else:
            callbacks.extend([
                EvalCallback(dataset, *MODEL.get_inference_tensor_names(), args.logdir, 1) #cfg.TRAIN.BATCH_SIZE_PER_GPU)
                for dataset in cfg.DATA.VAL
            ])

        if not is_horovod:
            callbacks.append(GPUUtilizationTracker())

        callbacks.append(ThroughputTracker(cfg.TRAIN.BATCH_SIZE_PER_GPU*cfg.TRAIN.NUM_GPUS,
                                           args.images_per_epoch,
                                           trigger_every_n_steps=args.throughput_log_freq,
                                           log_fn=logger.info))

        if args.tfprof:
            # We only get tf profiling chrome trace on rank==0
            if hvd.rank() == 0:
                callbacks.append(EnableCallbackIf(
                    GraphProfiler(dump_tracing=True, dump_event=True),
                    lambda self: self.trainer.global_step >= args.tfprof_start_step and self.trainer.global_step <= args.tfprof_end_step))

        if is_horovod and hvd.rank() > 0:
            session_init = None
        else:
            if args.load:
                session_init = get_model_loader(args.load)
            else:
                session_init = get_model_loader(cfg.BACKBONE.WEIGHTS) if cfg.BACKBONE.WEIGHTS else None


        traincfg = TrainConfig(
            model=MODEL,
            data=QueueInput(train_dataflow),
            callbacks=callbacks,
            extra_callbacks=[
               MovingAverageSummary(),
               ProgressBar(),
               MergeAllSummaries(period=250),
               RunUpdateOps()
            ],
            steps_per_epoch=steps_per_epoch,
            max_epoch=max_epoch,
            session_init=session_init,
            session_config=None,
            starting_epoch=cfg.TRAIN.STARTING_EPOCH
        )

        if args.async_eval:
            trainer = AsyncHorovodTrainer(average=True)
        elif is_horovod:
            trainer = HorovodTrainer(average=True)
        else:
            # nccl mode appears faster than cpu mode
            trainer = SyncMultiGPUTrainerReplicated(cfg.TRAIN.NUM_GPUS, average=True, mode='nccl')
        launch_train_with_config(traincfg, trainer)

    training_duration_secs = time.time() - start_time
    logger.info(f'Total duration: {humanize_float(training_duration_secs)}')
