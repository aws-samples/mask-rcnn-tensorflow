# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import tensorpack
import time
import tensorflow as tf

def humanize_float(num):
    return "{0:,.2f}".format(num)



def summarize_tensor(tensor_name, tensor_to_summarize, trigger_tensor, additional_print_val=None):
    t_shape = tf.shape(tensor_to_summarize)
    t_size = tf.size(tensor_to_summarize)

    t_sizerange = tf.range(t_size)
    t_sizerange_float = tf.cast(t_sizerange, dtype=tensor_to_summarize.dtype)
    t_reshaped = tf.reshape(t_sizerange_float, shape=t_shape)
    t_mult = tf.multiply(t_reshaped, tensor_to_summarize)
    summary = tf.reduce_sum(t_mult)
    if additional_print_val is None:
        print_op = tf.print(tensor_name, summary, summarize=-1)
    else:
        print_op = tf.print(tensor_name, summary, additional_print_val, summarize=-1)
    with tf.control_dependencies([print_op]):
        return tf.identity(trigger_tensor)









# Checks at graph_build time
def print_buildtime_shape(name, tensor, prefix=None):
    if prefix is not None:
        prefix = f' [{prefix}]'
    else:
        prefix = ""

    print(f'[buildtime_shape]{prefix} {name}: {tensor.shape}')




def print_runtime_shape(name, tensor, prefix=None):
    s = "[runtime_shape] "
    if prefix is not None:
        s += f'[{prefix}] '
    s += f'{name}: '
    return runtime_print([s, tf.shape(tensor)], tensor)





# A method if you want tf.print to behave like tf.Print (i.e. the 'print' exists as an op in the computation graph)
"""
some_tensor = tf.op(some_other_tensor)
some_tensor = runtime_print("String to print", some_tensor)
"""
def runtime_print(message, trigger_tensor):
    print_op = tf.print(message)
    with tf.control_dependencies([print_op]):
        return tf.identity(trigger_tensor)


def runtime_print_str(message_str, trigger_tensor, prefix=None):
    if prefix is not None:
        message_str = f'[{prefix}] {message_str}'

    return runtime_print(message_str, trigger_tensor)



"""
some_tensor = print_runtime_tensor("some_tensor", some_tensor, prefix="example_fn")
"""
def print_runtime_tensor(name, tensor, prefix=None, summarize=-1):
    s = "[runtime_tensor] "
    if prefix is not None:
        s += f'[{prefix}] '
    s += name

    print_op = tf.print(s, tensor, summarize=summarize)
    with tf.control_dependencies([print_op]):
        return tf.identity(tensor)





"""
trigger_tensor = print_runtime_tensor_loose_branch("tensor_to_examine", tensor_to_examine, prefix="example_fn", trigger_tensor=trigger_tensor)
Print a tensor, even if the tensor is not used by the graph. Useful when you want to transform and print a tensor to
examine it, but the transformed tensor is not used in the actual graph so the transform+print is not executed.
"""
def print_runtime_tensor_loose_branch(name, tensor, prefix=None, summarize=-1, trigger_tensor=None):
    assert trigger_tensor is not None

    s = "[runtime_tensor_freehanging_branch] "
    if prefix is not None:
        s += f'[{prefix}] '
    s += name

    print_op = tf.print(s, tensor, summarize=summarize)
    with tf.control_dependencies([print_op]):
        return tf.identity(trigger_tensor)




class ThroughputTracker(tensorpack.Callback):
    """
    Calculate and display throughput of model, by keeping track of the duration of each step and each epoch. Saves and
    outputs throughput as items/second. Prints output and saves
    Args:
        items_per_step:         The number of items processed in each step
        items_per_epoch:        The number of items processed in each epoch
        trigger_every_n_steps:  If this argument is None, throughput will be calculated once per epoch. If this argument
                                is a number N, throughput will also be calculated and output every N steps. The step
                                counter starts over each epoch.
        log_fn:                 The function to call to display throughput in logs. If None, throughput
                                will not be printed. This argument does not impact saving throughput as tf.scalar
    """

    def __init__(self, items_per_step, items_per_epoch, trigger_every_n_steps=None, log_fn=None):
        self._items_per_step = items_per_step
        self._items_per_epoch = items_per_epoch
        self._trigger_every_n_steps = trigger_every_n_steps

        if log_fn is None:
            self._log_fn = lambda x: None    # Do nothing logger
        else:
            self._log_fn = log_fn

        self._step_counter = 0

    def _before_epoch(self):
        epoch_start_time = time.time()
        self._epoch_start_time = epoch_start_time
        self._step_start_time = epoch_start_time
        self._epoch_step_durations = []
        self._step_counter = 0



    def _trigger_step(self):
        self._step_end_time = time.time()
        step_duration = self._step_end_time - self._step_start_time
        self._epoch_step_durations.append(step_duration)

        if self._trigger_every_n_steps is not None:
            self._step_counter += 1
            if self._step_counter % self._trigger_every_n_steps == 0:
                sum_step_durations = sum(self._epoch_step_durations[-self._trigger_every_n_steps:]) / self._trigger_every_n_steps
                mean_step_duration = sum_step_durations

                log_prefix = f'[ThroughputTracker] Over last {self._trigger_every_n_steps} steps'
                self._log_fn(f'{log_prefix}, MeanDuration={humanize_float(mean_step_duration)} seconds')
                self._log_fn(f'{log_prefix}, MeanThroughput={humanize_float(self._items_per_step / mean_step_duration)} items/sec')
                self._step_counter = 0

        self._step_start_time = self._step_end_time

    def _after_epoch(self):
        self._epoch_end_time = time.time()

    def _trigger_epoch(self):
        epoch_run_clock_time = sum(self._epoch_step_durations)
        epoch_wall_clock_time = self._epoch_end_time - self._epoch_start_time

        overhead_time = epoch_wall_clock_time - epoch_run_clock_time
        mean_epoch_throughput = self._items_per_epoch / epoch_wall_clock_time

        log_prefix = "[ThroughputTracker] Over last epoch"
        self._log_fn(f'{log_prefix}, MeanEpochThroughput: {humanize_float(mean_epoch_throughput)}')
        self._log_fn(f'{log_prefix}, EpochWallClockDuration: {humanize_float(epoch_wall_clock_time)}')
        self._log_fn(f'{log_prefix}, CallbackOverheadDuration: {humanize_float(overhead_time)}')

        self.trainer.monitors.put_scalar("Throughput/MeanEpochThroughput", mean_epoch_throughput)
        self.trainer.monitors.put_scalar('Throughput/EpochWallClockDuration', epoch_wall_clock_time)
        self.trainer.monitors.put_scalar('Throughput/CallbackOverheadDuration', overhead_time)
