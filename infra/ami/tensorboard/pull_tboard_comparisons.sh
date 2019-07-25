# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env bash

# A script to download old tensorboard event files to compare against. Downloads an example of successful convergence,
# an example of non-successful convergence and all of the 'convergence' codebase runs. Allows Tensorboard to display
# all runs together if the tensorboard.sh command is used

# Use in VM, not inside docker container

rm -r ~/old_logs
mkdir -p ~/old_logs


aws s3 cp --recursive s3://aws-tensorflow-benchmarking/maskrcnn/results/nobatch_notconverging_20190315_t1 ~/old_logs/nobatch_notconverging_20190315_t1
aws s3 cp --recursive s3://aws-tensorflow-benchmarking/maskrcnn/results/cuda10_baseline_converge_2019021_3030540 ~/old_logs/cuda10_baseline_converge_2019021_3030540
aws s3 cp --recursive s3://aws-tensorflow-benchmarking/maskrcnn/results/convergence_codebase_iso_baseline_20190319 ~/old_logs/convergence_codebase_iso_baseline_20190319


################################################################################################################################
# Batch-by-default
################################################################################################################################

# After first code cleanup, PR #21 (commit c7ecdc029cecb967c4aee08f7c5a6b1ccd4b2c15)
aws s3 cp --recursive s3://aws-tensorflow-benchmarking/maskrcnn/results/batch-by-default/master_24epoch_20190410 ~/old_logs/batch-by-default/master_24epoch_p3dn_20190410



################################################################################################################################
# Convergence isolation codebase -
################################################################################################################################
aws s3 cp --recursive s3://aws-tensorflow-benchmarking/maskrcnn/results/convergence_codebase_iso_baseline_20190319 ~/old_logs/convergence_codebase_iso_baseline_20190319
#aws s3 cp --recursive s3://aws-tensorflow-benchmarking/maskrcnn/results/convergence_codebase_iso_baseline_20190320 ~/old_logs/convergence_codebase_iso_baseline_20190320
#aws s3 cp --recursive s3://aws-tensorflow-benchmarking/maskrcnn/results/convergence_codebase_iso_baseline_20190322 ~/old_logs/convergence_codebase_iso_baseline_20190322
#aws s3 cp --recursive s3://aws-tensorflow-benchmarking/maskrcnn/results/convergence_codebase_iso_baseline_20190323a ~/old_logs/convergence_codebase_iso_baseline_20190323a
#aws s3 cp --recursive s3://aws-tensorflow-benchmarking/maskrcnn/results/convergence_codebase_iso_baseline_20190323b ~/old_logs/convergence_codebase_iso_baseline_20190323b
#aws s3 cp --recursive s3://aws-tensorflow-benchmarking/maskrcnn/results/convergence_codebase_iso_baseline_20190323c ~/old_logs/convergence_codebase_iso_baseline_20190323c
#aws s3 cp --recursive s3://aws-tensorflow-benchmarking/maskrcnn/results/convergence_codebase_iso_baseline_20190323d ~/old_logs/convergence_codebase_iso_baseline_20190323d
#aws s3 cp --recursive s3://aws-tensorflow-benchmarking/maskrcnn/results/convergence_codebase_iso_roialign_box_20190319 ~/old_logs/convergence_codebase_iso_roialign_box_20190319
#aws s3 cp --recursive s3://aws-tensorflow-benchmarking/maskrcnn/results/convergence_codebase_iso_fastrcnn_losses_20190321 ~/old_logs/convergence_codebase_iso_fastrcnn_losses_20190321
#aws s3 cp --recursive s3://aws-tensorflow-benchmarking/maskrcnn/results/convergence_codebase_iso_fastrcnn_outputs_20190321 ~/old_logs/convergence_codebase_iso_fastrcnn_outputs_20190321
#aws s3 cp --recursive s3://aws-tensorflow-benchmarking/maskrcnn/results/convergence_codebase_iso_maskloss_20190321 ~/old_logs/convergence_codebase_iso_maskloss_20190321
#aws s3 cp --recursive s3://aws-tensorflow-benchmarking/maskrcnn/results/convergence_codebase_iso_crop_and_resize_mask_20190321 ~/old_logs/convergence_codebase_iso_crop_and_resize_mask_20190321
#aws s3 cp --recursive s3://aws-tensorflow-benchmarking/maskrcnn/results/convergence_codebase_iso_crop_and_resize_mask_20190323 ~/old_logs/convergence_codebase_iso_crop_and_resize_mask_20190323
#aws s3 cp --recursive s3://aws-tensorflow-benchmarking/maskrcnn/results/convergence_codebase_iso_sampletargets_20190322 ~/old_logs/convergence_codebase_iso_sampletargets_20190322
#aws s3 cp --recursive s3://aws-tensorflow-benchmarking/maskrcnn/results/convergence_codebase_iso_sampletargets_and_roialignbox_20190322 ~/old_logs/convergence_codebase_iso_sampletargets_and_roialignbox_20190322
#aws s3 cp --recursive s3://aws-tensorflow-benchmarking/maskrcnn/results/convergence_codebase_iso_fastrcnnoutputs_and_fastrcnnlosses_20190322 ~/old_logs/convergence_codebase_iso_fastrcnnoutputs_and_fastrcnnlosses_20190322
#aws s3 cp --recursive s3://aws-tensorflow-benchmarking/maskrcnn/results/convergence_codebase_iso_genprops_sampletargets_roialignbox_20190323 ~/old_logs/convergence_codebase_iso_genprops_sampletargets_roialignbox_20190323
#aws s3 cp --recursive s3://aws-tensorflow-benchmarking/maskrcnn/results/convergence_codebase_iso_cropandresizemask_maskloss_20190325 ~/old_logs/convergence_codebase_iso_cropandresizemask_maskloss_20190325
#aws s3 cp --recursive s3://aws-tensorflow-benchmarking/maskrcnn/results/convergence_codebase_iso_frcnnoutputs_frcnnlosses_cropandresizemask_maskloss_20190325 ~/old_logs/convergence_codebase_iso_frcnnoutputs_frcnnlosses_cropandresizemask_maskloss_20190325
#aws s3 cp --recursive s3://aws-tensorflow-benchmarking/maskrcnn/results/convergence_codebase_iso_layer4_block1_convergence_20190325 ~/old_logs/convergence_codebase_iso_layer4_block1_convergence_20190325
#aws s3 cp --recursive s3://aws-tensorflow-benchmarking/maskrcnn/results/convergence_codebase_iso_layer4_block1_throughput_20190325 ~/old_logs/convergence_codebase_iso_layer4_block1_throughput_20190325
#aws s3 cp --recursive s3://aws-tensorflow-benchmarking/maskrcnn/results/convergence_codebase_iso_layer4_block1_fp16_convergence_20190325 ~/old_logs/convergence_codebase_iso_layer4_block1_fp16_convergence_20190325
#aws s3 cp --recursive s3://aws-tensorflow-benchmarking/maskrcnn/results/convergence_codebase_iso_layer4_block1_fp16_throughput_p3dn_20190325 ~/old_logs/convergence_codebase_iso_layer4_block1_fp16_throughput_p3dn_20190325
#aws s3 cp --recursive s3://aws-tensorflow-benchmarking/maskrcnn/results/convergence_codebase_iso_layer4_block1_fp16_throughput_p316xl_20190325 ~/old_logs/convergence_codebase_iso_layer4_block1_fp16_throughput_p316xl_20190325
#aws s3 cp --recursive s3://aws-tensorflow-benchmarking/maskrcnn/results/convergence_codebase_iso_all_but_rpnloss_p3dn_throughput_20190403 ~/old_logs/convergence_codebase_iso_all_but_rpnloss_p3dn_20190403 # For throughput
#aws s3 cp --recursive s3://aws-tensorflow-benchmarking/maskrcnn/results/convergence_codebase_iso_all_but_rpnloss_16xl_throughput_20190403 ~/old_logs/convergence_codebase_iso_all_but_rpnloss_16xl_20190403 # For throughput



################################################################################################################################
# Convergence isolation codebase with errors
################################################################################################################################

# # SampleTargets with randomness removed. Reduced final accuracy
# aws s3 cp --recursive s3://aws-tensorflow-benchmarking/maskrcnn/results/convergence_codebase_iso_sampletargets_20190320 ~/old_logs/convergence_codebase_iso_sampletargets_20190320

# # Layer 2, Block 5. Converges TTA on bbox but not segm. Unknown cause. Unlikely due to mask loss
#aws s3 cp --recursive s3://aws-tensorflow-benchmarking/maskrcnn/results/convergence_codebase_iso_roialignmask_and_cropandresizemask_and_maskloss_20190322 ~/old_logs/convergence_codebase_iso_roialignmask_and_cropandresizemask_and_maskloss_20190322

# # Bratin data 20190321. Converges TTA on bbox but not segm. Many flags enabled, problem isolated down to Layer 2, Block 5
#aws s3 cp --recursive s3://aws-tensorflow-benchmarking/maskrcnn/results/convergence_codebase_iso_bratin_throughput_20190321 ~/old_logs/convergence_codebase_iso_bratin_throughput_20190321
#aws s3 cp --recursive s3://aws-tensorflow-benchmarking/maskrcnn/results/convergence_codebase_iso_bratin_convergence_20190321 ~/old_logs/convergence_codebase_iso_bratin_convergence_20190321

# # RPN Loss. Does not converge TTA on bbox of segme. Unknown cause
#aws s3 cp --recursive s3://aws-tensorflow-benchmarking/maskrcnn/results/convergence_codebase_iso_rpnloss_20190323 ~/old_logs/convergence_codebase_iso_rpnloss_20190323


# # Layer 3, Block 2. Converges TTA on bbox but not segm. Very likely due to ROIAlignMask
#aws s3 cp --recursive s3://aws-tensorflow-benchmarking/maskrcnn/results/convergence_codebase_iso_fastrcnnoutputs_fastrcnnlosses_roialignmask_20190323 ~/old_logs/convergence_codebase_iso_fastrcnnoutputs_fastrcnnlosses_roialignmask_20190323

# # ROIAlignMask
#aws s3 cp --recursive s3://aws-tensorflow-benchmarking/maskrcnn/results/convergence_codebase_iso_roialignmask_20190323a ~/old_logs/convergence_codebase_iso_roialignmask_20190323a
#aws s3 cp --recursive s3://aws-tensorflow-benchmarking/maskrcnn/results/convergence_codebase_iso_roialignmask_20190323b ~/old_logs/convergence_codebase_iso_roialignmask_20190323b

# # Bratin data 20190322. Converges TTA on bbox but not segm. fp16 appeared to have no negative accuracy impact, but need to confirm
#aws s3 cp --recursive logs/ s3://aws-tensorflow-benchmarking/maskrcnn/results/convergence_codebase_iso_bratin_throughput_20190322/
#aws s3 cp --recursive logs/ s3://aws-tensorflow-benchmarking/maskrcnn/results/convergence_codebase_iso_bratin_convergence_20190322/
#aws s3 cp --recursive logs/ s3://aws-tensorflow-benchmarking/maskrcnn/results/convergence_codebase_iso_bratin_fp16_throughput_20190322/
#aws s3 cp --recursive logs/ s3://aws-tensorflow-benchmarking/maskrcnn/results/convergence_codebase_iso_bratin_fp16_convergence_20190322/






# aws s3 cp --recursive logs/ s3://aws-tensorflow-benchmarking/maskrcnn/results/batch-by-default/master_24epoch_20190410




