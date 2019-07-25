# Results

Detailed results coming soon!

## Advanced configurations

There are a few advanced configurations that you should be aware of for optimal performance.

### p3dn 

When using p3dn, you will want to use 13 NCCL rings. With p3.16xl, 8 NCCL rings is a good choice.

### Prioritizing bounding box accuracy

You can use a improved bounding box regression weight (`cfg.FRCNN.BBOX_REG_WEIGHTS`) to get better bounding box mAP. If you use `[20, 20, 10, 10]` instead of `[10., 10., 5., 5.]` you will see a solid improvement in bbox mAP (for 12 epochs, 8x4 training, from 37.3 to 398) with a slight decrease in segmentation accuracy (34.3 to 34.2). As you increase the total batch size, the bbox improvement decreases and the segm penalty increases.

### SyncBN

You can use SyncBN to train with very large batch sizes without getting NaN losses. However, currently the accuracy is generally lower than when using FreezeBN and the throughput is significantly worse.

### Large batch size

When training in the 32x4 configuration, you will get NaN ~5% of the time if you do not use gradient clipping. To enable gradient clipping, you need to add `TRAIN.GRADIENT_CLIP=1.5` to the config. This has a minor throughput impact, but eliminates NaN runs.