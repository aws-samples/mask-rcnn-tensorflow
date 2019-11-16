## Launching notebooks

These notebooks are designed to be used within the mask r-cnn docker container on a P3 EC2 instance. After starting an instance, get the public IP address, and ssh from your terminal using

```ssh -i ~/your/pem/file -L localhost:8890:localhost:8888 ubuntu@[ip-address]```

This will ssh into your instance, while forwarding whatever is running on port 8888 in your instance to port 8890 on your local machine.

From within the instance, start the container by running

```docker run -t -d --rm --gpus 8 --net=host --name mrcnn -v /home/ubuntu/data/:/data -v /home/ubuntu/logs/:/logs awssamples/mask-rcnn-tensorflow:latest /bin/bash -c "nohup jupyter notebook --no-browser --ip=0.0.0.0 > notebook.log"```

followed by

```docker exec mrcnn /bin/bash -c "jupyter notebook list"```

You should get an output that looks something like

```
Currently running servers:
http://0.0.0.0:8888/?token=d4414589af37ed69b1efce555c8eddc3139e5b3baaa4ff05 :: /
```

Copy the token portion ```d4414589af37ed69b1efce555c8eddc3139e5b3baaa4ff05``` and go to ```localhost:8890``` in your browser.

From the Jupyter homepage, you can browse files within the container, upload and download files, create a new notebook, or start a terminal from within the container.

Included are three notebooks for various model functionality.

#### coco_inspection
An overview of the coco datasets and tools for understanding how the data is constructed, and tools for visualizing and manipulating COCO data. 

- Pycocotools: a set of tools developed by the COCO team for loading and visualizing COCO data
- Cocosubsetter: A small tool developed for these notebooks that can subset COCO data for model debugging

#### tensor_inspection
Often when training a model it is useful to see what individual tensors in the graph are doing. This can be tricky with Tensorpack, so we include a notebook that covers how to output, store, and analyze specific tensors at runtime.

#### visualization
The notebook allows the user to see what their model is finding in the images, and compare their performance to the ground truth in the COCO data. Also allows users to load their own images.
