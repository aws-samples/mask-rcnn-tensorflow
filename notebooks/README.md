## Launching notebooks

These notebooks are designed to be used within the mask r-cnn docker container on a P3 EC2 instance. After starting an instance, get the public IP address, and ssh from your terminal using

```ssh -i ~/your/pem/file -L localhost:8890:localhost:8888 -L localhost:6008:localhost:6006 ubuntu@[ip-address]```

This will ssh into your instance, while forwarding whatever is running on ports 8888 and 6006 in your instance to ports 8890 and 6008 on your local machine.

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