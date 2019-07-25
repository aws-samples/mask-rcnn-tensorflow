# Train with Sagemaker

## To launch training

- (1) Set up your Sagemaker role according to https://medium.com/ml-bytes/how-to-a-create-a-sagemaker-execution-role-539866910bda and record it as `$YOUR_SM_ROLE`
    - Make sure you have full access for S3
- (2) Modify the `launch_sm_job.py`, pick your sagemaker_iam_role, instance type, instance numbers, GPUs per instance and other Sagemaker specifications.
- (3) Modify the `run.sh`, pick your batch_size, training epoches and other training parameters.
- (4) Create a repo in ECR with `$YOUR_JOB_NAME`
- (4) Launch your training job by run `./build_push_submit $YOUR_JOB_NAME $YOUR_SM_ROLE`
    - If your have your image ready in ECS and just want to launch the job, you can run `python3 Launch_sm_job.py $YOUR_JOB_NAME $YOUR_SM_ROLE`

## What happened inside?

### (1) Build image and push it to ECR
- The `Dockerfile_base` is similar to dockerfile used for EKS and EC2, use that to build the base image
- The `Dockerfile_sm` is specially for SageMaker, everytime if the `run_mpi.py` or `run.sh` is changed, the image needs to be rebuilt
- The `build_and_push.sh` will build the image and push it to ECR
### (2) Launch SageMaker estimator job
- `Launch_sm_job.py` will lauch the estimator, which essentially launch the instances in container with the docker image we built before. Once the instance is started, it will launch the `run_mpi.py`
- `run_mpi.py` build all mpi commands to run multi-node multi-gpu training. It will run the `run.sh`, which launch the training job.
