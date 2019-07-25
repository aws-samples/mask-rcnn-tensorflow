# Training on EKS

## Steps

* (1) Set up EKS cluster using eksctl with p3.16xl or p3dn.24xl nodes.
* (2) Set up FSx filesystem and expose it in k8s
* (3) Install Helm and Tiller
* (4) Install MPIJob CRD
* (5) Run training job


### (1) Set up EKS cluster

- Install the eksctl from https://github.com/weaveworks/eksctl

- There are some requirements when setting up the EKS cluster:
    - Make sure the nodes have fsx access
    - Make sure the nodes live in a single AZ
    - Make sure the nodes have the NVIDIA GPU daemonset

- The commands in `eksctl/p3_create.sh` handles those requirements.
    - You should update `eksctl/p3_config.yaml` to match your needs (name/region/vpc-id/subnets/instance-type/az/capacity/ssh-public-key)
        - sshPublicKeyPath is the name of an EC2 KeyPair.
        - some examples can be found at https://github.com/weaveworks/eksctl/tree/master/examples
    - Run the commands individually, not via script
        - you need to update the `KUBECONFIG` to match your path



### (2) Set up FSx for Lustre

- Create FSx filesystem if this is the first time
    - Alter FSx security group to allow port 988 traffic from anywhere - https://docs.aws.amazon.com/fsx/latest/LustreGuide/limit-access-security-groups.html#fsx-vpc-security-groups
        - When add the inbound rule, the `Type` should be `Custom TCP Rule`
    - Add S3 permissions to worker role so stage-data.yaml can download the files
        - Open the AWS IAM console, find the eks nodegroup role created by eksctl
        - add the s3 policy (e.g. s3 AWS Tensorflow benchmarking policy)
- Add FSx support to the cluster
    - Install FSx CSI driver with `kubectl apply -f https://raw.githubusercontent.com/kubernetes-sigs/aws-fsx-csi-driver/master/deploy/kubernetes/manifest.yaml`
- Add FSx as a persistant volume and claim
    - Customize `fsx/pv-fsx.yaml` for your FSx file-system id and AWS region
    - Execute: `kubectl apply -f fsx/p3/pv-fsx.yaml`
    - Check to see the persistent-volume was successfully created by executing: `kubectl get pv`
    - Execute: `kubectl apply -f fsx/p3/pvc-fsx.yaml` to create an EKS persistent-volume-claim
- Stage data on fsx
    - Customize `fsx/stage-data.yaml` with image name and location of data on s3
    - Run `kubectl apply -f fsx/p3/stage-data.yaml`
    - Confirm that it worked with  `kubectl apply -f fsx/attach-pvc-2.yaml` and `kubectl exec attach-pvc-2 -it -- /bin/bash`
    - To clean up: `kubectl delete pod stage-data`. It can be helpful to leave the `attach-pvc-2` pod running to view the fsx contents (e.g. experiment results) later.

### (3) Install Helm and Tiller

- Install helm locally
    - `brew install kubernetes-helm`
- Set up tiller in the cluster
    - `kubectl create -f helm/tiller-rbac-config.yaml`
    - `helm init --service-account tiller --history-max 200`


### (4) Install MPIJob Operator

- `helm install --name mpijob helm/mpijob/`


### (5) Launch training

- Update `maskrcnn/values.yaml` with your info
    - To launch the training, use `helm install --name maskrcnn ./maskrcnn/`
    - To delete, use `helm del --purge maskrcnn`

### Deleting the cluster

- See `eksctl/delete.sh` for commands to delete cluster
- If you attached a policy to the EKS worker IAM role (e.g. to download data from S3, you will need to manually remove that policy from the role in order for CloudFormation to be able to delete all resources for your cluster)


### Multiple Training Jobs

Scale the nodegroup to the desired number of nodes. We do not have an autoscaling solution yet (may investigate Escalator).

- either by scaling up the existing nodegroup
    - `eksctl scale nodegroup --cluster CLUSTER_NAME --name ng-1 --nodes 4`
- or by creating a new nodegroup based on `eksctl/additional_nodegroup.yaml`
    - `eksctl create nodegroup -f eks/eksctl/p3_additional_nodegroup.yaml`

`maskrcnn/values.yaml` holds the default training params for 1 node, 8 GPU training. To launch a training job with a different configuration, we suggest you create a new yaml file with the desired params.

To make that easier, we use a the `overyaml.py` utlity, which takes in a base yaml, applies a list of changes (overlays) to it and prints the new yaml to stdout. See `overyaml.md` for details.

To run 4 node training (32x4) with the 24 epoch schedule, we do the following

```
export OVERLAY_DIR=maskrcnn/overlays
./overyaml.py maskrcnn/values.yaml 32x4 24epoch > maskrcnn/values/determinism-32x4-24epoch.yaml
```

Then we use helm to launch training, telling it to use the newly created values yaml instead of the deafult `maskrcnn/values.yaml`

```
helm install --name maskrcnn-determinism-32x4-24epoch ./maskrcnn/ -f maskrcnn/values/determinism-32x4-24epoch.yaml
```

#### Multiple identical jobs

If you need to run multiple identical jobs without naming conflict, we have the runX overlays to help.

```
export OVERLAY_DIR=maskrcnn/overlays
./yaml_overlay maskrcnn/values.yaml 32x4 24epoch run1 > maskrcnn/values/determinism-32x4-24epoch-run1.yaml
./yaml_overlay maskrcnn/values.yaml 32x4 24epoch run2 > maskrcnn/values/determinism-32x4-24epoch-run2.yaml

helm install --name maskrcnn-determinism-32x4-24epoch-run1 ./maskrcnn/ -f maskrcnn/values/determinism-32x4-24epoch-run1.yaml
helm install --name maskrcnn-determinism-32x4-24epoch-run2 ./maskrcnn/ -f maskrcnn/values/determinism-32x4-24epoch-run2.yaml
```



### Tensorboard

`kubectl apply -f eks/tensorboard/tensorboard.yaml`

`kubectl port-forward tensorboard 6006:6006`

Shortcut is `./tboard.sh`

### Examine fsx

`kubectl apply -f fsx/apply-pvc-2`

`./ssh.sh`

We use `apply-pvc-2` because it uses the tensorboard-mask-rcnn image, which has useful tools like the AWS CLI
