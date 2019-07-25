# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from invoke import task
import yaml
import json




@task
def repeat(c, cmd, repeat=5, verbose=False, sub='|N|'):
    """Repeatedly run a string as command, replacing the substring '|N|' with the iteration number

    $ inv repeat 'echo |N|'
    1
    2
    3
    4
    5

    $ inv repeat 'echo [I]' --repeat=2 --verbose --sub='[I]'
    [cmd = echo 1]
    1
    [cmd = echo 2]
    2

    """
    def log(s):
        if verbose:
            print(s)

    for i in range(repeat):
        cmd_cp = cmd.replace(sub, str(i+1))
        log(f'[cmd = {cmd_cp}]')

        c.run(cmd_cp)


def build_names(csv_str):
    overlays = csv_str.split(",")
    yaml_loader_argstr = csv_str.replace(",", " ")              # space separated. yaml_loader cli arguments
    name = "-".join([o.replace("_", "-") for o in overlays])    # helm. Underscores are disallowed in helm names

    return name, yaml_loader_argstr

@task
def create_experiment_yamls(c,
                            csv,
                            runs=5,
                            overlay_dir="maskrcnn/overlays",
                            base_yaml="maskrcnn/values.yaml",
                            output_dir="maskrcnn/values"):
    """
    $ inv create-experiment-yamls mapdata,32x4,24epoch
    [cmd = ./yaml_overlay maskrcnn/values.yaml mapdata 32x4 24epoch run1 > maskrcnn/values-test/mapdata-32x4-24epoch-run1.yaml]
    [cmd = ./yaml_overlay maskrcnn/values.yaml mapdata 32x4 24epoch run2 > maskrcnn/values-test/mapdata-32x4-24epoch-run2.yaml]
    [cmd = ./yaml_overlay maskrcnn/values.yaml mapdata 32x4 24epoch run3 > maskrcnn/values-test/mapdata-32x4-24epoch-run3.yaml]
    [cmd = ./yaml_overlay maskrcnn/values.yaml mapdata 32x4 24epoch run4 > maskrcnn/values-test/mapdata-32x4-24epoch-run4.yaml]
    [cmd = ./yaml_overlay maskrcnn/values.yaml mapdata 32x4 24epoch run5 > maskrcnn/values-test/mapdata-32x4-24epoch-run5.yaml]

    Will create:

    maskrcnn/
        values/
            32x4-mapdata-24epoch-run1.yaml
            32x4-mapdata-24epoch-run2.yaml
            32x4-mapdata-24epoch-run3.yaml
            32x4-mapdata-24epoch-run4.yaml
            32x4-mapdata-24epoch-run5.yaml`

    """
    csv += f',run|N|' # `inv repeat` will replace run|N| with {run1,run2,run3,run4,run5}
    name, yaml_loader_argstr = build_names(csv)
    yaml_variant_cmd = f'./yaml_overlay --overlay_dir={overlay_dir} {base_yaml} {yaml_loader_argstr}'
    save_variant_cmd = f'{yaml_variant_cmd} > {output_dir}/{name}.yaml'
    repeated_cmd = f'inv repeat "{save_variant_cmd}" --verbose --repeat={runs}'
    c.run(repeated_cmd)


@task
def run_experiment_yamls(c,
                         csv,
                         runs=5,
                         values_dir="maskrcnn/values",
                         name_prefix="maskrcnn",
                         helm_dir="./maskrcnn/"):
    csv += f',run|N|'  # `inv repeat` will replace run|N| with {run1,run2,run3,run4,run5}
    name, _ = build_names(csv)
    helm_cmd = f'helm install --name {name_prefix}-{name} {helm_dir} -f {values_dir}/{name}.yaml'
    repeated_cmd = f'invoke repeat "{helm_cmd}" --verbose --repeat={runs}'
    c.run(repeated_cmd)

@task
def delete_experiment(c,
                      csv,
                      runs=5,
                      name_prefix="maskrcnn"):
    csv += f',run|N|'  # `inv repeat` will replace run|N| with {run1,run2,run3,run4,run5}
    name, _ = build_names(csv)
    helm_cmd = f'helm del --purge {name_prefix}-{name}'
    repeated_cmd = f'invoke repeat "{helm_cmd}" --verbose --repeat={runs}'
    c.run(repeated_cmd)

@task
def kubex(c, cmd, pod="attach-pvc-2"):
    c.run(f'kubectl exec {pod} -- {cmd}')

# inv repeat "kubectl exec attach-pvc-2 -- tail -40 /fsx/logs/correctlr/maskrcnn-32x4-correctlr-24epoch-run|N|/train_log/log.log | ./parse_logs - --output_type=csv"

@task
def collect_results(c, log_prefix, runs=5, verbose=False, extra_maskrcnn_dir=False):
    hide = not verbose
    csv_strings = []
    for i in range(1, runs+1):
        log_path = f'{log_prefix}-run{i}/train_log/{"maskrcnn/" if extra_maskrcnn_dir else ""}log.log'
        if verbose:
            print(log_path)
        try:
            csv_str = c.run(f'invoke kubex "tail -40 {log_path}" | ./parse_logs - --output_type=csv', hide=hide)
            csv_strings.append(csv_str.stdout)
        except Exception as e:
            if verbose:
                print(f'Error in {log_path}. {str(e)}')

    print()
    cols = [[] for _ in csv_strings[0].split(",")]
    for csv_str in csv_strings:
        csv_str = csv_str.replace("\n", "")
        row = csv_str.split(",")
        print(csv_str)
        for i, val in enumerate(row):
            val = float(val)
            if i == 0 and val == 0: break
            cols[i].append(val)
    avgs = [sum(col)/len(col) for col in cols]
    print()
    if verbose:
        print("Avgs")
    print(",".join([str(a) for a in avgs]))

# def parse_pods(pod_lines):
#     output_lines = []
#     for line in pod_lines.split("\n"):
#         if line.strip() == "":
#             continue
#
#         line = line.strip()
#         # print(line.split(" "))
#         cols = [c for c in line.split(" ") if c != ""]
#         _, pod_name, *_ = cols
#         output_lines.append(pod_name)
#     return output_lines
#
# def get_kubectl_nodes_and_pods(c, instance_type_prefix="p3", namespace="default"):
#     """
#
#     :param c:
#     :param instance_type_prefix:
#     :param namespace:
#     :return:
#     """
#     out = c.run('kubectl get nodes -o=json', hide=True)
#     node_dicts = json.loads(out.stdout)['items']
#
#     nodes = []
#
#     for node_dict in node_dicts:
#
#         node = {
#             'instance_type':    node_dict["metadata"]["labels"]["beta.kubernetes.io/instance-type"],
#             'name':             node_dict["metadata"]["name"],
#             'cluster':          node_dict["metadata"]["labels"]["alpha.eksctl.io/nodegroup-name"],
#             'nodegroup':        node_dict["metadata"]["labels"]["alpha.eksctl.io/cluster-name"]
#         }
#
#         if node['instance_type'].startswith(instance_type_prefix):
#             nodes.append(node)
#
#     raw_descriptions = []
#     for i, node in enumerate(nodes):
#         node_name = node['name']
#         raw_running_pod_lines = c.run(f'kubectl describe node {node_name} | grep {namespace}', hide=True, warn=True)
#         node['pods'] = parse_pods(raw_running_pod_lines.stdout)
#         print(node)
#     return nodes
#
#
#
# @task
# def kubep2n(c):
#     # kubep2n = KUBEctl Pod 2 Node
#     get_kubectl_nodes_and_pods(c, instance_type_prefix="p3", namespace="default")
#
