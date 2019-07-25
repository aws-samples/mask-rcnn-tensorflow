# Overyaml

Take a base yaml file, apply a series of changes (overlays) and print out new yaml.

e.g. take base maskrcnn params and change to run 5 experiments of 24 epochs, predefined_padding=True, 32x4 GPU configuration without helm naming conflicts. Then run 5 more experiments with 32x2 GPU configuration.

* Be able to make changes to the base yaml and have it impact all other configurations.
* Add a new experiment without having an exploding number of yaml files to maintain and update.

## CLI Syntax

`./yaml_overlay $BASE $OVERLAY1 $OVERLAY2 $OVERLAY3 ...`

Takes a base yaml and applies overlays sequentially. At the end, prints new yaml out to stdout. Overlay names should be the path to the overlay file minus '.yaml'.

`./yaml_overlay maskrcnn/values.yaml maskrcnn/overlays/24epoch maskrcnn/overlays/32x4`

## Overlay folder

You can keep all your overlays in a single folder and then pass in an `overlay_dir` either through the `--overlay_dir` flag or through the `OVERLAY_DIR` environment variable.

```
export OVERLAY_DIR=maskrcnn/overlays
./yaml_overlay maskrcnn/values.yaml 24epoch 32x4
```

## Overlay syntax

An overlay is a yaml file containing two sets of changes - changes where you want to `set` a new value for a field and changes where you want to `append` a postfix to the existing value.

```
set:
    someScope:
        someField: "new_value"
append:
    someScope:
        someOtherField: "_new_postfix"
```

Both `set` and `append` are optional.

Changes are represented as a copy of the original object with unchanged fields ommitted and each changed field holding the new value or the postfix as the field's value. See example below.


## Example

**base.yaml**

```
someScope:
    someField: 1
    someOtherField: "my_name"
```

**overlay.yaml**

```
set:
    someScope:
        someField: "new_value"
append:
    someScope:
        someOtherField: "_new_postfix"
```



###`$ ./yaml_overlay base.yaml overlay > output.yaml`


**output.yaml**
```
someScope:
    someField: "new_value"
    someOtherField: "my_name_new_postfix"
```
