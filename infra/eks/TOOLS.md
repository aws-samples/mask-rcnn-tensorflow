# Tools

# Invoke Tasks

We use [invoke](https://www.pyinvoke.org/) to automate tasks around EKS training.

invoke tasks are essentially python functions that can be launched with `inv task_name args` or `invoke task_name args`

## repeat

Repeatedly run a string as command, replacing the substring '|N|' (can be changed) with the iteration number

```
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
```

## 
