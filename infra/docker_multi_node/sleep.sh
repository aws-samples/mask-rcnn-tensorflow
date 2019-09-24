# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env bash
PORT_ID=${1:-1234}
/usr/sbin/sshd -p $PORT_ID; sleep infinity
