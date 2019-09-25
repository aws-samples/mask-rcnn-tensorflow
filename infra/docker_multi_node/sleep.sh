#!/usr/bin/env bash
PORT_ID=${1:-1234}
/usr/sbin/sshd -p $PORT_ID; sleep infinity
