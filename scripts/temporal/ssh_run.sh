#!/bin/bash
# 用法: ./ssh_run.sh "<远程命令>"
SSH_ASKPASS="$PWD/.ssh-askpass.sh" SSH_ASKPASS_REQUIRE=force \
ssh -p 31311 \
  -o StrictHostKeyChecking=accept-new \
  -o UserKnownHostsFile=/dev/null \
  -o ConnectTimeout=20 \
  -o LogLevel=ERROR \
  root@connect.cqa1.seetacloud.com "$1"
