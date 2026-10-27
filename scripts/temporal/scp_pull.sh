#!/bin/bash
# 用法: ./scp_pull.sh <远程文件> <本地目录>
SSH_ASKPASS="$PWD/.ssh-askpass.sh" SSH_ASKPASS_REQUIRE=force \
scp -P 31311 \
  -o StrictHostKeyChecking=accept-new \
  -o UserKnownHostsFile=/dev/null \
  -o ConnectTimeout=20 \
  -o LogLevel=ERROR \
  "root@connect.cqa1.seetacloud.com:$1" "$2"
