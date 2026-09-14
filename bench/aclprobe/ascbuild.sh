#!/usr/bin/env bash
# Build an AscendC mixed host+device probe with ccec for Ascend950PR (dav-c310-vec).
set -euo pipefail
src=${1:?src}; out=${2:-${src%.cc}}
A=${ASCEND_HOME_PATH:?source env-npu.sh first}
D=$A/x86_64-linux
GXX="-I/usr/include/c++/11 -I/usr/include/x86_64-linux-gnu/c++/11 -I/usr/include/c++/11/backward"
INC="-I$D/asc/impl/adv_api -I$D/asc/impl/basic_api -I$D/asc/impl/c_api -I$D/asc/impl/basic_api/reg_compute \
 -I$D/asc/impl/simt_api -I$D/asc/impl/utils -I$D/asc -I$D/asc/include -I$D/asc/include/adv_api \
 -I$D/asc/include/basic_api -I$D/asc/include/aicpu_api -I$D/asc/include/c_api \
 -I$D/asc/include/basic_api/reg_compute -I$D/asc/include/simt_api -I$D/asc/include/utils \
 -I$D/tikcpp/tikcfw -I$D/tikcpp/tikcfw/interface -I$D/tikcpp/tikcfw/impl \
 -I$A/include -I$A/include/aclnnop"
exec $A/bin/ccec -O2 -std=c++17 --cce-aicore-lang -D__DAV_C310__ \
  --cce-aicore-arch=dav-c310-vec --cce-auto-sync --cce-mask-opt -w \
  -DTILING_KEY_VAR=0 $GXX $INC "$src" -o "$out" \
  -L$A/lib64 -lascendcl -lruntime -lnnopbase -lopapi \
  -L/usr/lib/gcc/x86_64-linux-gnu/11 -lstdc++ -lm
