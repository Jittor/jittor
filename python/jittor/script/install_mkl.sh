#!/bin/bash
set -xe
if [ "$cache_path" = "" ]; then
    bpath=$(dirname "${BASH_SOURCE[0]}")
    cd $bpath
    cd ../extern/mkl
else
    cd $cache_path
fi
filename="mkldnn_lnx_1.0.2_cpu_gomp.tgz"
dirname="mkldnn_lnx_1.0.2_cpu_gomp"
if [ ! -f $filename ]; then
    wget https://github.com/intel/mkl-dnn/releases/download/v1.0.2/$filename
fi
if [ ! -d $dirname ]; then
    tar zxvf $filename
fi

if [ ! -f $dirname/examples/test ]; then
    echo "compile mkldnn example and test"
    cd $dirname/examples
    g++ -std=c++14 cpu_cnn_inference_f32.cpp -Ofast -lmkldnn -I ../include -L ../lib -o test && LD_LIBRARY_PATH=../lib/ ./test
fi