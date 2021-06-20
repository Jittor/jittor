# wget https://github.com/oneapi-src/oneDNN/archive/refs/tags/v2.2.zip
# extract zip
# cd to root folder

mkdir -p build
cd build
make clean
export CC=aarch64-linux-gnu-gcc-8
export CXX=aarch64-linux-gnu-g++-8
cmake .. \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=AARCH64 \
    -DCMAKE_LIBRARY_PATH=/usr/aarch64-linux-gnu/lib \
    -DCMAKE_BUILD_TYPE=Release
    # -DCMAKE_SHARED_LINKER_FLAGS=' -lm ' \
make -j8

name=dnnl_lnx_2.2.0_cpu_gomp_aarch64
mkdir -p $name
cp -r ../include ./$name/
mkdir -p ./$name/lib
cp ./src/libmkldnn.so ./$name/lib/libmkldnn.so
cp -r ../examples ./$name/
cp ./include/oneapi/dnnl/* ./$name/include/oneapi/dnnl/

tar -acvf $name.tgz ./$name/

rsync -avPu $name.tgz jittor-web:Documents/jittor-blog/assets/
ssh jittor-web Documents/jittor-blog.git/hooks/post-update
echo "https://cg.cs.tsinghua.edu.cn/jittor/assets/$name.tgz"
md5sum $name.tgz