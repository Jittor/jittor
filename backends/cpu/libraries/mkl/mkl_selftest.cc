#include "onednn_runtime.h"

int mkl_test_entry() {
    float a[4] = {1,2,3,4}, b[4] = {1,0,0,1}, c[4] = {};
    jittor::onednn_matmul_execute(1,2,2,2,false,false,a,b,c);
    for (int i=0; i<4; ++i) if (a[i] != c[i]) return 1;
    return 0;
}
