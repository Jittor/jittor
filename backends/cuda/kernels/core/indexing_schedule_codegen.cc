#include "indexing_codegen.h"

namespace jittor {
void cuda_loop_schedule(NanoVector o_shape, int* masks, int* tdims) {
    // bz by bx tz ty tx
    // 5  4  3  2  1  0
    // LOi: bitmask of used dims of loop i
    // LOi bit 6: need for
    //    if need for, keep for range: for (int i@i=tid; tid<range; tid+=tnum)
    //    if not need for, replace range -> tnum, for -> int i@i = tid
    int rtnum = 1024;
    // int max_tnum = {1024, 1024, 64, (1u<<31)-1, 65535, 65535};
    int loop_id = (int)o_shape.size()-1;
    int tid = 0;
    int64 block_size = 1;
    int thread_size = 1;
    for (int i=0; i<6; i++) tdims[i] = 1;
    for (; tid<3 && loop_id>=0 && rtnum>1; tid++) {
        int64 si = o_shape[loop_id];
        int mask = 1<<tid;
        if (tid==2) rtnum = std::min(64, rtnum);
        if (si>rtnum*4) {
            // need for, use tid(1<<i) and bx(8)
            mask |= 8|(1<<6);
            block_size = (si-1)/rtnum+1;
            tdims[tid] = rtnum;
            tdims[3] = block_size;
            tid = 3;
            thread_size *= rtnum;
            rtnum = 0;
        } else
        if (si>rtnum) {
            mask |= (1<<6);
            thread_size *= rtnum;
            tdims[tid] = rtnum;
            rtnum = 0;
        } else {
            rtnum = rtnum / std::max(si, (int64)1);
            thread_size *= si;
            tdims[tid] = si;
            if (si == 0) mask |= 1<<7;
        }
        masks[loop_id] = mask;
        loop_id --;
    }
    int64 total_size = (int64)block_size*thread_size;
    if (tid<3) tid=3;
    for (; tid<6 && loop_id>=0 && total_size<(256*1024); tid++) {
        int64 si = o_shape[loop_id];
        int mask = 1<<tid;
        if (si == 0) mask |= 1<<7;
        int64 max_thread = tid>=4 ? 65535 : (1u<<31)-1;
        if (si > max_thread) {
            si = max_thread;
            mask |= 1<<6;
        }
        total_size *= si;
        tdims[tid] = si;
        masks[loop_id] = mask;
        loop_id --;
    }
    while (loop_id>=0) {
        masks[loop_id--] = 0;
    }
}
} // namespace jittor
