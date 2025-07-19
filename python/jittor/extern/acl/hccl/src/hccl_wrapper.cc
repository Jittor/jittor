// ***************************************************************
// Copyright (c) 2025 Jittor.
// All Rights Reserved. 
// Maintainers:
//     Jiapeng Zhang <zjp24@mails.tsinghua.edu.cn>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************

#include "hccl_wrapper.h"
#include "event_queue.h"
#include "acl_jittor.h"
#include <acl/acl.h>

namespace jittor {

HcclRootInfo root_info;
HcclComm comm;
uint32_t hccl_device_id = 0;

struct hccl_initer {
    uint32_t device_count = 0;
    hccl_initer() { 
        ACLCHECK(aclrtGetDeviceCount(&device_count));
        if (!device_count) return;
        if (!inside_mpi) return;
        hccl_device_id = mpi_local_rank;
        if (mpi_local_rank >= device_count) {
            LOGw << "mpi_local_rank(">>mpi_local_rank>>") is larger than device_count("
                >>device_count>>")";
            hccl_device_id = hccl_device_id % device_count;
        }
        LOGv << "HCCL init in device" << hccl_device_id << "local_rank" << mpi_local_rank;
        //LOGir << aclstream;
        //event_queue.run_sync([]() {
            ACLCHECK(aclrtSetDevice(hccl_device_id));
        //});
        use_device_mpi = true;
        LOGir << "HCCL init in device" << hccl_device_id << "local_rank" << mpi_local_rank;
        if (mpi_world_rank == 0)
            HCCLCHECK(HcclGetRootInfo(&root_info));
        MPI_CHECK(MPI_Bcast(&root_info, HCCL_ROOT_INFO_BYTES, MPI_CHAR, 0, MPI_COMM_WORLD));
        //MPI_Barrier(MPI_COMM_WORLD);
        LOGir << "Count:" << device_count << "HCCL init in device" << hccl_device_id;
        HCCLCHECK(HcclCommInitRootInfo(device_count, &root_info, hccl_device_id, &comm));
        ACLCHECK(aclrtCreateStream(&aclstream));
        LOGi << "HCCL init success in device" << hccl_device_id;
    }

    ~hccl_initer() {
        if (!device_count) return;
        if (!inside_mpi) return;
        if (!use_device_mpi) return;
        HCCLCHECK(HcclCommDestroy(comm));
    }
};

static hccl_initer hccl_initer;
}