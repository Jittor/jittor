# ***************************************************************
# Copyright (c) 2020 Jittor. Authors: 
#     Guowei Yang <471184555@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import os
import jittor as jt
import numpy as np
import copy

def find_cache_path():
    from pathlib import Path
    path = str(Path.home())
    dirs = [".cache", "jittor"]
    for d in dirs:
        path = os.path.join(path, d)
        if not os.path.isdir(path):
            os.mkdir(path)
        assert os.path.isdir(path)
    return path

cache_path = find_cache_path()

class TestMpi(unittest.TestCase):
    def test(self):
        # Modified from: https://mpitutorial.com/tutorials/mpi-hello-world/zh_cn/
        content="""
        #include <mpi.h>
        #include <stdio.h>

        int main(int argc, char** argv) {
            MPI_Init(NULL, NULL);

            int world_size;
            MPI_Comm_size(MPI_COMM_WORLD, &world_size);

            int world_rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

            char processor_name[MPI_MAX_PROCESSOR_NAME];
            int name_len;
            MPI_Get_processor_name(processor_name, &name_len);

            printf("Hello world from processor %s, rank %d out of %d processors\\n",processor_name, world_rank, world_size);

            MPI_Finalize();
        }
        """
        test_path=os.path.join(cache_path,"test_mpi.cc")
        f=open(test_path,"w")
        f.write(content)
        f.close()
        mpi_path=jt.flags.mpi_path
        mpi_include = os.path.join(mpi_path, "include")
        mpi_lib = os.path.join(mpi_path, "lib")
        cmd = f"cd {cache_path} && g++ {test_path} -I {mpi_include} -L {mpi_lib} -lmpi -o test_mpi && mpirun -n 4 ./test_mpi"
        self.assertEqual(os.system(cmd), 0)

if __name__ == "__main__":
    unittest.main()