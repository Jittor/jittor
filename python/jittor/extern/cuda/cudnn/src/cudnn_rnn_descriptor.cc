// ***************************************************************
// Copyright (c) 2022 Jittor. All Rights Reserved. 
// Maintainers: 
//      Zheng-Ning Liu <lzhengning@gmail.com>
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "cudnn_rnn_descriptor.h"

namespace jittor {

vector<int32_t> cudnn_rnn_weight_offset(string mode, int input_size, int hidden_size, int num_layers, int proj_size, bool bias, bool bidirectional) {
    // A pseudo mini-batch for fetching weight space size.
    int dimX[] = {1, input_size, 1};
    int strideX[] = {input_size, 1, 1};
    cudnnTensorDescriptor_t xDesc;
    checkCudaErrors(cudnnCreateTensorDescriptor(&xDesc));
    checkCudaErrors(cudnnSetTensorNdDescriptor(xDesc, CUDNN_DATA_FLOAT, 3, dimX, strideX));

    RnnDescriptor rnn_desc = RnnDescriptor(cudnn_handle, mode, hidden_size, num_layers, 0, bidirectional);
    int weightSpaceSize = rnn_desc.weight_space_size(xDesc);
    RnnWeightDescriptor w_desc(weightSpaceSize);
    
    vector<int> weight_offsets;
    weight_offsets.push_back(weightSpaceSize / sizeof(float));    

    int num_directions = bidirectional + 1;
    int num_linear_layers = rnn_string_to_num_linear_layers(mode);
    
    for (int layer = 0; layer < num_layers * num_directions; layer++) {
        for (int linLayerID = 0; linLayerID < num_linear_layers; linLayerID++) {
            cudnnFilterDescriptor_t linLayerMatDesc;
            cudnnFilterDescriptor_t linLayerBiasDesc;
            float *linLayerMat = nullptr;
            float *linLayerBias = nullptr;

            checkCudaErrors(cudnnCreateFilterDescriptor(&linLayerMatDesc));
            checkCudaErrors(cudnnCreateFilterDescriptor(&linLayerBiasDesc));

            checkCudaErrors(cudnnGetRNNLinLayerMatrixParams(
                cudnn_handle, rnn_desc.desc,
                layer, 
                xDesc, 
                w_desc.desc, 
                nullptr,
                linLayerID,
                linLayerMatDesc, 
                (void **) &linLayerMat
            ));
            weight_offsets.push_back(linLayerMat - (float *) nullptr);

            if (bias) {
                checkCudaErrors(cudnnGetRNNLinLayerBiasParams(
                    cudnn_handle, rnn_desc.desc,
                    layer, 
                    xDesc, 
                    w_desc.desc, 
                    nullptr,
                    linLayerID,
                    linLayerBiasDesc, 
                    (void **) &linLayerBias
                ));
                weight_offsets.push_back(linLayerBias - (float *) nullptr);
            }
        }
    }

    checkCudaErrors(cudnnDestroyTensorDescriptor(xDesc));

    return weight_offsets;
}


} // jittor
