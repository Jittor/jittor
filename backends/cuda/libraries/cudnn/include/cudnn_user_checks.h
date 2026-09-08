#pragma once
#include "core/var.h"
#include "cudnn_wrapper.h"

namespace jittor {

inline void cudnn_check_conv_axis(int stride, int padding, int dilation, const char* axis) {
    USER_CHECK(stride > 0) << "cuDNN convolution stride must be positive on" << axis;
    USER_CHECK(padding >= 0) << "cuDNN convolution padding must be nonnegative on" << axis;
    USER_CHECK(dilation > 0) << "cuDNN convolution dilation must be positive on" << axis;
}

inline void cudnn_check_conv_layout(const string& layout, const char* axes, const char* operand) {
    const string expected(axes);
    if (layout.empty()) return;
    USER_CHECK(layout.size() == expected.size()) << "Not a valid format for" << operand << layout;
    for (char axis : layout)
        USER_CHECK(expected.find(axis) != string::npos && layout.find(axis) == layout.rfind(axis))
            << "Not a valid format for" << operand << layout;
}

inline void cudnn_check_conv_inputs(Var* first, Var* second, int groups) {
    USER_CHECK(groups > 0) << "cuDNN convolution groups must be positive";
}

} // namespace jittor
