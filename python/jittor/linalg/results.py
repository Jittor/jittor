# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers:
#     Haoyang Peng <2247838039@qq.com>
#     Guowei Yang <471184555@qq.com>
#     Dun Liang <randonlang@gmail.com>.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Named tuple results shared by public linear algebra functions."""
from collections import namedtuple

SVD = namedtuple("svd", ["U", "S", "Vh"])
INVEX = namedtuple("inv_ex", ["inverse", "info"])

# The canonical result module can expose the original tuple type names without
# colliding with the public functions of the same names.
svd = SVD
inv_ex = INVEX
