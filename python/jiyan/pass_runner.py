# ***************************************************************
# Copyright (c) 2020 Jittor. All Rights Reserved.
# Authors:
#   Dun Liang <randonlang@gmail.com>.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************

from jiyan.passes.split_tuple_assign import split_tuple_assign

class PassRunner:
    def run(self, ir):
        split_tuple_assign(ir)
