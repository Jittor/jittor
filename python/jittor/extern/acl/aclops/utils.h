#pragma once
#include <unordered_map>
namespace jittor
{
    aclDataType get_dtype(NanoString s) {
        switch (s.data) {
            case 22667: return ACL_FLOAT;
            case 14474: return ACL_FLOAT16;
            case 27781: return ACL_INT64;
            case 19588: return ACL_INT32;
            case 3202: return ACL_INT8;
            case 11395: return ACL_INT16;
            case 3206: return ACL_UINT8;
            case 11399: return ACL_UINT16;
            case 19592: return ACL_UINT32;
            case 3713: return ACL_BOOL;
            default:
                LOGf << "Not supported dtype: " << s;
                return ACL_FLOAT; // 默认返回 ACL_FLOAT
        }
    }
}