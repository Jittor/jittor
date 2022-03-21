// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************

#include "common.h"
using std::string;
using std::unordered_map;

typedef int aclError;

static inline unordered_map<aclError,string> gen_map(string s) {
    unordered_map<aclError,string> smap;
    for (int i=0; i<s.size(); i++) {
        if (s[i] == ';') {
            int j=s.rfind(" ", i);
            int code = std::stoi(s.substr(j+1, i-j-1));
            int k = s.rfind(" ", j-1);
            int l = s.rfind(" ACL_", k-1);
            smap[code] = s.substr(l+1, k-l-1);
        }
    }
    return smap;
}

string acl_error_to_string(aclError error) {

static unordered_map<aclError,string> acl_error_map = gen_map(R"(
// from acl_base.h
static const int ACL_ERROR_INVALID_PARAM = 100000;
static const int ACL_ERROR_UNINITIALIZE = 100001;
static const int ACL_ERROR_REPEAT_INITIALIZE = 100002;
static const int ACL_ERROR_INVALID_FILE = 100003;
static const int ACL_ERROR_WRITE_FILE = 100004;
static const int ACL_ERROR_INVALID_FILE_SIZE = 100005;
static const int ACL_ERROR_PARSE_FILE = 100006;
static const int ACL_ERROR_FILE_MISSING_ATTR = 100007;
static const int ACL_ERROR_FILE_ATTR_INVALID = 100008;
static const int ACL_ERROR_INVALID_DUMP_CONFIG = 100009;
static const int ACL_ERROR_INVALID_PROFILING_CONFIG = 100010;
static const int ACL_ERROR_INVALID_MODEL_ID = 100011;
static const int ACL_ERROR_DESERIALIZE_MODEL = 100012;
static const int ACL_ERROR_PARSE_MODEL = 100013;
static const int ACL_ERROR_READ_MODEL_FAILURE = 100014;
static const int ACL_ERROR_MODEL_SIZE_INVALID = 100015;
static const int ACL_ERROR_MODEL_MISSING_ATTR = 100016;
static const int ACL_ERROR_MODEL_INPUT_NOT_MATCH = 100017;
static const int ACL_ERROR_MODEL_OUTPUT_NOT_MATCH = 100018;
static const int ACL_ERROR_MODEL_NOT_DYNAMIC = 100019;
static const int ACL_ERROR_OP_TYPE_NOT_MATCH = 100020;
static const int ACL_ERROR_OP_INPUT_NOT_MATCH = 100021;
static const int ACL_ERROR_OP_OUTPUT_NOT_MATCH = 100022;
static const int ACL_ERROR_OP_ATTR_NOT_MATCH = 100023;
static const int ACL_ERROR_OP_NOT_FOUND = 100024;
static const int ACL_ERROR_OP_LOAD_FAILED = 100025;
static const int ACL_ERROR_UNSUPPORTED_DATA_TYPE = 100026;
static const int ACL_ERROR_FORMAT_NOT_MATCH = 100027;
static const int ACL_ERROR_BIN_SELECTOR_NOT_REGISTERED = 100028;
static const int ACL_ERROR_KERNEL_NOT_FOUND = 100029;
static const int ACL_ERROR_BIN_SELECTOR_ALREADY_REGISTERED = 100030;
static const int ACL_ERROR_KERNEL_ALREADY_REGISTERED = 100031;
static const int ACL_ERROR_INVALID_QUEUE_ID = 100032;
static const int ACL_ERROR_REPEAT_SUBSCRIBE = 100033;
static const int ACL_ERROR_STREAM_NOT_SUBSCRIBE = 100034;
static const int ACL_ERROR_THREAD_NOT_SUBSCRIBE = 100035;
static const int ACL_ERROR_WAIT_CALLBACK_TIMEOUT = 100036;
static const int ACL_ERROR_REPEAT_FINALIZE = 100037;
static const int ACL_ERROR_NOT_STATIC_AIPP = 100038;
static const int ACL_ERROR_COMPILING_STUB_MODE = 100039;
static const int ACL_ERROR_GROUP_NOT_SET = 100040;
static const int ACL_ERROR_GROUP_NOT_CREATE = 100041;
static const int ACL_ERROR_PROF_ALREADY_RUN = 100042;
static const int ACL_ERROR_PROF_NOT_RUN = 100043;
static const int ACL_ERROR_DUMP_ALREADY_RUN = 100044;
static const int ACL_ERROR_DUMP_NOT_RUN = 100045;
static const int ACL_ERROR_PROF_REPEAT_SUBSCRIBE = 148046;
static const int ACL_ERROR_PROF_API_CONFLICT = 148047;
static const int ACL_ERROR_INVALID_MAX_OPQUEUE_NUM_CONFIG = 148048;
static const int ACL_ERROR_INVALID_OPP_PATH = 148049;
static const int ACL_ERROR_OP_UNSUPPORTED_DYNAMIC = 148050;
static const int ACL_ERROR_RELATIVE_RESOURCE_NOT_CLEARED = 148051;

static const int ACL_ERROR_BAD_ALLOC = 200000;
static const int ACL_ERROR_API_NOT_SUPPORT = 200001;
static const int ACL_ERROR_INVALID_DEVICE = 200002;
static const int ACL_ERROR_MEMORY_ADDRESS_UNALIGNED = 200003;
static const int ACL_ERROR_RESOURCE_NOT_MATCH = 200004;
static const int ACL_ERROR_INVALID_RESOURCE_HANDLE = 200005;
static const int ACL_ERROR_FEATURE_UNSUPPORTED = 200006;
static const int ACL_ERROR_PROF_MODULES_UNSUPPORTED = 200007;

static const int ACL_ERROR_STORAGE_OVER_LIMIT = 300000;

static const int ACL_ERROR_INTERNAL_ERROR = 500000;
static const int ACL_ERROR_FAILURE = 500001;
static const int ACL_ERROR_GE_FAILURE = 500002;
static const int ACL_ERROR_RT_FAILURE = 500003;
static const int ACL_ERROR_DRV_FAILURE = 500004;
static const int ACL_ERROR_PROFILING_FAILURE = 500005;

// from ge_error_codes.h
static const uint32_t ACL_ERROR_GE_PARAM_INVALID = 145000U;
static const uint32_t ACL_ERROR_GE_EXEC_NOT_INIT = 145001U;
static const uint32_t ACL_ERROR_GE_EXEC_MODEL_PATH_INVALID = 145002U;
static const uint32_t ACL_ERROR_GE_EXEC_MODEL_ID_INVALID = 145003U;
static const uint32_t ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID = 145006U;
static const uint32_t ACL_ERROR_GE_EXEC_MODEL_ADDR_INVALID = 145007U;
static const uint32_t ACL_ERROR_GE_EXEC_MODEL_QUEUE_ID_INVALID = 145008U;
static const uint32_t ACL_ERROR_GE_EXEC_LOAD_MODEL_REPEATED = 145009U;
static const uint32_t ACL_ERROR_GE_DYNAMIC_INPUT_ADDR_INVALID = 145011U;
static const uint32_t ACL_ERROR_GE_DYNAMIC_INPUT_LENGTH_INVALID = 145012U;
static const uint32_t ACL_ERROR_GE_DYNAMIC_BATCH_SIZE_INVALID = 145013U;
static const uint32_t ACL_ERROR_GE_AIPP_BATCH_EMPTY = 145014U;
static const uint32_t ACL_ERROR_GE_AIPP_NOT_EXIST = 145015U;
static const uint32_t ACL_ERROR_GE_AIPP_MODE_INVALID = 145016U;
static const uint32_t ACL_ERROR_GE_OP_TASK_TYPE_INVALID = 145017U;
static const uint32_t ACL_ERROR_GE_OP_KERNEL_TYPE_INVALID = 145018U;
static const uint32_t ACL_ERROR_GE_PLGMGR_PATH_INVALID = 145019U;
static const uint32_t ACL_ERROR_GE_FORMAT_INVALID = 145020U;
static const uint32_t ACL_ERROR_GE_SHAPE_INVALID = 145021U;
static const uint32_t ACL_ERROR_GE_DATATYPE_INVALID = 145022U;
static const uint32_t ACL_ERROR_GE_MEMORY_ALLOCATION = 245000U;
static const uint32_t ACL_ERROR_GE_MEMORY_OPERATE_FAILED = 245001U;
static const uint32_t ACL_ERROR_GE_INTERNAL_ERROR = 545000U;
static const uint32_t ACL_ERROR_GE_LOAD_MODEL = 545001U;
static const uint32_t ACL_ERROR_GE_EXEC_LOAD_MODEL_PARTITION_FAILED = 545002U;
static const uint32_t ACL_ERROR_GE_EXEC_LOAD_WEIGHT_PARTITION_FAILED = 545003U;
static const uint32_t ACL_ERROR_GE_EXEC_LOAD_TASK_PARTITION_FAILED = 545004U;
static const uint32_t ACL_ERROR_GE_EXEC_LOAD_KERNEL_PARTITION_FAILED = 545005U;
static const uint32_t ACL_ERROR_GE_EXEC_RELEASE_MODEL_DATA = 545006U;
static const uint32_t ACL_ERROR_GE_COMMAND_HANDLE = 545007U;
static const uint32_t ACL_ERROR_GE_GET_TENSOR_INFO = 545008U;
static const uint32_t ACL_ERROR_GE_UNLOAD_MODEL = 545009U;


static const int32_t ACL_ERROR_RT_PARAM_INVALID              = 107000; // param invalid
static const int32_t ACL_ERROR_RT_INVALID_DEVICEID           = 107001; // invalid device id
static const int32_t ACL_ERROR_RT_CONTEXT_NULL               = 107002; // current context null
static const int32_t ACL_ERROR_RT_STREAM_CONTEXT             = 107003; // stream not in current context
static const int32_t ACL_ERROR_RT_MODEL_CONTEXT              = 107004; // model not in current context
static const int32_t ACL_ERROR_RT_STREAM_MODEL               = 107005; // stream not in model
static const int32_t ACL_ERROR_RT_EVENT_TIMESTAMP_INVALID    = 107006; // event timestamp invalid
static const int32_t ACL_ERROR_RT_EVENT_TIMESTAMP_REVERSAL   = 107007; // event timestamp reversal
static const int32_t ACL_ERROR_RT_ADDR_UNALIGNED             = 107008; // memory address unaligned
static const int32_t ACL_ERROR_RT_FILE_OPEN                  = 107009; // open file failed
static const int32_t ACL_ERROR_RT_FILE_WRITE                 = 107010; // write file failed
static const int32_t ACL_ERROR_RT_STREAM_SUBSCRIBE           = 107011; // error subscribe stream
static const int32_t ACL_ERROR_RT_THREAD_SUBSCRIBE           = 107012; // error subscribe thread
static const int32_t ACL_ERROR_RT_GROUP_NOT_SET              = 107013; // group not set
static const int32_t ACL_ERROR_RT_GROUP_NOT_CREATE           = 107014; // group not create
static const int32_t ACL_ERROR_RT_STREAM_NO_CB_REG           = 107015; // callback not register to stream
static const int32_t ACL_ERROR_RT_INVALID_MEMORY_TYPE        = 107016; // invalid memory type
static const int32_t ACL_ERROR_RT_INVALID_HANDLE             = 107017; // invalid handle
static const int32_t ACL_ERROR_RT_INVALID_MALLOC_TYPE        = 107018; // invalid malloc type
static const int32_t ACL_ERROR_RT_WAIT_TIMEOUT               = 107019; // wait timeout

static const int32_t ACL_ERROR_RT_FEATURE_NOT_SUPPORT        = 207000; // feature not support
static const int32_t ACL_ERROR_RT_MEMORY_ALLOCATION          = 207001; // memory allocation error
static const int32_t ACL_ERROR_RT_MEMORY_FREE                = 207002; // memory free error
static const int32_t ACL_ERROR_RT_AICORE_OVER_FLOW           = 207003; // aicore over flow
static const int32_t ACL_ERROR_RT_NO_DEVICE                  = 207004; // no device
static const int32_t ACL_ERROR_RT_RESOURCE_ALLOC_FAIL        = 207005; // resource alloc fail
static const int32_t ACL_ERROR_RT_NO_PERMISSION              = 207006; // no permission
static const int32_t ACL_ERROR_RT_NO_EVENT_RESOURCE          = 207007; // no event resource
static const int32_t ACL_ERROR_RT_NO_STREAM_RESOURCE         = 207008; // no stream resource
static const int32_t ACL_ERROR_RT_NO_NOTIFY_RESOURCE         = 207009; // no notify resource
static const int32_t ACL_ERROR_RT_NO_MODEL_RESOURCE          = 207010; // no model resource
static const int32_t ACL_ERROR_RT_NO_CDQ_RESOURCE            = 207011; // no cdq resource
static const int32_t ACL_ERROR_RT_OVER_LIMIT                 = 207012; // over limit
static const int32_t ACL_ERROR_RT_QUEUE_EMPTY                = 207013; // queue is empty
static const int32_t ACL_ERROR_RT_QUEUE_FULL                 = 207014; // queue is full
static const int32_t ACL_ERROR_RT_REPEATED_INIT              = 207015; // repeated init
static const int32_t ACL_ERROR_RT_AIVEC_OVER_FLOW            = 207016; // aivec over flow

static const int32_t ACL_ERROR_RT_INTERNAL_ERROR             = 507000; // runtime internal error
static const int32_t ACL_ERROR_RT_TS_ERROR                   = 507001; // ts internel error
static const int32_t ACL_ERROR_RT_STREAM_TASK_FULL           = 507002; // task full in stream
static const int32_t ACL_ERROR_RT_STREAM_TASK_EMPTY          = 507003; // task empty in stream
static const int32_t ACL_ERROR_RT_STREAM_NOT_COMPLETE        = 507004; // stream not complete
static const int32_t ACL_ERROR_RT_END_OF_SEQUENCE            = 507005; // end of sequence
static const int32_t ACL_ERROR_RT_EVENT_NOT_COMPLETE         = 507006; // event not complete
static const int32_t ACL_ERROR_RT_CONTEXT_RELEASE_ERROR      = 507007; // context release error
static const int32_t ACL_ERROR_RT_SOC_VERSION                = 507008; // soc version error
static const int32_t ACL_ERROR_RT_TASK_TYPE_NOT_SUPPORT      = 507009; // task type not support
static const int32_t ACL_ERROR_RT_LOST_HEARTBEAT             = 507010; // ts lost heartbeat
static const int32_t ACL_ERROR_RT_MODEL_EXECUTE              = 507011; // model execute failed
static const int32_t ACL_ERROR_RT_REPORT_TIMEOUT             = 507012; // report timeout
static const int32_t ACL_ERROR_RT_SYS_DMA                    = 507013; // sys dma error
static const int32_t ACL_ERROR_RT_AICORE_TIMEOUT             = 507014; // aicore timeout
static const int32_t ACL_ERROR_RT_AICORE_EXCEPTION           = 507015; // aicore exception
static const int32_t ACL_ERROR_RT_AICORE_TRAP_EXCEPTION      = 507016; // aicore trap exception
static const int32_t ACL_ERROR_RT_AICPU_TIMEOUT              = 507017; // aicpu timeout
static const int32_t ACL_ERROR_RT_AICPU_EXCEPTION            = 507018; // aicpu exception
static const int32_t ACL_ERROR_RT_AICPU_DATADUMP_RSP_ERR     = 507019; // aicpu datadump response error
static const int32_t ACL_ERROR_RT_AICPU_MODEL_RSP_ERR        = 507020; // aicpu model operate response error
static const int32_t ACL_ERROR_RT_PROFILING_ERROR            = 507021; // profiling error
static const int32_t ACL_ERROR_RT_IPC_ERROR                  = 507022; // ipc error
static const int32_t ACL_ERROR_RT_MODEL_ABORT_NORMAL         = 507023; // model abort normal
static const int32_t ACL_ERROR_RT_KERNEL_UNREGISTERING       = 507024; // kernel unregistering
static const int32_t ACL_ERROR_RT_RINGBUFFER_NOT_INIT        = 507025; // ringbuffer not init
static const int32_t ACL_ERROR_RT_RINGBUFFER_NO_DATA         = 507026; // ringbuffer no data
static const int32_t ACL_ERROR_RT_KERNEL_LOOKUP              = 507027; // kernel lookup error
static const int32_t ACL_ERROR_RT_KERNEL_DUPLICATE           = 507028; // kernel register duplicate
static const int32_t ACL_ERROR_RT_DEBUG_REGISTER_FAIL        = 507029; // debug register failed
static const int32_t ACL_ERROR_RT_DEBUG_UNREGISTER_FAIL      = 507030; // debug unregister failed
static const int32_t ACL_ERROR_RT_LABEL_CONTEXT              = 507031; // label not in current context
static const int32_t ACL_ERROR_RT_PROGRAM_USE_OUT            = 507032; // program register num use out
static const int32_t ACL_ERROR_RT_DEV_SETUP_ERROR            = 507033; // device setup error
static const int32_t ACL_ERROR_RT_VECTOR_CORE_TIMEOUT        = 507034; // vector core timeout
static const int32_t ACL_ERROR_RT_VECTOR_CORE_EXCEPTION      = 507035; // vector core exception
static const int32_t ACL_ERROR_RT_VECTOR_CORE_TRAP_EXCEPTION = 507036; // vector core trap exception
static const int32_t ACL_ERROR_RT_CDQ_BATCH_ABNORMAL         = 507037; // cdq alloc batch abnormal
static const int32_t ACL_ERROR_RT_DIE_MODE_CHANGE_ERROR      = 507038; // can not change die mode
static const int32_t ACL_ERROR_RT_DIE_SET_ERROR              = 507039; // single die mode can not set die
static const int32_t ACL_ERROR_RT_INVALID_DIEID              = 507040; // invalid die id
static const int32_t ACL_ERROR_RT_DIE_MODE_NOT_SET           = 507041; // die mode not set

static const int32_t ACL_ERROR_RT_DRV_INTERNAL_ERROR         = 507899; // drv internal error
static const int32_t ACL_ERROR_RT_AICPU_INTERNAL_ERROR       = 507900; // aicpu internal error
static const int32_t ACL_ERROR_RT_SOCKET_CLOSE               = 507901; // hdc disconnect

)");
    if (acl_error_map.count(error))
        return acl_error_map[error];
    return "unknown " + std::to_string((int)error);
}