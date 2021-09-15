
#pragma once
#ifdef _WIN32
#include <windows.h>
#include "common.h"

namespace jittor {

EXTERN_LIB void raise_win_error(int ierr);
EXTERN_LIB void raise_cxx_exception(DWORD code, const EXCEPTION_RECORD* pr);
EXTERN_LIB DWORD HandleException(EXCEPTION_POINTERS *ptrs,
                             DWORD *pdw, EXCEPTION_RECORD *record);

#define _JT_SEH_TRY \
    DWORD dwExceptionCode = 0; \
    EXCEPTION_RECORD record; \
    __try {

#define _JT_SEH_CATCH \
    } \
    __except (HandleException(GetExceptionInformation(), \
                              &dwExceptionCode, &record)) { \
        raise_cxx_exception(dwExceptionCode, &record); \
    }

#define _JT_SEH_START \
    return [&]() { \
    _JT_SEH_TRY; \
    return [&]() {

#define _JT_SEH_END \
    }(); \
    _JT_SEH_CATCH; \
    }(); \
    

#define _JT_SEH_START2 \
    [&]() { \
    _JT_SEH_TRY;

#define _JT_SEH_END2 \
    _JT_SEH_CATCH; \
    }();

#ifdef JT_SEH_FULL


#define _JT_SEH_START3 \
    return [&]() { \
    _JT_SEH_TRY; \
    return [&]() {

#define _JT_SEH_END3 \
    }(); \
    _JT_SEH_CATCH; \
    }(); \

#else

#define _JT_SEH_START3
#define _JT_SEH_END3

#endif

}
#else

#define _JT_SEH_TRY
#define _JT_SEH_CATCH
#define _JT_SEH_START
#define _JT_SEH_END
#define _JT_SEH_START2
#define _JT_SEH_END2
#define _JT_SEH_START3
#define _JT_SEH_END3

#endif