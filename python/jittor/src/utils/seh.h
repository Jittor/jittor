
#pragma once
#ifdef _WIN32
#include <windows.h>
#include <exception>
#include <eh.h>
#include <sstream>
#include "common.h"

namespace jittor {

using std::stringstream;
    
inline void raise_win_error(int ierr) {
    DWORD err = (DWORD)ierr;
    WCHAR *s_buf = NULL; /* Free via LocalFree */
    stringstream message;

    if (err==0) {
        err = GetLastError();
    }
    
    auto len = FormatMessageW(
        /* Error API error */
        FORMAT_MESSAGE_ALLOCATE_BUFFER |
        FORMAT_MESSAGE_FROM_SYSTEM |
        FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL,           /* no message source */
        err,
        MAKELANGID(LANG_NEUTRAL,
        SUBLANG_DEFAULT), /* Default language */
        (LPWSTR) &s_buf,
        0,              /* size not used */
        NULL);          /* no args */
        
    if (len==0) {
        /* Only seen this in out of mem situations */
        message << "Windows Error " << err;
        s_buf = NULL;
    } else {
        /* remove trailing cr/lf and dots */
        while (len > 0 && (s_buf[len-1] <= L' ' || s_buf[len-1] == L'.'))
            s_buf[--len] = L'\0';
        message << s_buf;
    }
    if (s_buf)
        LocalFree(s_buf);
    throw std::runtime_error(message.str());
}

inline void raise_cxx_exception(unsigned int code, _EXCEPTION_POINTERS* pExp) {
    std::cerr << "raise_cxx_exception " << code << std::endl;
    EXCEPTION_RECORD* pr = pExp->ExceptionRecord;

    /* The 'code' is a normal win32 error code so it could be handled by
    raise_win_error(). However, for some errors, we have additional
    information not included in the error code. We handle those here and
    delegate all others to the generic function. */
    stringstream message;
    switch (code) {
    case EXCEPTION_ACCESS_VIOLATION:
        /* The thread attempted to read from or write
           to a virtual address for which it does not
           have the appropriate access. */
        if (pr->ExceptionInformation[0] == 0)
            message << "exception: access violation reading " << (void*)pr->ExceptionInformation[1];
        else
            message << "exception: access violation writing " << (void*)pr->ExceptionInformation[1];
        break;

    case EXCEPTION_BREAKPOINT:
        /* A breakpoint was encountered. */
        message << "exception: breakpoint encountered";
        break;

    case EXCEPTION_DATATYPE_MISALIGNMENT:
        /* The thread attempted to read or write data that is
           misaligned on hardware that does not provide
           alignment. For example, 16-bit values must be
           aligned on 2-byte boundaries, 32-bit values on
           4-byte boundaries, and so on. */
        message << "exception: datatype misalignment";
        break;

    case EXCEPTION_SINGLE_STEP:
        /* A trace trap or other single-instruction mechanism
           signaled that one instruction has been executed. */
        message << "exception: single step";
        break;

    case EXCEPTION_ARRAY_BOUNDS_EXCEEDED:
        /* The thread attempted to access an array element
           that is out of bounds, and the underlying hardware
           supports bounds checking. */
        message << "exception: array bounds exceeded";
        break;

    case EXCEPTION_FLT_DENORMAL_OPERAND:
        /* One of the operands in a floating-point operation
           is denormal. A denormal value is one that is too
           small to represent as a standard floating-point
           value. */
        message << "exception: floating-point operand denormal";
        break;

    case EXCEPTION_FLT_DIVIDE_BY_ZERO:
        /* The thread attempted to divide a floating-point
           value by a floating-point divisor of zero. */
        message << "exception: float divide by zero";
        break;

    case EXCEPTION_FLT_INEXACT_RESULT:
        /* The result of a floating-point operation cannot be
           represented exactly as a decimal fraction. */
        message << "exception: float inexact";
        break;

    case EXCEPTION_FLT_INVALID_OPERATION:
        /* This exception represents any floating-point
           exception not included in this list. */
        message << "exception: float invalid operation";
        break;

    case EXCEPTION_FLT_OVERFLOW:
        /* The exponent of a floating-point operation is
           greater than the magnitude allowed by the
           corresponding type. */
        message << "exception: float overflow";
        break;

    case EXCEPTION_FLT_STACK_CHECK:
        /* The stack overflowed or underflowed as the result
           of a floating-point operation. */
        message << "exception: stack over/underflow";
        break;

    case EXCEPTION_STACK_OVERFLOW:
        /* The stack overflowed or underflowed as the result
           of a floating-point operation. */
        message << "exception: stack overflow";
        break;

    case EXCEPTION_FLT_UNDERFLOW:
        /* The exponent of a floating-point operation is less
           than the magnitude allowed by the corresponding
           type. */
        message << "exception: float underflow";
        break;

    case EXCEPTION_INT_DIVIDE_BY_ZERO:
        /* The thread attempted to divide an integer value by
           an integer divisor of zero. */
        message << "exception: integer divide by zero";
        break;

    case EXCEPTION_INT_OVERFLOW:
        /* The result of an integer operation caused a carry
           out of the most significant bit of the result. */
        message << "exception: integer overflow";
        break;

    case EXCEPTION_PRIV_INSTRUCTION:
        /* The thread attempted to execute an instruction
           whose operation is not allowed in the current
           machine mode. */
        message << "exception: privileged instruction";
        break;

    case EXCEPTION_NONCONTINUABLE_EXCEPTION:
        /* The thread attempted to continue execution after a
           noncontinuable exception occurred. */
        message << "exception: nocontinuable";
        break;

    case 0xE06D7363:
        /* magic number(0xE06D7363) of c++ exception:
            https://devblogs.microsoft.com/oldnewthing/20100730-00/?p=13273
        */
        message << "Error c++ exception";
        break;

    default:
        raise_win_error(code);
        break;
    }
    // std::cout << message.str() << std::endl;
    throw std::runtime_error(message.str());
}

}
#define SEH_HOOK int _seh_hook = (_set_se_translator(raise_cxx_exception), 0)
#else
#define SEH_HOOK
#endif