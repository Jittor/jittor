// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "utils/cache_compile.h"
#include "pyjt/py_converter.h"
#include "pyjt/py_arg_printer.h"
#ifdef __clang__
#pragma clang diagnostic ignored "-Wdefaulted-function-deleted"
#endif
#ifdef __GNUC__
#endif
#include <sys/prctl.h>
#include <signal.h>
#include <iterator>
#include <algorithm>
#include <cstring>

namespace jittor {

void init_subprocess() {
    prctl(PR_SET_PDEATHSIG, SIGKILL);
}

static void __log(
    const std::string& fileline,
    char level,
    int verbose,
    const std::string& s)
{
    // return if verbose level not match
    if (level=='i' && !(
        jittor::log_vprefix.size() ?
            jittor::check_vlog(fileline.c_str(), verbose) :
            verbose <= jittor::log_v))
        return;
    if (level != 'f')
        jittor::LogVoidify() &&
        jittor::Log(fileline.c_str(), level, verbose) << s;
    else
        jittor::LogFatalVoidify() &&
        jittor::Log(fileline.c_str(), level, verbose) << s;
}

// Buffer that writes to Python instead of C++
class pythonbuf : public std::streambuf {
private:
    using traits_type = std::streambuf::traits_type;

    const size_t buf_size;
    std::unique_ptr<char[]> d_buffer;
    PyObject* _pywrite;
    PyObject* _pyflush;

    int overflow(int c) override {
        if (!traits_type::eq_int_type(c, traits_type::eof())) {
            *pptr() = traits_type::to_char_type(c);
            pbump(1);
        }
        return sync() == 0 ? traits_type::not_eof(c) : traits_type::eof();
    }

    // Computes how many bytes at the end of the buffer are part of an
    // incomplete sequence of UTF-8 bytes.
    // Precondition: pbase() < pptr()
    size_t utf8_remainder() const {
        const auto rbase = std::reverse_iterator<char *>(pbase());
        const auto rpptr = std::reverse_iterator<char *>(pptr());
        auto is_ascii = [](char c) {
            return (static_cast<unsigned char>(c) & 0x80) == 0x00;
        };
        auto is_leading = [](char c) {
            return (static_cast<unsigned char>(c) & 0xC0) == 0xC0;
        };
        auto is_leading_2b = [](char c) {
            return static_cast<unsigned char>(c) <= 0xDF;
        };
        auto is_leading_3b = [](char c) {
            return static_cast<unsigned char>(c) <= 0xEF;
        };
        // If the last character is ASCII, there are no incomplete code points
        if (is_ascii(*rpptr))
            return 0;
        // Otherwise, work back from the end of the buffer and find the first
        // UTF-8 leading byte
        const auto rpend   = rbase - rpptr >= 3 ? rpptr + 3 : rbase;
        const auto leading = std::find_if(rpptr, rpend, is_leading);
        if (leading == rbase)
            return 0;
        const auto dist    = static_cast<size_t>(leading - rpptr);
        size_t remainder   = 0;

        if (dist == 0)
            remainder = 1; // 1-byte code point is impossible
        else if (dist == 1)
            remainder = is_leading_2b(*leading) ? 0 : dist + 1;
        else if (dist == 2)
            remainder = is_leading_3b(*leading) ? 0 : dist + 1;
        // else if (dist >= 3), at least 4 bytes before encountering an UTF-8
        // leading byte, either no remainder or invalid UTF-8.
        // Invalid UTF-8 will cause an exception later when converting
        // to a Python string, so that's not handled here.
        return remainder;
    }

    // This function must be non-virtual to be called in a destructor. If the
    // rare MSVC test failure shows up with this version, then this should be
    // simplified to a fully qualified call.
    int _sync() {
        if (pbase() != pptr()) { // If buffer is not empty
            if (pbase() != pptr()) { // Check again under the lock
                // This subtraction cannot be negative, so dropping the sign.
                auto size        = static_cast<size_t>(pptr() - pbase());
                size_t remainder = utf8_remainder();

                if (size > remainder) {
                    string line(pbase(), size - remainder);
                    pywrite(line);
                    pyflush();
                }

                // Copy the remainder at the end of the buffer to the beginning:
                if (remainder > 0)
                    std::memmove(pbase(), pptr() - remainder, remainder);
                setp(pbase(), epptr());
                pbump(static_cast<int>(remainder));
            }
        }
        return 0;
    }

    int sync() override {
        return _sync();
    }

    void pywrite(const string& s) {
        PyObjHolder pys(to_py_object<string>(s));
        PyObjHolder args(PyTuple_New(1));
        PyTuple_SET_ITEM(args.obj, 0, pys.release());
        PyObjHolder ret(PyObject_Call(_pywrite, args.obj, nullptr));
    }

    void pyflush() {
        PyObjHolder args(PyTuple_New(0));
        PyObjHolder ret(PyObject_Call(_pyflush, args.obj, nullptr));
    }

public:

    pythonbuf(PyObject* pyostream, size_t buffer_size = 1024)
        : buf_size(buffer_size),
          d_buffer(new char[buf_size]) {
        
        PyObjHolder pywrite(PyObject_GetAttrString(pyostream, "write"));
        _pywrite = pywrite.release();
        PyObjHolder pyflush(PyObject_GetAttrString(pyostream, "flush"));
        _pyflush = pyflush.release();
        setp(d_buffer.get(), d_buffer.get() + buf_size - 1);

    }

    pythonbuf(pythonbuf&&) = default;

    /// Sync before destroy
    ~pythonbuf() override {
        _sync();
    }
};

static void ostream_redirect(bool stdout, bool stderr) {
    if (stdout) {
        PyObjHolder a(PyImport_ImportModule("sys"));
        PyObjHolder b(PyObject_GetAttrString(a.obj,"stdout"));
        auto buf = new pythonbuf(b.obj);
        std::cout.rdbuf(buf);
    }
    if (stderr) {
        PyObjHolder a(PyImport_ImportModule("sys"));
        PyObjHolder b(PyObject_GetAttrString(a.obj,"stderr"));
        auto buf = new pythonbuf(b.obj);
        std::cerr.rdbuf(buf);
    }
}

static void pyjt_def_core(PyObject* m) {
    static PyMethodDef defs[] = {
    { R""(cache_compile)"",
    
    (PyCFunction)(PyObject* (*)(PyObject*,PyObject**,int64,PyObject*))[](PyObject* self, PyObject** args, int64 n, PyObject* kw) -> PyObject* {
        try {
            ;
            uint64 arg_filled=0;
            (void)arg_filled;
            
            if (n+(kw?Py_SIZE(kw):0)<=3 && n+(kw?Py_SIZE(kw):0)>=1 && is_type<string>(args[0])) {
                
                    ;
                    string arg0 = from_py_object<string>(args[0]);
        
                    ;
                    string arg1;
                    if (n>1) {
                        CHECK((is_type<string>(args[1])));
                        arg1 = from_py_object<string>(args[1]);
                        arg_filled |= 1ull << 1;
                    }
        
                    ;
                    string arg2;
                    if (n>2) {
                        CHECK((is_type<string>(args[2])));
                        arg2 = from_py_object<string>(args[2]);
                        arg_filled |= 1ull << 2;
                    }
        
                    CHECK(!PyErr_Occurred());
    ;
                
                    if (kw) {
                        auto kw_n = Py_SIZE(kw);
                        for (int i=0; i<kw_n; i++) {
                            auto ko = PyTuple_GET_ITEM(kw, i);
                            auto vo = args[i+n];
                            auto ks = PyUnicode_AsUTF8(ko);
                            uint khash = hash(ks);
                            
                            if (khash == 308594u) {
                                // hash match cmd
                                CHECK((is_type<string>(vo)));
                                arg0 = from_py_object<string>(vo);
                                arg_filled |= 1ull << 0;
                                continue;
                            }
                            
                            if (khash == 370544278u) {
                                // hash match cache_path
                                CHECK((is_type<string>(vo)));
                                arg1 = from_py_object<string>(vo);
                                arg_filled |= 1ull << 1;
                                continue;
                            }
                            
                            if (khash == 1219769050u) {
                                // hash match jittor_path
                                CHECK((is_type<string>(vo)));
                                arg2 = from_py_object<string>(vo);
                                arg_filled |= 1ull << 2;
                                continue;
                            }
                            
                            LOGf << "Not a valid keyword:" << ks;
                        }
                    }
    
                    if (!(arg_filled & (1ull<<1))) {
                        arg1 = "";
                    }
        
                    if (!(arg_filled & (1ull<<2))) {
                        arg2 = "";
                    }
        ;
                return to_py_object<bool>((jit_compiler::cache_compile(arg0,arg1,arg2)));
            }
            
            LOGf << "Not a valid call.";
        } catch (const std::exception& e) {
            if (!PyErr_Occurred()) {
                PyErr_Format(PyExc_RuntimeError, e.what());
            }
        }
        return nullptr;
    }
    ,
    METH_FASTCALL | METH_KEYWORDS,
    R""(Declaration:
bool cache_compile(const string& cmd, const string& cache_path="", const string& jittor_path="")

)""
    },
    { R""(log)"",
    
    (PyCFunction)(PyObject* (*)(PyObject*,PyObject**,int64,PyObject*))[](PyObject* self, PyObject** args, int64 n, PyObject* kw) -> PyObject* {
        try {
            ;
            uint64 arg_filled=0;
            (void)arg_filled;
            
            if (n+(kw?Py_SIZE(kw):0)<=4 && n+(kw?Py_SIZE(kw):0)>=4 && is_type<std::string>(args[0]) && PyUnicode_CheckExact(args[1]) && PyLong_CheckExact(args[2]) && is_type<std::string>(args[3])) {
                
                    ;
                    std::string arg0 = from_py_object<std::string>(args[0]);
        
                    ;
                    const char* arg1 = PyUnicode_AsUTF8(args[1]);
        
                    ;
                    int arg2 = PyLong_AsLong(args[2]);
        
                    ;
                    std::string arg3 = from_py_object<std::string>(args[3]);
        
                    CHECK(!PyErr_Occurred());
    ;
                
                    if (kw) {
                        auto kw_n = Py_SIZE(kw);
                        for (int i=0; i<kw_n; i++) {
                            auto ko = PyTuple_GET_ITEM(kw, i);
                            auto vo = args[i+n];
                            auto ks = PyUnicode_AsUTF8(ko);
                            uint khash = hash(ks);
                            
                            if (khash == 3883819440u) {
                                // hash match fileline
                                CHECK((is_type<std::string>(vo)));
                                arg0 = from_py_object<std::string>(vo);
                                arg_filled |= 1ull << 0;
                                continue;
                            }
                            
                            if (khash == 1005433988u) {
                                // hash match level
                                CHECK((PyUnicode_CheckExact(vo)));
                                arg1 = PyUnicode_AsUTF8(vo);
                                arg_filled |= 1ull << 1;
                                continue;
                            }
                            
                            if (khash == 2796496354u) {
                                // hash match verbose
                                CHECK((PyLong_CheckExact(vo)));
                                arg2 = PyLong_AsLong(vo);
                                arg_filled |= 1ull << 2;
                                continue;
                            }
                            
                            if (khash == 115u) {
                                // hash match s
                                CHECK((is_type<std::string>(vo)));
                                arg3 = from_py_object<std::string>(vo);
                                arg_filled |= 1ull << 3;
                                continue;
                            }
                            
                            LOGf << "Not a valid keyword:" << ks;
                        }
                    }
    ;
                return GET_PY_NONE((__log(arg0,arg1[0],arg2,arg3)));
            }
            
            LOGf << "Not a valid call.";
        } catch (const std::exception& e) {
            if (!PyErr_Occurred()) {
                PyErr_Format(PyExc_RuntimeError, e.what());
            }
        }
        return nullptr;
    }
    ,
    METH_FASTCALL | METH_KEYWORDS,
    R""(Declaration:
void log(const std::string& fileline, const char* level, int verbose, const std::string& s)

)""
    },
    { R""(init_subprocess)"",
    
    (PyCFunction)(PyObject* (*)(PyObject*,PyObject**,int64,PyObject*))[](PyObject* self, PyObject** args, int64 n, PyObject* kw) -> PyObject* {
        try {
            ;
            uint64 arg_filled=0;
            (void)arg_filled;
            
            if (n<=0 && n>=0) {
                ;
                ;
                return GET_PY_NONE((init_subprocess()));
            }
            
            LOGf << "Not a valid call.";
        } catch (const std::exception& e) {
            if (!PyErr_Occurred()) {
                PyErr_Format(PyExc_RuntimeError, e.what());
            }
        }
        return nullptr;
    }
    ,
    METH_FASTCALL | METH_KEYWORDS,
    R""(Declaration:
void init_subprocess()

)""
    },
    { R""(log_capture_start)"",
    
    (PyCFunction)(PyObject* (*)(PyObject*,PyObject**,int64,PyObject*))[](PyObject* self, PyObject** args, int64 n, PyObject* kw) -> PyObject* {
        try {
            ;
            uint64 arg_filled=0;
            (void)arg_filled;
            
            if (n<=0 && n>=0) {
                ;
                ;
                return GET_PY_NONE((log_capture_start()));
            }
            
            LOGf << "Not a valid call.";
        } catch (const std::exception& e) {
            if (!PyErr_Occurred()) {
                PyErr_Format(PyExc_RuntimeError, e.what());
            }
        }
        return nullptr;
    }
    ,
    METH_FASTCALL | METH_KEYWORDS,
    R""(Declaration:
void log_capture_start()

)""
    },
    { R""(log_capture_stop)"",
    
    (PyCFunction)(PyObject* (*)(PyObject*,PyObject**,int64,PyObject*))[](PyObject* self, PyObject** args, int64 n, PyObject* kw) -> PyObject* {
        try {
            ;
            uint64 arg_filled=0;
            (void)arg_filled;
            
            if (n<=0 && n>=0) {
                ;
                ;
                return GET_PY_NONE((log_capture_stop()));
            }
            
            LOGf << "Not a valid call.";
        } catch (const std::exception& e) {
            if (!PyErr_Occurred()) {
                PyErr_Format(PyExc_RuntimeError, e.what());
            }
        }
        return nullptr;
    }
    ,
    METH_FASTCALL | METH_KEYWORDS,
    R""(Declaration:
void log_capture_stop()

)""
    },
    { R""(log_capture_read)"",
    
    (PyCFunction)(PyObject* (*)(PyObject*,PyObject**,int64,PyObject*))[](PyObject* self, PyObject** args, int64 n, PyObject* kw) -> PyObject* {
        try {
            ;
            uint64 arg_filled=0;
            (void)arg_filled;
            
            if (n<=0 && n>=0) {
                ;
                ;
                // return GET_PY_NONE((log_capture_stop()));
                auto ret = log_capture_read();
                return to_py_object(move(ret));
            }
            
            LOGf << "Not a valid call.";
        } catch (const std::exception& e) {
            if (!PyErr_Occurred()) {
                PyErr_Format(PyExc_RuntimeError, e.what());
            }
        }
        return nullptr;
    }
    ,
    METH_FASTCALL | METH_KEYWORDS,
    R""(Declaration:
void log_capture_read()

)""
    },
    { R""(ostream_redirect)"",
    
    (PyCFunction)(PyObject* (*)(PyObject*,PyObject**,int64,PyObject*))[](PyObject* self, PyObject** args, int64 n, PyObject* kw) -> PyObject* {
        try {
            ;
            uint64 arg_filled=0;
            (void)arg_filled;
            
            if (n+(kw?Py_SIZE(kw):0)<=2 && n+(kw?Py_SIZE(kw):0)>=2 && is_type<bool>(args[0]) && is_type<bool>(args[1])) {
                
                    ;
                    bool arg0 = from_py_object<bool>(args[0]);
        
                    ;
                    bool arg1 = from_py_object<bool>(args[1]);
        
                    CHECK(!PyErr_Occurred());
    ;
                
                    if (kw) {
                        auto kw_n = Py_SIZE(kw);
                        for (int i=0; i<kw_n; i++) {
                            auto ko = PyTuple_GET_ITEM(kw, i);
                            auto vo = args[i+n];
                            auto ks = PyUnicode_AsUTF8(ko);
                            uint khash = hash(ks);
                            
                            if (khash == 3635812397u) {
                                // hash match stdout
                                CHECK((is_type<bool>(vo)));
                                arg0 = from_py_object<bool>(vo);
                                arg_filled |= 1ull << 0;
                                continue;
                            }
                            
                            if (khash == 2600128022u) {
                                // hash match stderr
                                CHECK((is_type<bool>(vo)));
                                arg1 = from_py_object<bool>(vo);
                                arg_filled |= 1ull << 1;
                                continue;
                            }
                            
                            LOGf << "Not a valid keyword:" << ks;
                        }
                    }
    ;
                return GET_PY_NONE((ostream_redirect(arg0,arg1)));
            }
            
            LOGf << "Not a valid call.";
        } catch (const std::exception& e) {
            if (!PyErr_Occurred()) {
                PyErr_Format(PyExc_RuntimeError, e.what());
            }
        }
        return nullptr;
    }
    ,
    METH_FASTCALL | METH_KEYWORDS,
    R""(Declaration:
void ostream_redirect(bool stdout, bool stderr)

)""
    },{0,0,0,0}
    };
    ASSERT(PyModule_AddFunctions(m, defs)==0);
}

}


static void init_module(PyModuleDef* mdef, PyObject* m) {
    mdef->m_doc = "Inner c++ core of jittor_utils";
    jittor::pyjt_def_core(m);
}
PYJT_MODULE_INIT(jit_utils_core);
