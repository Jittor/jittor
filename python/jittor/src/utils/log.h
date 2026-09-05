// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include <string>
#include <sstream>
#include <functional>
#include <iostream>
#include <type_traits>
#include <limits>
#include <cstdlib>
#include <cerrno>
#include <cctype>
#include "types.h"

namespace jittor {

// define in tracer.cc
void print_trace();
// Signal-handler-safe: hands the frames to a process forked before
// the crash. See tracer.cc.
void print_trace_from_signal(int signal, void* fault_pc, void* caller_pc);
void start_trace_helper();
void stop_trace_helper();
void breakpoint();
#ifdef _WIN32
string GbkToUtf8(const char *src_str);
string Utf8ToGbk(const char *src_str);
#define _to_winstr(x) Utf8ToGbk(x.c_str())
#define _from_winstr(x) GbkToUtf8(x.c_str())
#else
#define _to_winstr(x) (x)
#define _from_winstr(x) (x)
#endif
    
constexpr int32_t basename_index(const char * const path, const int32_t index = 0, const int32_t slash_index = -1) {
   return path[index]
       ? ((path[index] == '/' || path[index] == '\\')
           ? basename_index (path, index + 1, index)
           : basename_index (path, index + 1, slash_index)
           )
       : (slash_index + 1);
}

#define STRINGIZE_DETAIL(x) #x
#define STRINGIZE(x) STRINGIZE_DETAIL(x)
    
#define __FILELINE__ \
    (&((__FILE__ ":" STRINGIZE(__LINE__))[jittor::basename_index(__FILE__)]))

#ifndef _WIN32
#define PREDICT_BRANCH_NOT_TAKEN(x) (__builtin_expect(x, 0))
#else
#define PREDICT_BRANCH_NOT_TAKEN(x) (x)
#endif


#ifdef _MSC_VER
#define STACK_ALLOC(T, a, n) T* a = (T*)_alloca(sizeof(T)*(n))
#define EXTERN_LIB extern __declspec(dllimport)
#define EXPORT_LIB __declspec(dllimport)
#else
#define STACK_ALLOC(T, a, n) T a[n]
#define EXTERN_LIB extern
#define EXPORT_LIB 
#endif

EXTERN_LIB uint32_t get_tid();
EXTERN_LIB bool g_supports_color;
EXTERN_LIB void print_prefix(std::ostream* out);

#ifdef _WIN32
constexpr char green[] = "\x1b[1;32m";
constexpr char red[] = "\x1b[1;31m";
constexpr char yellow[] = "\x1b[1;33m";


inline static void get_color(char level, int verbose, const char*& color_begin, const char*& color_end) {
    if (level == 'i' || level == 'I') {
        if (verbose == 0) color_begin = "\x1b[1;32m"; else
        if (verbose < 10) color_begin = "\x1b[1;32m"; else
        if (verbose < 100) color_begin = "\x1b[1;32m"; else
        if (verbose < 1000) color_begin = "\x1b[1;32m";
        else color_begin = "\x1b[1;32m";
    } else if (level == 'w')
        color_begin = yellow;
    else if (level == 'e')
        color_begin = red;
    else // level == 'f'
        color_begin = red;
    color_end = "\x1b[m";
}

#else
constexpr char green[] = "\033[38;5;2m";
constexpr char red[] = "\033[38;5;1m";
constexpr char yellow[] = "\033[38;5;3m";

inline static void get_color(char level, int verbose, const char*& color_begin, const char*& color_end) {
    if (level == 'i' || level == 'I') {
        if (verbose == 0) color_begin = "\033[38;5;2m"; else
        if (verbose < 10) color_begin = "\033[38;5;250m"; else
        if (verbose < 100) color_begin = "\033[38;5;244m"; else
        if (verbose < 1000) color_begin = "\033[38;5;238m";
        else color_begin = "\033[38;5;232m";
    } else if (level == 'w')
        color_begin = yellow;
    else if (level == 'e')
        color_begin = red;
    else // level == 'f'
        color_begin = red;
    color_end = "\033[m";
}

#endif

EXTERN_LIB void send_log(std::ostringstream&& out, char level, int verbose);
EXTERN_LIB void flush_log();
EXTERN_LIB void log_capture_start();
EXTERN_LIB void log_capture_stop();
EXTERN_LIB std::vector<std::map<string,string>> log_capture_read();
EXTERN_LIB string& get_thread_name();

struct Log {
    std::ostringstream out;
    const char* color_end;
    int verbose;
    char level;

    inline Log(const char* const fileline, char level, int verbose) {
        this->verbose = verbose;
        this->level = level;
        const char* color_begin;
        get_color(level, verbose, color_begin, color_end);
        if (g_supports_color) out << color_begin;
        out << '[' << level << ' ';
        print_prefix(&out);
        if (verbose) out << 'v' << verbose << ' ';
        out << fileline << ']';
    }

    inline void end() {
        if (g_supports_color) out << color_end;
        out << '\n';
        send_log(move(out), level, verbose);
    }
    inline void flush() { flush_log(); }

    template<class T>
    Log& operator<<(const T& a) { out << ' ' << a; return *this; }
    template<class T>
    Log& operator>>(const T& a) { out << a; return *this; }
};

struct LogVoidify {
    inline void operator&&(Log& log) { log.end(); }
};

struct JittorError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct UserError : JittorError {
    using JittorError::JittorError;
};

struct InternalInvariantError : JittorError {
    using JittorError::JittorError;
};

template <class Error>
struct LogErrorVoidify {
    inline void operator&&(Log& log) {
        log.flush();
        if (g_supports_color) log.out << log.color_end;
        throw Error(log.out.str());
    }
};

struct LogFatalVoidify {
    inline void operator&&(Log& log) {
        log.flush();
        if (g_supports_color) log.out << log.color_end;
        throw std::runtime_error(log.out.str()); 
    }
};

#define _LOGi(v) jittor::LogVoidify() && jittor::Log(__FILELINE__, 'i', v)
#define _LOGw(v) jittor::LogVoidify() && jittor::Log(__FILELINE__, 'w', v)
#define _LOGe(v) jittor::LogVoidify() && jittor::Log(__FILELINE__, 'e', v)
#define _LOGf(v) jittor::LogFatalVoidify() && jittor::Log(__FILELINE__, 'f', v)
#define LOGi _LOGi(0)
#define LOGw _LOGw(0)
#define LOGe _LOGe(0)
#define LOGf _LOGf(0)

#define _TYPED_ERROR_IF(error_type, cond) \
    !(cond) ? (void) 0 : \
        jittor::LogErrorVoidify<error_type>() && \
        jittor::Log(__FILELINE__, 'f', 0)

// A caller supplied an unsupported value, shape, dtype, or index. This path
// is part of the public API and is expected to be caught by the caller.
#define USER_ERROR \
    jittor::LogErrorVoidify<jittor::UserError>() && \
        jittor::Log(__FILELINE__, 'f', 0)
#define USER_CHECK(cond) \
    _TYPED_ERROR_IF(jittor::UserError, PREDICT_BRANCH_NOT_TAKEN(!(cond))) \
        << "User check failed: " #cond " "
#define USER_CHECKop(a, op, b) \
    _TYPED_ERROR_IF(jittor::UserError, !((a) op (b))) \
        << "User check failed" \
        << #a "(" >> a >> ") " #op " " #b "(" >> b >> ")"

// The framework reached a state its own implementation says is impossible.
// This remains a distinct exception while the legacy call sites are migrated,
// so existing top-level diagnostics keep working without confusing it with a
// recoverable user input error.
#define INTERNAL_ERROR \
    jittor::LogErrorVoidify<jittor::InternalInvariantError>() && \
        jittor::Log(__FILELINE__, 'f', 0)
#define INTERNAL_ASSERT(cond) \
    _TYPED_ERROR_IF(jittor::InternalInvariantError, \
        PREDICT_BRANCH_NOT_TAKEN(!(cond))) \
        << "Internal invariant failed: " #cond " "
#define INTERNAL_ASSERTop(a, op, b) \
    _TYPED_ERROR_IF(jittor::InternalInvariantError, !((a) op (b))) \
        << "Internal invariant failed" \
        << #a "(" >> a >> ") " #op " " #b "(" >> b >> ")"

#define _LOG(level, v) _LOG ## level(v)
#define LOG(level) _LOG(level, 0)

#define CHECK(cond) \
    LOG_IF(f, PREDICT_BRANCH_NOT_TAKEN(!(cond))) \
        << "Check failed: " #cond " "

#define _LOG_IF(level, cond, v) \
    !(cond) ? (void) 0 : _LOG(level, v)
#define LOG_IF(level, cond) _LOG_IF(level, cond, 0)

// Parse a whole environment value, or fail.
//
// The previous implementation encoded "parsed successfully" as "reading one
// more character failed", which is not the same question. `export log_v="1 "`
// leaves the trailing space in the stream without setting failbit, so the
// override was dropped and the default used; the single warning it emitted was
// level 'w', which log_silent swallows (send_log in log.cc). A flag that does
// not take effect and says nothing about it is worse than a startup failure,
// so an unparsable value is fatal now.
//
// std::from_chars would be the natural tool, but this header is included by
// nvcc-compiled translation units and the build is -std=c++14 (compiler.py),
// where <charconv> does not exist. strtoll/strtoull/strtold answer the same two
// questions: did it parse, and did it consume every character.
// strtoll/strtold skip leading whitespace; a flag's value is the whole string.
inline bool env_value_is_parsable(const string& s) {
    return !s.empty() && !std::isspace((unsigned char)s[0]);
}

inline bool parse_env_integer(const string& s, long long& out) {
    if (!env_value_is_parsable(s)) return false;
    errno = 0;
    char* end = nullptr;
    out = std::strtoll(s.c_str(), &end, 10);
    return errno == 0 && end == s.c_str() + s.size();
}

inline bool parse_env_integer(const string& s, unsigned long long& out) {
    // strtoull silently wraps a negative literal into a huge positive value.
    if (!env_value_is_parsable(s) || s[0] == '-') return false;
    errno = 0;
    char* end = nullptr;
    out = std::strtoull(s.c_str(), &end, 10);
    return errno == 0 && end == s.c_str() + s.size();
}

template<class T>
inline typename std::enable_if<std::is_integral<T>::value, bool>::type
parse_env_value(const string& s, T& out) {
    // uint8 flags used to go through operator>>(unsigned char&), which reads
    // one *character*: node_order=1 meant 49. Parsing as a number and range
    // checking is what every caller already assumed.
    typename std::conditional<std::is_signed<T>::value,
                              long long, unsigned long long>::type v = 0;
    if (!parse_env_integer(s, v)) return false;
    if (v > (decltype(v))std::numeric_limits<T>::max()) return false;
    if (std::is_signed<T>::value && v < (decltype(v))std::numeric_limits<T>::min())
        return false;
    out = (T)v;
    return true;
}

template<class T>
inline typename std::enable_if<std::is_floating_point<T>::value, bool>::type
parse_env_value(const string& s, T& out) {
    if (!env_value_is_parsable(s)) return false;
    errno = 0;
    char* end = nullptr;
    long double v = std::strtold(s.c_str(), &end);
    if (errno != 0 || end != s.c_str() + s.size()) return false;
    out = (T)v;
    return true;
}

template<class T>
inline typename std::enable_if<!std::is_arithmetic<T>::value, bool>::type
parse_env_value(const string& s, T& out) {
    // The only flags that get here are containers (cuda_archs is a vector<int>,
    // compile_options a map). Their extractors in types.h stop by *failing* at
    // end of input -- that is how the loop terminates -- so "did it fail" is
    // the wrong question for them and only "did it reach the end of the value"
    // can be asked. Whitespace is a separator inside these, not trailing junk.
    if (s.empty()) return false;
    std::istringstream is(s);
    is >> out;
    return is.eof();
}

template<class T> T get_from_env(const char* name,const T& _default) {
    auto ss = getenv(name);
    if (ss == NULL) return _default;
    string s = ss;
    T env = _default;
    if (parse_env_value(s, env))
        return env;
    LOGf << "Cannot parse environment variable" << name >> "=\"" >> s >> "\":"
        << "not a valid value for this flag. Fix it or unset it."
        << "(This used to be ignored, silently leaving the default"
        << _default >> ".)";
    return _default;
}

template<> std::string get_from_env(const char* name, const std::string& _default);

#define DECLARE_FLAG(type, name) \
EXTERN_LIB type name; \
EXTERN_LIB std::string doc_ ## name; \
EXTERN_LIB void set_ ## name (const type&);

#define DECLARE_RUNTIME_FLAG(type, name) \
EXTERN_LIB type& runtime_flag_ ## name (); \
EXTERN_LIB std::string doc_ ## name; \
EXTERN_LIB void set_ ## name (const type&);


#ifdef JIT

#define DEFINE_FLAG(type, name, default, doc) \
    DECLARE_FLAG(type, name)
#define DEFINE_FLAG_WITH_SETTER(type, name, default, doc) \
    DECLARE_FLAG(type, name)
#define DEFINE_RUNTIME_FLAG(type, name, default, doc) \
    DECLARE_RUNTIME_FLAG(type, name)
#define DEFINE_RUNTIME_FLAG_WITH_SETTER(type, name, default, doc) \
    DECLARE_RUNTIME_FLAG(type, name)

#else

#define DEFINE_FLAG(type, name, default, doc) \
    DECLARE_FLAG(type, name) \
    type name; \
    std::string doc_ ## name = doc; \
    void set_ ## name (const type& value) { \
        name = value; \
    }; \
    void init_ ## name (const type& value) { \
        name = value; \
        if (getenv(#name)) LOGi << "Load " #name":" << value; \
    }; \
    int caller_ ## name = (init_ ## name (jittor::get_from_env<type>(#name, default)), 0);

// The setter runs *after* the assignment and is handed both values.
//
// It used to run before, so every setter saw the flag still holding the old
// value. Setters that needed the new one wrote it themselves first (tracer.cc's
// setter_gdb_path, allocator.cc's setter_use_cuda_host_allocator, cuda_flags.cc's
// setter_sync_run, which did nothing else), and each of those hand-written
// assignments was a chance to forget. A setter that threw also left the flag
// untouched while the exception blamed the value: assignment and side effect
// were not one operation. Now the assignment happens first and is rolled back
// if the setter throws, so the pair either both take effect or neither does.
#define DEFINE_FLAG_WITH_SETTER(type, name, default, doc) \
    DECLARE_FLAG(type, name) \
    type name; \
    std::string doc_ ## name = doc; \
    void setter_ ## name (const type& old_value, const type& new_value); \
    void set_ ## name (const type& value) { \
        type old_value = name; \
        name = value; \
        try { \
            setter_ ## name (old_value, value); \
        } catch (...) { \
            name = old_value; \
            throw; \
        } \
    }; \
    void init_ ## name (const type& value) { \
        type old_value = name; \
        name = value; \
        setter_ ## name (old_value, value); \
        if (getenv(#name)) LOGi << "Load " #name":" << value; \
    }; \
    int caller_ ## name = (init_ ## name (jittor::get_from_env<type>(#name, default)), 0);

// Runtime flags retain the same initialization/setter protocol, but the core
// accessor supplies their storage. No namespace-scope reference is initialized.
#define DEFINE_RUNTIME_FLAG(type, name, default, doc) \
    DECLARE_RUNTIME_FLAG(type, name) \
    std::string doc_ ## name = doc; \
    void set_ ## name (const type& value) { \
        runtime_flag_ ## name () = value; \
    }; \
    void init_ ## name (const type& value) { \
        runtime_flag_ ## name () = value; \
        if (getenv(#name)) LOGi << "Load " #name":" << value; \
    }; \
    int caller_ ## name = (init_ ## name (jittor::get_from_env<type>(#name, default)), 0);

#define DEFINE_RUNTIME_FLAG_WITH_SETTER(type, name, default, doc) \
    DECLARE_RUNTIME_FLAG(type, name) \
    std::string doc_ ## name = doc; \
    void setter_ ## name (const type& old_value, const type& new_value); \
    void set_ ## name (const type& value) { \
        type& storage = runtime_flag_ ## name (); \
        type old_value = storage; \
        storage = value; \
        try { \
            setter_ ## name (old_value, value); \
        } catch (...) { \
            storage = old_value; \
            throw; \
        } \
    }; \
    void init_ ## name (const type& value) { \
        type& storage = runtime_flag_ ## name (); \
        type old_value = storage; \
        storage = value; \
        setter_ ## name (old_value, value); \
        if (getenv(#name)) LOGi << "Load " #name":" << value; \
    }; \
    int caller_ ## name = (init_ ## name (jittor::get_from_env<type>(#name, default)), 0);

#endif

DECLARE_FLAG(int, log_v);
DECLARE_FLAG(std::string, log_vprefix);
bool check_vlog(const char* fileline, int verbose);

#define V_ON(v) PREDICT_BRANCH_NOT_TAKEN(jittor::log_vprefix.size() ? \
        jittor::check_vlog(__FILELINE__, v) : \
        (v) <= jittor::log_v)

#define LOGV(v) \
    _LOG_IF(i, jittor::log_vprefix.size() ? \
        jittor::check_vlog(__FILELINE__, v) : \
        (v) <= jittor::log_v, v)

#define LOGv LOGV(1)
#define LOGvv LOGV(10)
#define LOGvvv LOGV(100)
#define LOGvvvv LOGV(1000)
#define CHECKop(a, op, b) LOG_IF(f, !((a) op (b))) \
    << "Check failed" \
    << #a "(" >> a >> ") " #op " " #b"(" >> b >> ")"

#define ASSERT(s) CHECK(s) << "Something wrong... Could you please report this issue?\n"
#define ASSERTop(a, op, b) CHECKop(a, op, b) << "Something wrong ... Could you please report this issue?\n"

#define LOGg LOGv >> jittor::green
#define LOGr LOGv >> jittor::red
#define LOGy LOGv >> jittor::yellow
#define LOGgg LOGvv >> jittor::green
#define LOGrr LOGvv >> jittor::red
#define LOGyy LOGvv >> jittor::yellow
#define LOGggg LOGvvv >> jittor::green
#define LOGrrr LOGvvv >> jittor::red
#define LOGyyy LOGvvv >> jittor::yellow
#define LOGgggg LOGvvvv >> jittor::green
#define LOGrrrr LOGvvvv >> jittor::red
#define LOGyyyy LOGvvvv >> jittor::yellow

#define LOGI jittor::LogVoidify() && jittor::Log(__FILELINE__, 'I', 0)
#define LOGir LOGI >> jittor::red
#define LOGig LOGI >> jittor::green
#define LOGiy LOGI >> jittor::yellow

void system_with_check(const char* cmd, const char* cwd=nullptr);

} // jittor
