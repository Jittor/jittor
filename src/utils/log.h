// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include <string>
#include <sstream>
#include <functional>
#include <iostream>
#include "types.h"

namespace jittor {

// define in tracer.cc
void print_trace();
    
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

#define PREDICT_BRANCH_NOT_TAKEN(x) (__builtin_expect(x, 0))

extern "C" uint32_t get_tid();
extern "C" bool g_supports_color;
extern "C" void print_prefix(std::ostream* out);

constexpr char green[] = "\033[38;5;2m";
constexpr char red[] = "\033[38;5;1m";
constexpr char yellow[] = "\033[38;5;3m";

static void get_color(char level, int verbose, const char*& color_begin, const char*& color_end) {
    if (level == 'i') {
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

extern "C" void send_log(std::ostringstream&& out);
extern "C" void flush_log();
extern "C" void log_capture_start();
extern "C" void log_capture_stop();
extern std::vector<std::map<string,string>> log_capture_read();
extern string thread_local thread_name;

struct Log {
    std::ostringstream out;
    const char* color_end;

    Log(const char* const fileline, char level, int verbose) {
        const char* color_begin;
        get_color(level, verbose, color_begin, color_end);
        if (g_supports_color) out << color_begin;
        out << '[' << level << ' ';
        print_prefix(&out);
        if (verbose) out << 'v' << verbose << ' ';
        out << fileline << ']';
    }

    void end() {
        if (g_supports_color) out << color_end;
        out << '\n';
        send_log(move(out));
    }
    void flush() { flush_log(); }

    template<class T>
    Log& operator<<(const T& a) { out << ' ' << a; return *this; }
    template<class T>
    Log& operator>>(const T& a) { out << a; return *this; }
};

struct LogVoidify {
    void operator&&(Log& log) { log.end(); }
};

struct LogFatalVoidify {
    void operator&&(Log& log) {
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

#define _LOG(level, v) _LOG ## level(v)
#define LOG(level) _LOG(level, 0)

#define CHECK(cond) \
    LOG_IF(f, PREDICT_BRANCH_NOT_TAKEN(!(cond))) \
        << "Check failed: " #cond " "

#define _LOG_IF(level, cond, v) \
    !(cond) ? (void) 0 : _LOG(level, v)
#define LOG_IF(level, cond) _LOG_IF(level, cond, 0)

template<class T> T get_from_env(const char* name,const T& _default) {
    auto s = getenv(name);
    if (s == NULL) return _default;
    std::istringstream is(s);
    T env;
    if (is >> env) {
        is.peek();
        if (!is) {
            return env;
        }
    }
    LOGw << "Load" << name << "from env(" << s << ") failed, use default" << _default;
    return _default;
}

template<> std::string get_from_env(const char* name, const std::string& _default);

#define DECLARE_FLAG(type, name) \
extern "C" type name; \
extern "C" std::string doc_ ## name; \
extern "C" void set_ ## name (const type&);

#ifdef JIT

#define DEFINE_FLAG(type, name, default, doc) \
    DECLARE_FLAG(type, name)
#define DEFINE_FLAG_WITH_SETTER(type, name, default, doc, setter) \
    DECLARE_FLAG(type, name)

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

#define DEFINE_FLAG_WITH_SETTER(type, name, default, doc) \
    DECLARE_FLAG(type, name) \
    type name; \
    std::string doc_ ## name = doc; \
    void setter_ ## name (type value); \
    void set_ ## name (const type& value) { \
        setter_ ## name (value); \
        name = value; \
    }; \
    void init_ ## name (const type& value) { \
        setter_ ## name (value); \
        name = value; \
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

#define LOGir LOGi >> jittor::red
#define LOGig LOGi >> jittor::green
#define LOGiy LOGi >> jittor::yellow

void system_with_check(const char* cmd);

} // jittor