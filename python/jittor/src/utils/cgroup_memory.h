// Copyright (c) 2026 Jittor. All Rights Reserved.
// This file is subject to the terms in LICENSE.txt.
#pragma once

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace jittor {
namespace cgroup_memory {

inline std::string read(const std::string& path) {
    std::ifstream file(path);
    std::ostringstream contents;
    contents << file.rdbuf();
    auto value = contents.str();
    auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    return value.substr(first, value.find_last_not_of(" \t\r\n")-first+1);
}

inline std::vector<std::string> split(const std::string& text, char separator) {
    std::istringstream input(text);
    std::vector<std::string> parts;
    std::string part;
    while (std::getline(input, part, separator)) parts.push_back(part);
    return parts;
}

inline bool contains(const std::string& list, const std::string& item) {
    auto parts = split(list, ',');
    return std::find(parts.begin(), parts.end(), item) != parts.end();
}

inline std::string unescape_mount(const std::string& text) {
    std::string result;
    for (size_t i=0; i<text.size(); ++i) {
        if (text[i] == '\\' && i+3<text.size() &&
            text[i+1]>='0' && text[i+1]<='7' &&
            text[i+2]>='0' && text[i+2]<='7' &&
            text[i+3]>='0' && text[i+3]<='7') {
            result += char((text[i+1]-'0')*64 + (text[i+2]-'0')*8 + text[i+3]-'0');
            i += 3;
        } else result += text[i];
    }
    return result;
}

inline std::string trim_path(std::string path) {
    while (path.size()>1 && path.back()=='/') path.pop_back();
    return path;
}

// Clamp only the compilation budget, not MemInfo or the runtime allocator.
// Keep this algorithm in sync with jittor_utils/cgroup.py.
inline uint64_t limit_memory(uint64_t memory,
    const char* cgroup_file="/proc/self/cgroup",
    const char* mountinfo_file="/proc/self/mountinfo") {
    std::map<std::string, std::string> groups;
    std::ifstream memberships(cgroup_file);
    std::string line;
    while (std::getline(memberships, line)) {
        auto first = line.find(':');
        auto second = first == std::string::npos ? first : line.find(':', first+1);
        if (second == std::string::npos) continue;
        auto controllers = line.substr(first+1, second-first-1);
        auto path = trim_path(line.substr(second+1));
        if (path.empty() || path[0]!='/') continue;
        if (contains(controllers, "memory")) groups["cgroup"] = path;
        else if (line.substr(0, first)=="0" && controllers.empty()) groups["cgroup2"] = path;
    }
    std::ifstream mounts(mountinfo_file);
    while (std::getline(mounts, line)) {
        auto separator = line.find(" - ");
        if (separator == std::string::npos) continue;
        std::istringstream before(line.substr(0, separator)), after(line.substr(separator+3));
        std::string id, parent, device, root, mount, options, kind, source, super_options;
        if (!(before >> id >> parent >> device >> root >> mount >> options) ||
            !(after >> kind >> source >> super_options)) continue;
        auto group = groups.find(kind);
        if (group == groups.end()) continue;
        if (kind=="cgroup" && !contains(super_options, "memory")) continue;
        root = trim_path(unescape_mount(root));
        mount = trim_path(unescape_mount(mount));
        auto& path = group->second;
        std::string relative;
        // A namespace-root membership does not match unrelated subtree mounts.
        if (path==root) relative = "";
        else if (root=="/") relative = path.substr(1);
        else if (path.compare(0, root.size()+1, root+"/")==0) relative = path.substr(root.size()+1);
        else continue;
        bool outside = false;
        for (auto& part : split(relative, '/'))
            if (part=="." || part=="..") outside = true;
        if (mount.empty() || mount[0]!='/' || outside) continue;
        std::string directory = relative.empty() ? mount : trim_path(mount)+"/"+relative;
        if (mount=="/" && !relative.empty()) directory = "/"+relative;
        bool leaf = true;
        while (true) {
            if (kind=="cgroup2" || leaf || read(directory+"/memory.use_hierarchy")=="1") {
                auto name = kind=="cgroup2" ? "/memory.max" : "/memory.limit_in_bytes";
                auto value = read(directory+name);
                if (!value.empty() && value.find_first_not_of("0123456789")==std::string::npos) {
                    try {
                        memory = std::min<uint64_t>(memory, std::stoull(value));
                    } catch (const std::exception&) {
                        // Missing, malformed and unlimited limits leave the host bound intact.
                    }
                }
            }
            if (directory==mount) break;
            auto slash = directory.find_last_of('/');
            directory = slash==0 ? "/" : directory.substr(0, slash);
            leaf = false;
        }
    }
    return memory;
}

} // namespace cgroup_memory
} // namespace jittor
