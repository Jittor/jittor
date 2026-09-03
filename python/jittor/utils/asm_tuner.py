#!/usr/bin/python3
# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers: 
#     Guowei Yang <471184555@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import sys
import os
os.environ["log_silent"] = "1"
import re
import jittor_utils as jit_utils
from jittor_utils import LOG
from jittor_utils.compiler_flags import remove_flags
jit_utils.try_import_jit_utils_core(silent=True)

def my_split(str):
    res=[]
    last=-1
    for i in range(len(str)):
        if str[i]==" " or str[i]=="\t":
            if last<i-1:
                res.append(str[last+1:i])
            last=i
        elif i==len(str)-1:
            res.append(str[last+1:])
    return res

def init(cc_path,s_path):
    global cc_content
    global s_content

    cc_content=[]
    s_content=[]
    with open(cc_path) as f:
        for line in f:
            cc_content.append(line)
    
    with open(s_path) as f:
        for line in f:
            s_content.append(line)

    global file_idx
    file_idx=0
    file_name=os.path.basename(cc_path)

    idx=0
    for line in s_content:
        idx=idx+1
        if idx<=2:
            continue
        if ".file" in line and file_name in line:
            ss = my_split(line)
            if len(ss) == 2:
                file_idx = 0
                break
            file_idx=int(my_split(line)[1])
            break

def deal_replace(start,end,s1,s2):
    global s_content
    global file_idx

    for i in range(len(s_content)):
        line=s_content[i]
        if ".loc\t" in line or ".loc " in line:
            args=my_split(line)[1:]
            if int(args[0])==file_idx and start<=int(args[1]) and int(args[1])<=end:
                for j in range(i+1,len(s_content)):
                    if ".loc" in s_content[j]:
                        break
                    tmp=s_content[j]
                    s_content[j]=re.sub(s1,s2,s_content[j])
                    if tmp!=s_content[j]:
                        tmp=tmp.replace("\n","")
                        tmp=s_content[j]
                        tmp=tmp.replace("\n","")

def pass_asm(cc_path,s_path):
    global cc_content
    global s_content

    for i in range(len(cc_content)):
        line=cc_content[i]
        if "@begin" in line:
            cmds=line.split("@begin")[1].split(" ")
            si=0
            while cmds[si]=="":
                si=si+1
            
            start=i+1
            end=i+1
            for j in range(i+1,len(cc_content)):
                if "@end" in cc_content[j]:
                    end=j-1
                    break
            if cmds[si]=="replace":
                args=[]
                idx=0
                while line.find('"',idx)!=-1:
                    idx=line.find('"',idx)+1
                    args.append(line[idx:line.find('"',idx)])
                    idx=line.find('"',idx)+1
                deal_replace(start+1,end+1,args[0],args[1])
            else:
                assert 0, "no such command: "+line.split("@begin")[1]
    
    output_path=s_path.replace(".post.s",".s")
    # Temporary file plus rename, not open(path, "w").
    #
    # Truncating the destination and writing into it means that between those
    # two moments the file on disk is a *prefix* of the assembly. Several
    # processes compile into one cache directory as a matter of course (a
    # DataLoader with num_workers=4 is four of them), so a reader lands in
    # that window and the assembler reports
    #
    #   Assembler messages: unknown pseudo-op: '.by'
    #   end of file not at end of a line
    #
    # on an operator that has nothing to do with anything anyone changed --
    # which reads as a corrupted cache, and gets dismissed as one. rename() is
    # atomic within a filesystem, so a reader sees either the old file or the
    # complete new one. Reproduced on three independent JITTOR_HOMEs before
    # this change; the .tmp name carries the pid so two writers do not share
    # one temporary either.
    temporary = "%s.tmp.%d" % (output_path, os.getpid())
    try:
        with open(temporary, "w") as f:
            f.write("".join(s_content))
        os.replace(temporary, output_path)
    except BaseException:
        try:
            os.remove(temporary)
        except OSError:
            pass
        raise

def run_cmd(cmd):
    LOG.vvvv(f"Run cmd: {cmd}")
    assert os.system(cmd) == 0, f"Run cmd failed: {cmd}"

args=sys.argv
args[1]=args[1][args[1].find("=")+1:]
compiler_path = sys.argv[1]
cc_path = sys.argv[2]
output_path = args[args.index("-o") + 1]

for i in range(len(args)):
    if args[i].find(" ")!=-1:
        args[i]="'"+args[i]+"'"
        
cmd = " ".join(args[1:])
cc_pos=cmd.find("_op.cc")
so_pos=cmd.find("_op.so")

def replace_output(cmd, output):
    pos = cmd.rfind(output_path)
    assert pos != -1, "output path not found in command: " + output_path
    return cmd[:pos] + output + cmd[pos+len(output_path):]

# remove -Xclang ...
remove_clang_flag = lambda s: re.sub("-Xclang (('[^']*')|([^ ]*))", "", s)

if cc_pos==-1:  #s_to_so
    run_cmd(remove_clang_flag(cmd))
elif so_pos==-1:  #cc_to_s
    asm_cmd=replace_output(cmd, cc_path.replace("_op.cc", "_op.post.s")) \
        .replace(" -o ", " -g -o ")
    run_cmd(asm_cmd)

    s_path = cc_path.replace("_op.cc","_op.post.s")
    init(cc_path,s_path)
    pass_asm(cc_path,s_path)
else:  #cc_to_so
    asm_cmd=replace_output(cmd, cc_path.replace("_op.cc", "_op.post.s")) \
        .replace("-lstdc++", "") \
        .replace("-ldl", "") \
        .replace("-shared", "-S") \
        .replace(" -o ", " -g -o ")
    asm_cmd = remove_flags(asm_cmd, ['-l', '-L', '-Wl,', '.lib', '-shared'])
    run_cmd(asm_cmd)

    s_path = cc_path.replace("_op.cc","_op.post.s")
    init(cc_path,s_path)
    pass_asm(cc_path,s_path)

    asm_cmd = cmd.replace("_op.cc", "_op.s") \
        .replace(" -g", "")
    run_cmd(remove_clang_flag(asm_cmd))
