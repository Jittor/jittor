# ***************************************************************
# Copyright (c) 2022 Jittor. All Rights Reserved. 
# Maintainers:
#     Guoye Yang <498731903@qq.com>
#     Dun Liang <randonlang@gmail.com>.
#
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************

'''
example:

export class_home=/mnt/disk/cjld/class_nn
mkdir -p $class_home
docker pull jittor/jittor-cuda
python3.7 -m jittor_utils.class.setup_env setup 4
python3.7 -m jittor_utils.class.setup_env start 4
python3.7 -m jittor_utils.class.setup_env report
python3.7 -m jittor_utils.class.setup_env restart 4
python3.7 -m jittor_utils.class.setup_env stop
'''
# export class_home
# setup [n]          // setup for n users. including build user paths, user_info.txt and docker imgs. !!!WILL RESET SUDENT_FILES!!!
# start [n_gpu]      // run n docker CONTAINERs with n_gpu GPUs.
# stop               // stop n docker CONTAINERs
# restart [n_gpu]    // restart n docker CONTAINERs with n_gpu GPUs.
import sys
import os 
import json as js
import random

class_home = os.environ["class_home"]
student_files_dir = class_home + "/student_files"
student_files_bk_dir = class_home + "/student_files_bak"
cwd = os.path.dirname(__file__)

def run_cmd(cmd):
    print("[CMD]:", cmd)
    ret = os.system(cmd)
    if ret:
        print("[CMD] return", ret)
    return ret

def generate_random_str(randomlength):
  random_str = ''
  base_str = 'ABCDEFGHIGKLMNOPQRSTUVWXYZabcdefghigklmnopqrstuvwxyz0123456789'
  length = len(base_str) - 1
  for i in range(randomlength):
    random_str += base_str[random.randint(0, length)]
  return random_str

def setup(n):
    if os.path.exists(student_files_dir):
        if os.path.exists(student_files_bk_dir):
            run_cmd(f"rm -rf {student_files_bk_dir}")
        run_cmd(f"mv {student_files_dir} {student_files_bk_dir}")
    os.makedirs(student_files_dir)
    user_info = []
    for i in range(n): # 0 for root
        port = 20000 + i
        passwd = generate_random_str(8)
        name = 'stu_'+str(i)
        path = os.path.abspath(os.path.join(student_files_dir, name))
        info = {'port': port,
                'passwd': passwd,
                'name': name,
                'path': path}
        user_info.append(info)
        student_files_src = class_home + "/student_files_src"
        if os.path.isdir(student_files_src):
            run_cmd(f"cp -r {student_files_src} {path}")
        else:
            run_cmd('mkdir -p ' + path)
    js.dump(user_info, open(student_files_dir+"/user_info.json", "w"))

def start(n, n_gpu):
    assert os.path.exists(student_files_dir+'/user_info.json')
    user_info = js.load(open(student_files_dir+'/user_info.json', 'r'))
    for i in range(len(user_info)):
        id = i % n
        ids = ''
        for j in range(n_gpu):
            if j > 0:
                ids+=','
            ids += str((i * n_gpu + j) % n)
        u = user_info[i]
        print('START', i, '/', len(user_info))
        assert 0 == run_cmd(f'docker run -itd --shm-size=8g --network host --name {u["name"]} -v {u["path"]}:/root --gpus \'"device={ids}"\' jittor/jittor-cuda bash')
        # assert 0 == run_cmd(f'docker exec -it {u["name"]} bash -c \'apt update && apt install openssh-server -y\'')
        assert 0 == run_cmd(f'docker cp {cwd}/setup.py {u["name"]}:/etc/ssh/setup.py')
        assert 0 == run_cmd(f'docker cp {cwd}/motd {u["name"]}:/etc/motd')
        assert 0 == run_cmd(f'docker exec -it {u["name"]} python3.7 /etc/ssh/setup.py passwd {u["passwd"]}')
        assert 0 == run_cmd(f'docker exec -it {u["name"]} python3.7 /etc/ssh/setup.py ssh {u["port"]}')
        assert 0 == run_cmd(f'docker exec -it {u["name"]} python3.7 -m pip install jittor -U')
        assert 0 == run_cmd(f'docker exec -it {u["name"]} python3.7 -m jittor.test.test_example')

def stop():
    assert os.path.exists(student_files_dir+'/user_info.json')
    user_info = js.load(open(student_files_dir+'/user_info.json', 'r'))
    for i in range(len(user_info)):
        u = user_info[i]
        print('STOP', i, '/', len(user_info))
        run_cmd(f'docker rm -f {u["name"]}')

def report():
    assert os.path.exists(student_files_dir+'/user_info.json')
    user_info = js.load(open(student_files_dir+'/user_info.json', 'r'))
    hostname = open("/etc/hostname", 'r').read().strip() + ".randonl.me"
    for i in range(len(user_info)):
        u = user_info[i]
        print(f"ssh -p {u['port']} root@{hostname} # passwd: {u['passwd']}")

def restart(n, n_gpu):
    stop()
    start(n, n_gpu)

args = sys.argv[1:]
if (args[0] == 'setup'):
    assert(len(args) == 2)
    assert(type(eval(args[1])) == int)
    n = int(args[1])
    assert(n < 999)
    setup(n)
elif (args[0] == 'start'):
    assert(len(args) in [2,3])
    assert(type(eval(args[1])) == int)
    n = int(args[1])
    if len(args) == 3:
        assert(type(eval(args[2])) == int)
        n_gpu = int(args[2])
    else:
        n_gpu=1
    start(n, n_gpu)
elif (args[0] == 'stop'):
    stop()
elif (args[0] == 'restart'):
    assert(len(args) in [2,3])
    assert(type(eval(args[1])) == int)
    n = int(args[1])
    if len(args) == 3:
        assert(type(eval(args[2])) == int)
        n_gpu = int(args[2])
    else:
        n_gpu=1
    restart(n, n_gpu)
elif (args[0] == 'report'):
    report()
else:
    assert(False)

