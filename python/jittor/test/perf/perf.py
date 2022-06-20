import sys, os

suffix = ""

import jittor as jt
import time
import jittor_utils as jit_utils
home_path = jit_utils.home()
perf_path = os.path.join(home_path, ".cache", "jittor_perf")

def main():
    os.makedirs(perf_path+"/src/jittor", exist_ok=True)
    os.makedirs(perf_path+"/src/jittor_utils", exist_ok=True)
    os.system(f"cp -rL {jt.flags.jittor_path} {perf_path+'/src/'}")
    os.system(f"cp -rL {jt.flags.jittor_path}/../jittor_utils {perf_path+'/src/'}")
    use_torch_1_4 = os.environ.get("use_torch_1_4", "0") == "1"
    dockerfile_src = r"""
FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

RUN echo \
"deb [trusted=yes] https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic main restricted universe multiverse\n\
deb [trusted=yes] https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-updates main restricted universe multiverse\n\
deb [trusted=yes] https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-backports main restricted universe multiverse\n\
deb [trusted=yes] https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-security main restricted universe multiverse" > /etc/apt/sources.list

# RUN rm -rf /var/lib/apt/lists/*
RUN apt update || true

RUN apt install wget \
        python3.7 python3.7-dev \
        g++ build-essential -y

WORKDIR /usr/src

RUN apt download python3-distutils && dpkg-deb -x ./python3-distutils* / \
    && wget -O - https://bootstrap.pypa.io/get-pip.py | python3.7

# change tsinghua mirror
RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip3 install  
        numpy \
        tqdm \
        pillow \
        astunparse

RUN pip3 install torch torchvision
"""
    global suffix
    if use_torch_1_4:
        suffix = "_1_4"
        dockerfile_src = dockerfile_src.replace("torch ", "torch==1.4.0 ")
        dockerfile_src = dockerfile_src.replace("torchvision", "torchvision==0.5.0")
    with open("/tmp/perf_dockerfile", 'w') as f:
        f.write(dockerfile_src)
    assert os.system("sudo nvidia-smi -lgc 1500") == 0

    # if the docker image is not built 
    if os.system(f"sudo docker image inspect jittor/jittor-perf{suffix}"):
        assert os.system(f"sudo docker build --tag jittor/jittor-perf{suffix} -f /tmp/perf_dockerfile .") == 0

    # run once for compile source
    jt_fps = test_main("jittor", "resnet50", 1)
    
    logs = ""
    # resnext50_32x4d with bs=8 cannot pass this test
    #### inference test
    for model_name in ["resnet50", "wide_resnet50_2", # "resnext50_32x4d", 
        "resnet152", "wide_resnet101_2", "resnext101_32x8d", 
        "alexnet", "vgg11", "squeezenet1_1", "mobilenet_v2", 
        "densenet121", "densenet169", "densenet201",
        "res2net50", "res2net101"]:
        for bs in [1, 2, 4, 8, 16, 32, 64, 128]:
            jt_fps = test_main("jittor", model_name, bs)
            logs += f"jittor-{model_name}-{bs} {jt_fps}\n"
            tc_fps = test_main("torch", model_name, bs)
            logs += f"torch-{model_name}-{bs} {tc_fps}\n"
            logs += f"compare-{model_name}-{bs} {jt_fps/tc_fps}\n"
            print(logs)
    #### train test
    for model_name in ["train_resnet50", "train_resnet101"
        ]:
        for bs in [1, 2, 4, 8, 16, 32, 64, 128]:
            jt_fps = test_main("jittor", model_name, bs)
            logs += f"jittor-{model_name}-{bs} {jt_fps}\n"
            tc_fps = test_main("torch", model_name, bs)
            logs += f"torch-{model_name}-{bs} {tc_fps}\n"
            logs += f"compare-{model_name}-{bs} {jt_fps/tc_fps}\n"
            print(logs)
    with open(f"{perf_path}/jittor-perf{suffix}-latest.txt", "w") as f:
        f.write(logs)
    from datetime import datetime
    with open(f"{perf_path}/jittor-perf{suffix}-{datetime.now()}.txt", "w") as f:
        f.write(logs)

def test_main(name, model_name, bs):
    cmd = f"sudo docker run --gpus all --rm -v {perf_path}:/root/.cache/jittor --network host jittor/jittor-perf{suffix} bash -c 'PYTHONPATH=/root/.cache/jittor/src python3.7 /root/.cache/jittor/src/jittor/test/perf/perf.py {name} {model_name} {bs}'"
    fps = -1
    try:
        print("run cmd:", cmd)
        if os.system(cmd) == 0:
            with open(f"{perf_path}/{name}-{model_name}-{bs}.txt", 'r') as f:
                fps = float(f.read().split()[3])
    except:
        pass
    return fps

def time_iter(duration=2, min_iter=5):
    start = time.time()
    for i in range(10000000):
        yield i
        end = time.time()
        if end-start>duration and i>=min_iter:
            return

def test(name, model_name, bs):
    print("hello", name, model_name, bs)
    import numpy as np
    import time
    is_train = False
    _model_name = model_name
    if model_name.startswith("train_"):
        is_train = True
        model_name = model_name[6:]
    if name == "torch":
        import torch
        import torchvision.models as tcmodels
        from torch import optim
        from torch import nn
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        model = tcmodels.__dict__[model_name]()
        model = model.cuda()
    else:
        import jittor as jt
        from jittor import optim
        from jittor import nn
        jt.flags.use_cuda = 1
        jt.cudnn.set_algorithm_cache_size(10000)
        import jittor.models as jtmodels
        model = jtmodels.__dict__[model_name]()
        if (model == "resnet152" or model == "resnet101") and bs == 128 and is_train:
            jt.cudnn.set_max_workspace_ratio(0.05)
    if is_train:
        model.train()
    else:
        model.eval()
    img_size = 224
    if model_name == "inception_v3":
        img_size = 300
    test_img = np.random.random((bs, 3, img_size, img_size)).astype("float32")
    if is_train:
        label = (np.random.random((bs,)) * 1000).astype("int32")
    if name == "torch":
        test_img = torch.Tensor(test_img).cuda()
        if is_train:
            label = torch.LongTensor(label).cuda()
            opt = optim.SGD(model.parameters(), 0.001)
        sync = lambda: torch.cuda.synchronize()
        jt = torch
    else:
        test_img = jt.array(test_img).stop_grad()
        if is_train:
            label = jt.array(label).stop_grad()
            opt = optim.SGD(model.parameters(), 0.001)
        sync = lambda: jt.sync_all(True)

    sync()
    use_profiler = os.environ.get("use_profiler", "0") == "1"
    if hasattr(jt, "nograd"):
        ng = jt.no_grad()
        ng.__enter__()
    def iter():
        x = model(test_img)
        if isinstance(x, tuple):
            x = x[0]
        if is_train:
            loss = nn.CrossEntropyLoss()(x, label)
            if name == "jittor":
                opt.step(loss)
            else:
                opt.zero_grad()
                loss.backward()
                opt.step()
        else:
            if name == "jittor":
                x.sync()
    sync()
    for i in time_iter():
        iter()
    sync()
    for i in time_iter():
        iter()
    sync()
    if use_profiler:
        if name == "torch":
            prof = torch.autograd.profiler.profile(use_cuda=True)
        else:
            prof = jt.profile_scope()
        prof.__enter__()
    if name == "jittor":
        if hasattr(jt.flags, "use_parallel_op_compiler"):
            jt.flags.use_parallel_op_compiler = 0
    start = time.time()
    for i in time_iter(10):
        iter()
    sync()
    end = time.time()
    if use_profiler:
        prof.__exit__(None,None,None)
        if name == "torch":
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
    total_iter = i+1
    print("duration:", end-start, "FPS:", total_iter*bs/(end-start))
    fpath = f"{home_path}/.cache/jittor/{name}-{_model_name}-{bs}.txt"
    with open(fpath, 'w') as f:
        f.write(f"duration: {end-start} FPS: {total_iter*bs/(end-start)}")
    os.chmod(fpath, 0x666)

if len(sys.argv) <= 1:
    main()
else:
    name, model, bs = sys.argv[1:]
    bs = int(bs)
    test(name, model, bs)