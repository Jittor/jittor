import os

def fix_config(in_name, out_name, src_path, out_path):
    data = open(in_name, 'r', encoding='utf8').readlines()
    out = []
    for d in data:
        if d.startswith('INPUT                  ='):
            d = f'INPUT                  ={src_path}\n'
        elif d.startswith('OUTPUT_DIRECTORY       ='):
            d = f'OUTPUT_DIRECTORY       ={out_path}\n'
        out.append(d)
    f = open(out_name, 'w', encoding='utf8')
    f.writelines(out)

jt_path = os.getcwd()
cache_path = f"{os.environ['HOME']}/.cache/jittor"

os.system(f"rm -rf {cache_path}/docxygen/jittor")
os.system(f"mkdir -p {cache_path}/docxygen/jittor")
os.chdir(f"{cache_path}/docxygen")
# copy jittor src code
os.system(f"cp -r {jt_path}/src {cache_path}/docxygen/jittor")
os.system(f"cp -r {jt_path}/python {cache_path}/docxygen/jittor")
os.system(f"cp -r {jt_path}/notebook {cache_path}/docxygen/jittor")
os.system(f"cp {jt_path}/README.src.md {cache_path}/docxygen/jittor")
#download doxygen & config file
if not os.path.exists('doxygen-1.8.17'):
    os.system("wget -O doxygen.tar.gz https://cloud.tsinghua.edu.cn/f/dfa8f16ab00c4fa6b158/?dl=1")
    os.system("wget -O Doxyfile https://cloud.tsinghua.edu.cn/f/caf3c3aa518248d5ad73/?dl=1")
    os.system("tar -xzvf doxygen.tar.gz")
#run docxygen
fix_config(f'{cache_path}/docxygen/Doxyfile', f'{cache_path}/docxygen/doxygen-1.8.17/bin/Doxyfile', f'{cache_path}/docxygen/jittor', f'{cache_path}/docxygen')
os.chdir(f"{cache_path}/docxygen/doxygen-1.8.17/bin")
os.system(f'./doxygen Doxyfile')
