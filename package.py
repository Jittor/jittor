import os
import jittor as jt
import jittor_utils as jit_utils
package_path = "package"
python = "python"

f = open("tmp.txt", "w")
f.close()
cache_path = os.path.join(jit_utils.home(), ".cache", "jittor")
print("===========delete cache===========")
os.system(f"rm -rf {cache_path}")
print("===========start build===========")
os.system(f"GET_PACKAGE=1 {python} -m jittor.test.test_example")
print("===========make package===========")
datas = open("tmp.txt", "r").readlines()
os.system(f"rm tmp.txt")

if os.path.exists(package_path):
    os.system(f"rm -rf {package_path}")
for i in range(len(datas) // 4):
    url = datas[i * 4 + 0].rstrip()
    filename = datas[i * 4 + 1].rstrip()
    root_folder = datas[i * 4 + 2].rstrip()
    md5 = datas[i * 4 + 3].rstrip()

    target_path = os.path.join(package_path, root_folder[root_folder.find(".cache/jittor"):])
    os.makedirs(target_path, exist_ok=True)
    os.system(f"cp {os.path.join(root_folder, filename)} {os.path.join(target_path, filename)}")
print("===========download jittor===========")
jittor_path = os.path.join(package_path, "jittor")
os.makedirs(jittor_path)
os.system(f"{python} -m pip download jittor -d {jittor_path}")
print("===========write setup script===========")
f = open(os.path.join(package_path, "setup.py"), "w")
code = """
import os
os.system("rm -rf ~/.cache/jittor")
os.makedirs("~/.cache", exist_ok=True)
os.system("cp -r ./.cache/jittor ~/.cache/jittor")
temp = [i for i in os.listdir("jittor") if i.startswith("jittor-")]
assert(len(temp) == 1)
jittor_file = temp[0]
os.system(f"cd jittor && python -m pip install {jittor_file}")
os.system(f"python -m jittor.test.test_example")
"""
f.write(code)
f.close()
print("===========done===========")
temp = [i for i in os.listdir(os.path.join(package_path, "jittor")) if i.startswith("jittor-")]
jittor_version = temp[0][7:-7]
os.system(f"zip -r jittor-offline-{jittor_version}.zip {package_path}")