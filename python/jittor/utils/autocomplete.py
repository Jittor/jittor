# usage: PYTHONPATH=/PATH/TO/JITTOR python autocomplete.py
import os
import re
from jittor import ops
jittor_path = os.environ.get("PYTHONPATH")

def get_pyi():
    # for __init__.py functions
    os.system(f"PYTHONPATH={jittor_path} stubgen -m jittor -o autoacompleteout")
    f = open(os.path.join("autoacompleteout","jittor","__init__.pyi"),"a")
    # for c++ ops
    var_text = "class Var:\n"
    for func_name,func in ops.__dict__.items():
        if func_name == "__doc__" or func_name == "__name__" or func_name == "__loader__" or func_name == "__spec__" or func_name == "__package__":
            continue
        text = func.__doc__
        declarations = re.findall(r"Declaration:\n(.+)\n",text)
        for decl in declarations:
            f.write(f"def {func_name}(")
            params = re.findall(r".+ [a-zA-Z_0-9]+\((.+)", decl)
            print(func_name,params)
            get_var = 0
            for param in params:
                para = param.split(",")
                for i,p in enumerate(para):
                    pa = p.strip().split(" ")[1]
                    if i == 0 and p.strip().split(" ")[0] == "VarHolder*":
                        get_var = 1
                        var_text += f"\tdef {func_name}(self,"
                    pf = pa.split("=")[0]
                    f.write(pf)
                    if get_var == 1:
                        var_text += pf
                    if i != len(para) - 1:
                        f.write(",")
                        if get_var == 1:
                            var_text += ","
                    else:
                        if len(pa.split("=")) > 1:
                            f.write("):...\n")
                            if get_var == 1:
                                var_text += "):...\n"
                        else:
                            f.write(":...\n")
                            if get_var == 1:
                                var_text += ":...\n"
    f.write(var_text)
    f.close()
    # mv to jittor folder and delete buffer
    abs_path = os.path.join(jittor_path,"jittor","__init__.pyi")
    pyi_path = os.path.join("autoacompleteout","jittor","__init__.pyi")
    os.system(f"mv {pyi_path} {abs_path}")
    os.system("rm -rf autoacompleteout")
    os.system("rm -rf .mypy_cache")

if __name__ == "__main__":
    get_pyi()