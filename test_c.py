import os
import re

def get_pyi():
    f = open(os.path.join(".","python","jittor","__init__.pyi"),"w")
    # fundamental declaration
    f.write("from typing import List, Tuple, Optional, Union, Any, ContextManager, Callable, overload\n")
    f.write("import builtins\nimport math\nimport pickle\n")
    # for c++ ops
    for func_name,func in ops.__dict__.items():
        if func_name == "__doc__" or func_name == "__name__" or func_name == "__loader__" or func_name == "__spec__" or func_name == "__package__":
            continue
        # print(func_name)
        text = func.__doc__
        declarations = re.findall(r"Declaration:\n(.+)\n",text)
        # print(declarations)
        for decl in declarations:
            f.write(f"def {func_name}(")
            params = re.findall(r".+ [a-zA-Z_0-9]+\((.+)", decl)
            # print(params)
            for param in params:
                para = param.split(",")
                for i,p in enumerate(para):
                    pa = p.strip().split(" ")[1]
                    pf = pa.split("=")[0]
                    # print(pa)
                    f.write(pf)
                    if i != len(para) - 1:
                        f.write(",")
                    else:
                        if len(pa.split("=")) > 1:
                            f.write("):...\n")
                        else:
                            f.write(":...\n")
    f.close()

if __name__ == "__main__":
    get_pyi()