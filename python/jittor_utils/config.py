import os
import sys
import jittor_utils
from jittor_utils import LOG


def search_file(dirs, name):
    for d in dirs:
        fname = os.path.join(d, name)
        if os.path.isfile(fname):
            return fname
    LOG.f(f"file {name} not found in {dirs}")

if __name__ == "__main__":
    help_msg = f"Usage: {sys.executable} -m jittor_utils.config --include-flags|--link-flags|--cxx-flags|--cxx-example|--help"
    if len(sys.argv) <= 1:
        print(help_msg)
        sys.exit(1)

    s = ""
    # base should be something like python3.7m python3.8
    base = jittor_utils.run_cmd(jittor_utils.py3_config_path + " --includes").split()[0]
    base = "python3" + base.split("python3")[-1]
    for arg in sys.argv[1:]:
        if arg == "--include-flags":
            s += jittor_utils.run_cmd(jittor_utils.py3_config_path + " --includes")
            s += " -I"+os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "jittor", "src"))
            s += " "
        elif arg == "--libs-flags":
            libbase = "/usr/lib/x86_64-linux-gnu"
            libpath = libbase + f"/lib{base}.so"
            assert os.path.isfile(libpath), f"lib not exist: {libpath}"
            s += f" -L{libbase} -l{base} -ldl "
        elif arg == "--cxx-flags":
            s += " --std=c++17 "
        elif arg == "--cxx-example":
            cc_src = '''
// please compile with: g++ a.cc $(python3 -m jittor_utils.config --include-flags --libs-flags --cxx-flags) -o a.out && ./a.out
#include <pyjt/pyjt_console.h>
#include <iostream>

using namespace std;

int main() {
    jittor::Console console;
    // run python code in console
    console.run("print('hello jt console', flush=True)");

    // set a python value: a = 1
    console.set<int>("a", 1);
    // get a python value
    cout << console.get<int>("a") << endl;

    // set a python string
    console.set<string>("b", "hello");
    cout << console.get<string>("b") << endl;

    // set a python array
    vector<int> x{1,2,3,4};
    console.set("x", x);
    auto x2 = console.get<std::vector<int>>("x");
    for (auto a : x2) cout << a << " "; cout << endl;

    // set and get a jittor array
    jittor::array<int, 2> arr2({2,3}, {6,5,4,3,2,1});
    arr2(0,0) = -1;
    console.set_array("arr2", arr2);
    console.run("print(arr2, flush=True); arr3 = arr2**2;");
    auto arr3 = console.get_array<int, 2>("arr3");
    cout << arr3.shape[0] << ' ' << arr3.shape[1] << endl;
    for (int i=0; i<arr3.shape[0]; i++) {
        for (int j=0; j<arr3.shape[1]; j++)
            cout << arr3(i,j) << ' ';
        cout << endl;
    }

    // run resnet18
    jittor::array<float, 4> input({2, 3, 224, 224});
    memset(input.data.get(), 0, input.nbyte());
    console.set_array("input", input);
    console.run(R"(
import jittor as jt
from jittor.models import resnet

model = resnet.resnet18()
pred = model(input)
    )");
    auto pred = console.get_array<float, 2>("pred");
    cout << "pred.shape " << pred.shape[0] << ' ' << pred.shape[1] << endl;

    return 0;
}
            '''
            print(cc_src)
        elif arg == "--help":
            print(help_msg)
            sys.exit(0)
        else:
            print(help_msg)
            sys.exit(1)
    print(s)
