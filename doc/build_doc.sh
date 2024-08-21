# sudo python3.7 -m pip install \
#     recommonmark \
#     sphinx sphinx-autobuild sphinx_rtd_theme \
#     sphinx-autobuild \
#     --timeout 100


bpath=$(readlink -f "${BASH_SOURCE[0]}")
bpath=$(dirname "${bpath}")

jittor_path=$(readlink -f "${bpath}/..")

echo "[doc path] $bpath" 
echo "[jittor path] $jittor_path" 

export PYTHONPATH=$jittor_path/python
cd $bpath
sphinx-autobuild -b html source build -H 0.0.0.0 -p 8890
