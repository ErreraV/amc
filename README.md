# Notes


### ns3 Build+Install
This was made to work properly with the pre-built python ns3 wrapper

#### venv creation

Creating a basic virtual environement for python 3.12
```
python3.12 -m venv  --system-site-packages ns346 

```
#### custom PATH creation

```
export VENVNS3=$HOME/venv/ns346

# Pour que le compilateur trouve les fichiers .hpp
export CPATH=$VENVNS3/include:$CPATH

# Pour que l'éditeur de liens trouve les libs au moment de la compilation
export LIBRARY_PATH=$VENVNS3/lib:$VENVNS3/lib64:$LIBRARY_PATH

# Pour que le système trouve les libs au moment de l'exécution
export LD_LIBRARY_PATH=$VENVNS3/lib:$VENVNS3/lib64:$LD_LIBRARY_PATH

export PIP_SITE=$VENVNS3/lib/python3.12/site-packages
```

#### installing JSON in PATH

Installing JSON.hpp as default lib
```

git clone https://github.com/nlohmann/json.git
cd json
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$VENVNS3
make -j$(nproc)
make install

```
#### custom configured and built ns3

```
./ns3 configure --build-profile=debug --enable-examples --enable-tests --disable-werror  \
--enable-python-bindings --enable-build-version --prefix=$VENVNS3/ \
-- -DNS3_BINDINGS_INSTALL_DIR="${PIP_SITE}"
```

TODO: Find-out why the installation doesnt work when using custom built python bidings

SUGESTIONS: Python bidings could make programming diferent ns3 scenarios easier

### ns-3.44:
This gives a hint that new demos all respect the UMi-Street Canyon scenario (need further investigation)
```
Changes from NR-v2.6 to v3.0

This release contains the upgrade of the supported ns-3 release, i.e., upgrade
from ns-3.40 to ns-3.41.

Changed behavior:
cttc-nr-demo example's parameters are updated to be according 3GPP TR 38.901 UMi-Street Canyon. Also,
the logged values in the NR tutorial are updated accordingly with the updated results.
```