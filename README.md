# How to deploy:
```
#This will call the Sionna + AMC+ Dashboard+ NS3+ metric_server
make deploy-amc-benchmarks 
```

```
#On tmux: set-up the mouse:
(ESC then Ctrl+B then : (two points))
set -g mouse on
```

## Change amc config:
To change AMC config in the dashboard, you can:
- Kill the amc.py (top right pane)
- run the following command

```
#To change the AMC config:
python src/amc/amc.py [OPTIONS]
```
### Possible options:
- to disable gan: "--no-gan"
- To use table: "--method table"
- To use RL (default) : "--method rl"
- Call for help :"--help"
ex:
```
#Calling the amc without the gan and with the table:
python src/amc/amc.py --no-gan --method table
```




# General setup

1. Create a local environment file from the template:
	`cp example.env .env`

After copying, open `.env` and set the path to your virtual environment directory so scripts can find your venv (for example, `VENV=.venv` or `VENVNS3=$HOME/venv/ns3`).
 
Prerequisites: this project expects `tmux` and `docker` to be installed on your system.
On Debian/Ubuntu you can install them with:

```bash
sudo apt update
sudo apt install -y tmux docker.io
```

On Fedora/CentOS derivatives use:

```bash
sudo dnf install -y tmux docker
sudo systemctl enable --now docker
```

## Additional prerequisites

- Python: this project targets **Python 3.12**. Create and activate a venv:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

- Python dependencies: install from `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

- System build tools (required to build ns-3 and native extensions). Example for Debian/Ubuntu:

```bash
sudo apt update
sudo apt install -y build-essential cmake pkg-config git wget lsof nmap python3.12-venv python3.12-dev
```

- Docker permissions: Docker commands may require `sudo` or adding your user to the `docker` group:

```bash
sudo usermod -aG docker $USER
# then log out and back in for the group change to take effect
```

- CUDA / GPU (optional): If you plan to run models on GPU, ensure matching NVIDIA drivers and CUDA toolkit are installed and install a compatible PyTorch build.

- Optional helpers: `python-dotenv` is optional but the project will auto-discover a `.env` file if available. For ns-3 dev the README already recommends `cppyy` and `pygraphviz`.

2. Create and activate a Python 3.12 virtual environment for local development.
3. Install the Python dependencies from `requirements.txt`.
4. Review the available local tasks with `make help`.
5. Start the services you need with the matching `make` target, such as `make run-amc` or `make run-dashboard-metrics`.

# Notes

### SSH commmand to run the amc on the server and benchmarks locally

```bash
ssh -R 5001:localhost:5001 username@server
```
### ns3 Build+Install
This was made to work properly with the pre-built python ns3 wrapper

#### create a workspace

```
cd
mkdir amc_ws
cd amc_ws
```
#### venv creation

Creating a basic virtual environement for python 3.12
```
cd ~/amc_ws
mkdir venv
python3.12 -m venv  --system-site-packages venv/ns3 

##recommended
pip install cppyy pygraphviz
pip install ns3
pip install sionna-no-rt
### if a package breaks others, we recomend doing:
pip install <package-name> --force-reinstall
###or even maybe:
pip install <package-name> --force-reinstall --no-cache
###example (after we broke tensorflow version):
pip install sionna-no-rt --force-reinstall --no-cache 
```
#### custom PATH creation

```
###inside .profile or .bash_profile
export VENVNS3=$HOME/venv/ns3

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
cd ~/amc_ws
git clone https://github.com/nlohmann/json.git
cd json
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$VENVNS3
make -j$(nproc)
make install
```
#### custom configured and built ns3

```
cd ~/amc_ws
git clone https://github.com/kohler/click
cd click/
./configure --disable-linuxmodule --enable-nsclick --enable-wifi --prefix=$VENVNS3
make -j
make install
```
```
cd ~/amc_ws
wget https://www.nsnam.org/releases/ns-allinone-3.44.tar.bz2

tar -xvjf ns-allinone-3.44.tar.bz2
cd ns-allinone-3.44

#in .profile or .bash_profile
export NS3SRC=$HOME/ns-allinone-3.44/ns-3.44/
```
```
cd $NS3SRC/contrib
#you should check right nr version compatible with ns3 on https://cttc-lena.gitlab.io/nr/html/index.html#autotoc_md153

git cl one https://gitlab.com/cttc-lena/nr.git -b  5g-lena-v4.0.y
```
```
cd $NS3SRC

./ns3 configure --build-profile=debug --enable-examples --enable-tests --disable-werror \
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
