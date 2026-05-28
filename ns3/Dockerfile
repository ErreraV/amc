FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /root/amc_ws

# Instala Python 3.12 e dependências de build
RUN apt-get update && apt-get install -y \
    cmake ninja-build g++ python3-dev python3-pip python3-venv \
    git wget bzip2 tcpdump sqlite3 libsqlite3-dev libxml2-dev \
    libgsl-dev qtbase5-dev gir1.2-goocanvas-2.0 python3-gi python3-gi-cairo \
    graphviz libgraphviz-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*
# Criação do VENV conforme suas notas
RUN python3 -m venv /opt/venv/ns3
ENV PATH="/opt/venv/ns3/bin:$PATH"
ENV VENVNS3="/opt/venv/ns3"

# Instalação dos pacotes Python solicitados

RUN INCLUDE_PATH=/usr/include/graphviz \
    LIBRARY_PATH=/usr/lib/graphviz \
    pip install --no-cache-dir cppyy pygraphviz

# Instalação do JSON.hpp no prefixo do VENV
RUN git clone https://github.com/nlohmann/json.git && \
    cd json && mkdir build && cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=$VENVNS3 && \
    make -j$(nproc) install

# Download e extração do ns-3.44
RUN wget https://www.nsnam.org/releases/ns-allinone-3.44.tar.bz2 && \
    tar -xvjf ns-allinone-3.44.tar.bz2

# Configuração do Módulo NR (5G-LENA)
WORKDIR /root/amc_ws/ns-allinone-3.44/ns-3.44/contrib
RUN git clone https://gitlab.com/cttc-lena/nr.git -b 5g-lena-v4.0.y

# Build final do ns-3 com bindings para o site-packages do VENV
WORKDIR /root/amc_ws/ns-allinone-3.44/ns-3.44
RUN CXXFLAGS="-I/opt/venv/ns3/include" ./ns3 configure --build-profile=debug --enable-examples --enable-tests \
    --prefix=$VENVNS3 --enable-python-bindings --disable-werror\
    -- -DNS3_BINDINGS_INSTALL_DIR="/opt/venv/ns3/lib/python3.12/site-packages"

RUN ./ns3 build -j 8 && ./ns3 install
