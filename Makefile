## Makefile for local development tasks (servers, benchmarks, helpers)
##
## Description:
##   Convenience targets to start servers, run demos and manage models.
##
## How to use:
##   From the repository root, run for example:
##     make run-dashboard-amc-gan
##   Or list targets with:
##     make help
##
HOST_IP= 127.0.0.1
NS3_DIR=./../ns-allinone-3.44/ns-3.44

kill-ports:
	@echo "Killing processes on ports 5000 and 8000..."
	@lsof -i :5000 -t | xargs -r kill -9
	@lsof -i :5001 -t | xargs -r kill -9
	@lsof -i :8000 -t | xargs -r kill -9
	@lsof -i :9000 -t | xargs -r kill -9
	@lsof -i :9001 -t | xargs -r kill -9
	@echo "Ports 5000 and 8000 are now free."

list-ports:
	nmap -sT $(HOST_IP)

run-dashboard-amc-gan:
	@echo "Starting AMC Dashboard with AMC-GAN data source..."
	@python -m src.runners.run_dashboard_metrics

run-amc-int-server:
	@echo "Starting AMC Integrated Server..."
	python src/amc/integrated_amc_gan_enzo.py

run-sionna-server:
	@echo "Starting Sionna Server..."
	python src/servers/sionna_serverTorch.py

run-amc-gan:
	@echo "Starting AMC-GAN..."
	python src/servers/amc_gan_server.py

run-demo-nr:
	$(NS3_DIR)/ns3 run scratch/cttc-nr-demo-1.cc  -- --simTime=10.0 --packetSize=8192 

build-ns3:
	$(NS3_DIR)/ns3 build

download-models:
	@echo "Downloading pre-trained AMC-GAN models..."
	@rm -rf models/gan/*
	@mkdir -p models/gan
	@wget https://github.com/munoz213/amc_/raw/refs/heads/main/Project/enhanced_snr_gan.pth -O models/gan/enhanced_snr_gan.pth
	@wget https://github.com/munoz213/amc_/raw/refs/heads/main/Project/gan_training_data_enhanced.json -O models/gan/gan_training_data_enhanced.json
	@rm -rf models/rl/*
	@mkdir -p models/rl
# 	@wget https://github.com/munoz213/amc_/raw/refs/heads/main/Project/enhanced_snr_gan.pth -O models/rl/enhanced_snr_gan.pth
	@wget https://github.com/munoz213/amc_/raw/refs/heads/main/Project/rl_amc_model_safeguard1.pth -O models/rl/rl_amc_model_safeguard1.pth

# Docker convenience targets for the ns3 Dockerfile
NS3_DOCKERFILE=ns3/Dockerfile
NS3_DOCKER_IMAGE=ns3-twin-dev:latest
DOCKER_BUILD_CONTEXT=.

.PHONY: docker-ns3-build docker-ns3-run docker-ns3-shell docker-ns3-clean

docker-ns3-build:
	@echo "Building ns-3 Docker image from $(NS3_DOCKERFILE)..."
	docker build -f $(NS3_DOCKERFILE) -t $(NS3_DOCKER_IMAGE) $(DOCKER_BUILD_CONTEXT)


docker-ns3-run:
	@echo "Running ns-3 inside Docker (build + run demo)..."
	@echo "Note: Do NOT mount the local 'ns3' folder unless it contains a full ns-allinone tree; mounting it overwrites the image's ns-3 installation. Only the data volume is mounted by default."
	docker run --rm -it --network host --name ns3_run \
		-v $(PWD)/data:/root/amc_ws/data \
		$(NS3_DOCKER_IMAGE) /bin/bash -c "cd /root/amc_ws/ns-allinone-3.44/ns-3.44 && .&& ./ns3 run scratch/cttc-nr-demo"

ns3-deploy:
	@echo "Deploying ns-3"
	docker run -it --rm --network host --name ns3_container \
  -v $(PWD)/ns3/scratch/cttc-nr-demo.cc:/root/amc_ws/ns-allinone-3.44/ns-3.44/scratch/cttc-nr-demo.cc \
  $(NS3_DOCKER_IMAGE) /bin/bash -c "./ns3 build && ./ns3 run scratch/cttc-nr-demo"



docker-ns3-shell:
	@echo "Starting interactive shell in ns-3 Docker image..."
	@echo "Note: Avoid mounting local 'ns3' unless it contains full ns-allinone sources."
	docker run --rm -it --network host --name ns3_shell \
		-v $(PWD)/data:/root/amc_ws/data \
		$(NS3_DOCKER_IMAGE) /bin/bash

docker-ns3-clean:
	@echo "Removing ns-3 Docker image $(NS3_DOCKER_IMAGE) if present..."
	docker image rm -f $(NS3_DOCKER_IMAGE) || true

deploy-amc:
	@echo "Starting full simulation suite in Tmux..."
	bash src/runners/deploy_amc.sh