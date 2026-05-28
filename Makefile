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

# Docker convenience targets for the ns3 Dockerfile
NS3_DOCKERFILE=ns3/Dockerfile
NS3_DOCKER_IMAGE=ns3-twin-dev:latest
DOCKER_BUILD_CONTEXT=.

.PHONY: kill-ports list-ports download-models ns3-build ns3-run ns3-shell ns3-clean deploy-amc deploy-amc-benchmarks run-dashboard-metrics run-sionna-torch run-sionna-tf run-benchmarks run-amc

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

ns3-build:
	@echo "Building ns-3 Docker image from $(NS3_DOCKERFILE)..."
	docker build -f $(NS3_DOCKERFILE) -t $(NS3_DOCKER_IMAGE) $(DOCKER_BUILD_CONTEXT)

ns3-run:
	@echo "Deploying ns-3"
	docker run -it --rm --network host --name ns3_container \
  -v $(PWD)/ns3/scratch/cttc-nr-demo.cc:/root/amc_ws/ns-allinone-3.44/ns-3.44/scratch/cttc-nr-demo.cc \
  $(NS3_DOCKER_IMAGE) /bin/bash -c "./ns3 build && ./ns3 run scratch/cttc-nr-demo"



ns3-shell:
	@echo "Starting interactive shell in ns-3 Docker image..."
	@echo "Note: Avoid mounting local 'ns3' unless it contains full ns-allinone sources."
	docker run --rm -it --network host --name ns3_shell \
		-v $(PWD)/data:/root/amc_ws/data \
		$(NS3_DOCKER_IMAGE) /bin/bash

ns3-clean:
	@echo "Removing ns-3 Docker image $(NS3_DOCKER_IMAGE) if present..."
	docker image rm -f $(NS3_DOCKER_IMAGE) || true

deploy-amc:
	@echo "Starting full simulation suite in Tmux..."
	bash src/runners/deploy_amc.sh $(ARGS)

deploy-amc-benchmarks:
	@echo "Starting AMC Benchmarks in Tmux..."
	bash src/runners/deploy_amc_benchmarks.sh $(ARGS)

run-dashboard-metrics:
	@echo "Starting AMC Dashboard with AMC-GAN data source..."
	@python -m src.runners.run_dashboard_metrics

run-sionna-torch:
	@echo "Starting Sionna Server..."
	python src/servers/sionna_server_torch.py

run-sionna-tf:
	@echo "Starting Sionna Server (TensorFlow version)..."
	python src/servers/sionna_server_tf.py

run-benchmarks:
	@echo "Running AMC Benchmarks..."
	python src/runners/run_benchmarks.py

run-amc:
	@echo "Starting AMC..."
	python src/amc/amc.py $(ARGS)