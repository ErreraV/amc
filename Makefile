HOST_IP= 127.0.0.1



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
		@python3 src/runners/run_amc_with_analytics.py

run-amc-int-server:
	@echo "Starting AMC Integrated Server..."
	python -m src.amc.integrated_amc_gan

run-sionna-server:
	@echo "Starting Sionna Server..."
	python src/servers/sionna_serverTorch.py

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