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

