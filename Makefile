SHELL:=/bin/bash
PIP=$(shell which pip3 || echo "pip3")
GPUS ?= "--gpus all"


build:
	@docker build -t birds_eye .
run_mcts: build run_mcts_nobuild
run_mcts_nobuild:
	@echo
	@echo "Running evaulation script with MCTS model"
	@docker run -it $(GPUS) birds_eye -c /BirdsEye/configs/mcts.yaml
	@echo
run_dqn: build run_dqn_nobuild
run_dqn_nobuild:
	@echo
	@echo "Running evaluation script with DQN model"
	@docker run -it $(GPUS) birds_eye -c /BirdsEye/configs/dqn.yaml
	@echo
run_batch: build run_batch_nobuild
run_batch_nobuild:
	@echo
	@echo "Running batch evaluation script"
	@docker run -it $(GPUS) birds_eye -b
	@echo
run_sigscan: build run_sigscan_nobuild
run_sigscan_nobuild:
	@echo
	@echo "Running sigscan script"
	@docker run -it --net=host $(GPUS) birds_eye sigscan.py
