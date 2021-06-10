# Specify cuda root path installed in your environment
# CUDA_PATH := /usr/local/cuda
# CUDA_VERSION := $(shell $(CUDA_PATH)/bin/nvcc --version | tail -n1 | awk '{print substr($$5,0,length($$5)-1)}')

# ifneq ($(strip $(KALDI)),)

all: miniconda3/envs/mini_asr/bin activate_python.sh env.sh 

export PYTHONNOUSERSITE=True
miniconda3.sh:
	wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $@

miniconda3: miniconda3.sh
	# -b: non-interactive install
	# -p: installed directory
	bash miniconda3.sh -b -p miniconda3

# virtual environment of python
miniconda3/envs/mini_asr/bin: miniconda3
	miniconda3/bin/conda update -y conda
	miniconda3/bin/conda env create -f environment.yml
update:
	miniconda3/bin/conda env update -f environment.yml

activate_python.sh:
	./setup_conda.sh

env.sh: miniconda3/envs/mini_asr/bin
	cp env.sh.in env.sh
	echo "export LD_LIBRARY_PATH=$(CUDA_PATH)/lib64:$$LD_LIBRARY_PATH" >> env.sh