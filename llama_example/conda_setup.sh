# create a new environment named "llama"
conda create -n llama python=3.10 -y

# activate the environment
conda activate llama

# install pytorch with cuda 12.1 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# install huggingface libraries
pip install transformers accelerate datasets huggingface_hub

# optional: bitsandbytes for quantization and memory savings
pip install bitsandbytes

# optional: rich printing
pip install rich

# login via a huggingface token (get one at https://huggingface.co/settings/tokens)
huggingface-cli login

# download the model
git lfs install
git clone https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
