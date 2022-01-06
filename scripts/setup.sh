pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r ../hf_requirements.txt
pip install -r ../requirements.txt
git clone https://github.com/NVIDIA/apex
#Comment apex installation below if fp16 isn't required
cd apex
python setup.py install --cuda_ext --cpp_ext
cd ..

apt-get update
apt-get install wget
apt-get install unzip

# transformers installation from source
# git clone https://github.com/huggingface/transformers
# cd transformers
# git checkout 21da895013a95e60df645b7d6b95f4a38f604759
# pip install .
# pip install -r examples/requirements.txt
# cd ..
