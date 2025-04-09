### Set up the python environment

```
conda create -n mmfd python=3.8
conda activate mmfd

# make sure that the pytorch cuda is consistent with the system cuda
# e.g., if your system cuda is 11.3, install torch 1.10.0 built from cuda 11.3
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch

# install dependencies
pip install -r requirements.txt
```
