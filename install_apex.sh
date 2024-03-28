#########################################################################
# File Name: install_apex.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Sun Mar 24 05:08:40 2024
#########################################################################
#!/bin/bash

#git clone https://github.com/NVIDIA/apex.git
cd apex
#git checkout b496d85fb88a801d8e680872a12822de310951fd
pip install -v --no-build-isolation --disable-pip-version-check --no-cache-dir --config-settings "--build-option=--cpp_ext --cuda_ext --fast_layer_norm --distributed_adam --deprecated_fused_adam" ./
