<div align="center">

<h1>AirTouch: A Low-Cost Versatile Visuotactile Feedback System for Enhanced Robotic Teleoperation</h1>

</div>

<div>

## Conda Environment Setup
```bash
# Create a conda environment
conda create -n airtouch python=3.9
# Install PyTorch
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
# Install other dependencies
git clone https://github.com/huangyan28/AirTouch.git
cd KeypointFusion
pip install -r ./requirements.txt
```
## Train Dataset

```bash
python train_rgbd.py
```

<div>
