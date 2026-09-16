# 安装、依赖与复现边界

运行过的环境：Linux + NVIDIA GPU，Python3.8.20，torch2.4.0+cu121，torchvision0.19.0，gym0.21.0，mujoco-py2.1.2.14，MuJoCo2.1。完整实际包版本见 `provenance/environment-packages.json`，它是环境审计，不是可无条件一次安装的lockfile。

## 新环境建议流程

```bash
conda create -n rl100 python=3.8 -y
conda activate rl100
python -m pip install 'setuptools==59.5.0' wheel
python -m pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
bash scripts/bootstrap_source.sh
python -m pip install zarr==2.12.0 hydra-core==1.2.0 omegaconf==2.3.0 dill==0.3.5.1 \
  numpy==1.23.5 numba==0.56.4 diffusers==0.33.1 einops==0.8.1 \
  wandb==0.20.1 open3d==0.19.0 opencv-python==4.11.0.86 \
  scipy==1.10.1 scikit-learn==1.3.2 scikit-image==0.21.0 \
  moviepy==1.0.3 imageio==2.35.1 av==12.3.0 matplotlib==3.7.5 \
  termcolor==2.4.0 tqdm==4.67.1 dm_control==1.0.23 \
  trimesh==4.6.12 h5py==3.11.0 ipdb==0.13.13 gdown==5.2.0
```

NumPy1.23.5与numba0.56.4对应历史环境。以上列出核心依赖，不是已在干净容器完整验证的lockfile；安装后检查依赖冲突。

MuJoCo二进制从其官方release安装到 `$HOME/.mujoco/mujoco210`，需要系统OpenGL/EGL、GLEW、patchelf、编译器等。不要关闭TLS证书验证。详细上游流程见固定提交的 `vendor/RL-100/INSTALL.md`，其真实机器人可选依赖不是本任务必需。

```bash
python -m pip install -e vendor/RL-100/third_party/gym-0.21.0
python -m pip install -e vendor/RL-100/third_party/mujoco-py-2.1.2.14
python -m pip install -e vendor/RL-100/third_party/Metaworld
python -m pip install -e vendor/RL-100/third_party/pytorch3d_simplified
python -m pip install -e vendor/RL-100/third_party/dexart-release
python -m pip install -e vendor/RL-100/third_party/rrl-dependencies/mj_envs
python -m pip install -e vendor/RL-100/third_party/rrl-dependencies/mjrl
python -m pip install -e vendor/RL-100/visualizer
```

MetaWorld旧版必须可导入 `SawyerPegInsertionSideV2Policy`，并保持原相机XML。历史环境的外部MetaWorld安装记录了commit `7c5df9a5d3111e5fa8d2fe814c4fdcb3acfa3f26`，但没有完成该外部工作树所有修改与vendor树的逐文件等价证明。上游资产bootstrap不是完整环境位级快照。

R3M是独立editable安装，上游gitlink为 `b2334e726887fa0206962d7984c69c5fb09cceab`。上游 `.gitmodules` 未完整声明它；需从 [facebookresearch/r3m](https://github.com/facebookresearch/r3m) 安装并取得官方ResNet18权重。权重不放入本仓库，首次加载可能下载。

`rl100/`本身不是pip package，通过run.sh的PYTHONPATH引入。部分环境模块仍会导入Adroit/DexArt，所以只装MetaWorld未必足够。没有接入真实机器人所需的设备或权限。

## 检查

```bash
GPU_ID=0 bash scripts/run.sh -c 'import torch,hydra,metaworld,mujoco_py,r3m; print(torch.__version__, torch.cuda.is_available())'
GPU_ID=0 bash scripts/run.sh tests/test_semantics.py
GPU_ID=0 bash scripts/run.sh experiments/dual_rgb/collect.py --shard 9 --smoke --dest /tmp/rl100-collect-smoke
```

不能用smoke的1条轨迹代替正式100条；正式采集目录已有数据时会拒绝覆盖。

若切换容器、驱动、GPU可见性，确认CUDA与EGL指向同一物理卡。新环境尚未完成所有长训练复现；失败时先保留完整异常、配置和依赖位置，避免把安装错误当算法失败。
