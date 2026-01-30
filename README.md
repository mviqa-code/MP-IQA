
# MP-IQA

## Configuration
- Python 3.9.0
- CUDA 11.7
- Intel Core i9-13900K CPU
- NVIDIA GeForce RTX 3090 GPU

## Requirement
```bash
pip install -r requirements.txt
```

## Download
Please download [RUOD](https://drive.google.com/file/d/1sqOglwWml1tqnGEfB2xpP7lvUbv9WIyE/view?usp=drive_link) dataset and model [weight](https://drive.google.com/file/d/141mgiQKWl_0ZyeiY3plOQw-mZ9hIyVu2/view?usp=drive_link).
Directory structure:
```
MP-IQA-main
├── checkpoints
│   ├── mpiqa_ruod.pth
├── data
│   ├── RUOD
│   │   ├── images
│   │   │   ├── 0.000001.jpg
│   │   │   ├── 0.000002.jpg
│   │   │   ├── ......
│   │   │   ├── 0.014000.jpg
│   │   ├── metas
│   │   │   ├── metas.json
│   │   │   ├── test_metas.json
│   │   │   ├── train_metas.json
├── log
├── model
├── build_load.py
├── config.py
├── extract_score.py
├── requirements.txt
```

## Test Demo
Terminal run:
```bash
python extract_score.py
```

## Launch TensorBoard
Terminal run:
```bash
tensorboard --logdir log/mpiqa_ruod_tensorboard --port 6006
```
