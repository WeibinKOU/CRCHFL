import torch
from torchvision import transforms

DEFAULT_GPU=0
BATCH_SIZE=32
MAX_PT_NUM=8846

RESNET_MODEL_PATH='./checkpoints/resnet/'
POINTNET_MODEL_PATH='./checkpoints/pointnet/'
ACTION_MODEL_PATH='./checkpoints/'

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
device = torch.device('cuda:%d' % (DEFAULT_GPU) if torch.cuda.is_available() else 'cpu')
data_transform = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize(mean=[0.49017215, 0.5100299, 0.5238774],
        std=[0.20810874, 0.209677, 0.22034127])
    ])

