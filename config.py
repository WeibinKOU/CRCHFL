import torch
from torchvision import transforms

DEFAULT_GPU=0
BATCH_SIZE=16
THREED_CHANNEL=8
MAX_PT_NUM=8846

RESNET_MODEL_PATH='./checkpoints/resnet/'
POINTNET_MODEL_PATH='./checkpoints/pointnet/'
ACTION_MODEL_PATH='./checkpoints/'

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
device = torch.device('cuda:%d' % (DEFAULT_GPU) if torch.cuda.is_available() else 'cpu')
data_transform = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize(mean=[0.51726955, 0.539306, 0.5525119],
        std=[0.19789377, 0.19814016, 0.21107046])
    ])

