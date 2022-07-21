import torch
from torchvision import transforms

DEFAULT_GPU=0
BATCH_SIZE=2
MAX_PT_NUM=8846

RESNET_MODEL_PATH='./checkpoints/resnet/'
POINTNET_MODEL_PATH='./checkpoints/pointnet/'
ACTION_MODEL_PATH='./checkpoints/'

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
device = torch.device('cuda:%d' % (DEFAULT_GPU) if torch.cuda.is_available() else 'cpu')
data_transform = transforms.Compose([transforms.ToTensor(),
    transforms.Resize([480, 640]),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
