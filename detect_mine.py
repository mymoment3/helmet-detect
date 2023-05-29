import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import *
from utils.datasets import *
from utils.utils import *

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 获取模型
model = torch.load('helme.pt', map_location=device)['model'].float() 
model.to(device).eval()

# 读取yaml配置
with open('helme.yaml') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)  

# 测试图片
img = 'data/images/your_image.jpg' 
img = torch.from_numpy(img).to(device)
img = img.float() 
img /= 255.0  
if img.ndimension() == 3: 
    img = img.unsqueeze(0)

# 推理 
pred = model(img)[0]

# 处理预测结果
pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms) 