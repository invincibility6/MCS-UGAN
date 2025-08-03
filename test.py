import os
import time
import argparse
import numpy as np
from PIL import Image
from glob import glob
from ntpath import basename
from os.path import join, exists
# pytorch libs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
import torchvision.transforms as transforms
from ptflops import get_model_complexity_info

## options
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="./Dataset/test/testA/")
parser.add_argument("--sample_dir", type=str, default="./output")
parser.add_argument("--model_name", type=str, default="MCS_UGAN")
parser.add_argument("--weights_path", type=str, default="checkpoints/UIEB/generator_300.pth")
opt = parser.parse_args()

## checks
assert exists(opt.weights_path), "model not found"
os.makedirs(opt.sample_dir, exist_ok=True)
is_cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if is_cuda else torch.FloatTensor 

# print(opt.model_name)
## model arch
if opt.model_name=='MCS_UGAN':
    from models import MCS_UGAN
    model = MCS_UGAN.test6_Generator()

## load weights
model.load_state_dict(torch.load(opt.weights_path))
if is_cuda: model.cuda()
model.eval()
print ("Loaded model from %s" % (opt.weights_path))

## data pipeline
img_width, img_height, channels = 256, 256, 3
transforms_ = [transforms.Resize((img_height, img_width), Image.BICUBIC),
               transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]
transform = transforms.Compose(transforms_)

input_res = (3, 256, 256)
macs, params = get_model_complexity_info(model, input_res, as_strings=True, print_per_layer_stat=True)
print(f"模型 FLOPs: {macs}")
print(f"模型参数量: {params}")


## testing loop
times = []
test_files = sorted(glob(join(opt.data_dir, "*.*")))
t1 = time.time()
for path in test_files:
    inp_img = transform(Image.open(path))
    inp_img = Variable(inp_img).type(Tensor).unsqueeze(0)
    inp_img = torch.cat((inp_img, inp_img), 0)      # 删
    s = time.time()
    gen_img = (model(inp_img))
    times.append(time.time()-s)
    img_sample = torch.cat((inp_img.data, gen_img.data), -1)
    # save_image(gen_img, join(opt.sample_dir, basename(path)), normalize=True)
    # save_image(img_sample, join(opt.sample_dir, basename(path)), normalize=True)
    gen_img = torch.split(gen_img, 1, 0)    # 删
    gen_img = gen_img[0].squeeze()          # 删
    save_image(gen_img, join(opt.sample_dir, basename(path)), normalize=True)
    print("Tested: %s" % path)
t2 = time.time()
print("Time: %sms" % ((t2 - t1)*1000))

## run-time    
if (len(times) > 1):
    print ("\nTotal samples: %d" % len(test_files)) 
    # accumulate frame processing times (without bootstrap)
    Ttime, Mtime = np.sum(times[1:]), np.mean(times[1:]) 
    print ("Time taken: %d sec at %0.3f fps" %(Ttime, 1./Mtime))
    print("Saved generated images in in %s\n" %(opt.sample_dir))

# python test.py --weights_path checkpoints/UIEB/generator_300.pth

