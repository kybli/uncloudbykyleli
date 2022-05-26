import argparse
import torch
from torchvision import transforms

import opt
from data_loader import data_loader
from evaluation import evaluate
from net import PConvUNet
from util.io import load_ckpt

import os

parser = argparse.ArgumentParser()
# training options
parser.add_argument('--root', type=str, default='./data')
parser.add_argument('--snapshot', type=str, default='')
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--metric-file', type=str, default=None)
parser.add_argument('--save-dir', type=str, default=None)
parser.add_argument('--specific-dataset', type=str, default=None)
parser.add_argument('--abs-data-cap', type=int, default=50)
parser.add_argument('--save-img', action='store_true')
parser.add_argument('--use-edge-metrics', action='store_true')
args = parser.parse_args()

device = torch.device('cuda')

size = (args.image_size, args.image_size)
img_transform = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor(),
     transforms.Normalize(mean=opt.MEAN, std=opt.STD)])
mask_transform = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor()])

if args.specific_dataset:
    dataset_val = data_loader(args.root, img_transform, mask_transform, 'val', include=[args.specific_dataset], one_val_per_dataset=False, abs_dataset_cap=args.abs_data_cap)
else:
    dataset_val = data_loader(args.root, img_transform, mask_transform, 'val', one_val_per_dataset=True)

model = PConvUNet().to(device)
load_ckpt(args.snapshot, [('model', model)])

pth_num = args.snapshot.split("/")[-1].split(".")[-2]

model.eval()
evaluate(model, dataset_val, device, os.path.join(args.save_dir, 'result.jpg'), limit_img_count_to_eight=args.specific_dataset is None, get_metrics=True, use_edge_metrics=args.use_edge_metrics, save_dir=args.save_dir, save_img=args.save_img, save_edge_img=args.use_edge_metrics, metric_file = args.metric_file, pth_num=pth_num)
