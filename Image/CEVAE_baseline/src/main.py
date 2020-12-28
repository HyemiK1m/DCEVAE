import matplotlib
matplotlib.use('Agg')
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--gpu', type=int, default=0, help='number of gpu')
parser.add_argument('--batch_size', type=int, default=64, help='number of gpu')

parser.add_argument('--u_dim', type=int, default=10, help='dimension of u')
parser.add_argument('--beta1', type=float, default=1, help='beta1')
parser.add_argument('--beta2', type=float, default=40, help='beta2')
parser.add_argument('--beta3', type=float, default=1, help='beta3')

parser.add_argument('--int', type=str, default='M', help='intervention variable) M: mustache; S: smiling')

parser.add_argument('--max_epochs', type=int, default=400, help='max epochs')
parser.add_argument('--save_per_epoch', type=int, default=10, help='save per epoch')
parser.add_argument('--early_stop', type=int, default=10, help='early stop')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--rep', type=int, default=1, help='rep')

parser.add_argument('--retrain', type=bool, default=False, help='retrain or not')

parser.add_argument('--tv', type=bool, default=True, help='if torch version is not 1.4.0, turn False')

args = parser.parse_args()

import os
import torch
if args.tv == False:
    entrypoints = torch.hub.list('pytorch/vision:v0.5.0', force_reload=True)
import logging.handlers

from dataloader import get_loader
from CEVAE import CEVAE
from train import train
from test import test
import random
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(args.seed)

print('-----------------------------------')
print(args.device)
print('-----------------------------------')

src_path = os.path.dirname(os.path.realpath('__file__'))
src_path = os.path.abspath(os.path.join(src_path, os.pardir))
src_path = os.path.abspath(os.path.join(src_path, os.pardir))
data_df = os.path.join(src_path, 'data', 'celebA', 'images')
attr_path = os.path.join(src_path, 'data', 'celebA','list_attr_celeba.txt')

if args.int == 'M':
    whole = ['Young', 'Male', 'Eyeglasses', 'Bald', 'Mustache', 'Smiling', \
             'Wearing_Lipstick', 'Mouth_Slightly_Open', 'Narrow_Eyes']
    sens = ['Mustache']
    des = []
elif args.int == 'S':
    whole = ['Young', 'Male', 'Eyeglasses', 'Bald', 'Mustache', 'Mustache', 'Wearing_Lipstick']
    sens = ['Smiling']
    des = ['Mouth_Slightly_Open', 'Narrow_Eyes']

train_loader = get_loader(data_df, attr_path, whole, sens, des, mode='train')
test_loader = get_loader(data_df, attr_path, whole, sens, des, mode='test')
valid_loader = get_loader(data_df, attr_path, whole, sens, des, mode='valid')

# Dimension
args.fixed_batch = next(iter(train_loader))
img, sens, rest_att, des_att = args.fixed_batch
sens_dim = sens.shape[1]
rest_dim = rest_att.shape[1]
des_dim = des_att.shape[1]

args.valid_batch = next(iter(valid_loader))

model = CEVAE(args, sens_dim=sens_dim, rest_dim=rest_dim, des_dim=des_dim, u_dim=args.u_dim).to(args.device)
model = model.train()

src_path = os.path.dirname(os.path.realpath('__file__'))
src_path = os.path.abspath(os.path.join(src_path, os.pardir))
result_path = os.path.join(src_path, 'result')

if not os.path.exists(result_path):
    os.mkdir(result_path)

args.save_path = os.path.join(result_path, "u_dim_{:d}_{:.1f}_{:.1f}_{:.1f}"\
                              .format(args.u_dim, args.beta1, args.beta2, args.beta3))
if not os.path.exists(args.save_path):
    os.mkdir(args.save_path)
args.save_path = os.path.join(args.save_path, args.int+str(args.seed))
if not os.path.exists(args.save_path):
    os.mkdir(args.save_path)

log_file = os.path.join(args.save_path, "intervention_log")
logger = logging.getLogger("mylogger")
logger.setLevel(level=logging.INFO)
streamHandler = logging.StreamHandler()
fileHandler = logging.FileHandler(log_file)
logger.addHandler(streamHandler)
logger.addHandler(fileHandler)
args.logger = logger

check = os.path.join(args.save_path, 'model.pth')
if os.path.exists(check) and args.retrain == False:
    print('reload saved model')
else:
    train(args, model, train_loader, valid_loader)
test(args, test_loader)



