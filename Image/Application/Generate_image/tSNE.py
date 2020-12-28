from dataloader import get_loader
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import random
import argparse
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='mCEVAE', help='mCEVAE')
parser.add_argument('--int', type=str, default='M', help='intervention variable) M: mustache; S: smiling')

parser.add_argument('--seed', type=int, default=3, help='random seed')
parser.add_argument('--gpu', type=int, default=0, help='number of gpu')
parser.add_argument('--batch_size', type=int, default=50, help='number of gpu')

args = parser.parse_args()

args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

# seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(args.seed)

# Image Folder (Create)
src_path = os.path.dirname(os.path.realpath('__file__'))
src_path = os.path.abspath(os.path.join(src_path, os.pardir))
data_df = os.path.join(src_path, 'data', 'celebA', 'images')

# local
src_path2 = os.path.abspath(os.path.join(src_path, os.pardir))
data_df = os.path.join(src_path2, 'data', 'celebA', 'images')

if args.model == 'mCEVAE':
    hyper = 'u_dim_10_1.0_40.0_10.0_10.0'
elif args.model == 'DCEVAE':
    hyper = 'ud_ur_dim_5_5_1.00_40.00_1.00_3.20'
elif args.model == 'CEVAE':
    hyper = 'u_dim_10_1.0_40.0_1.0'
elif args.model == 'CVAE':
    hyper = 'u_dim_10_1.0_40.0_1.0'

new_path = os.path.join(src_path, 'tSNE')
if not os.path.exists(new_path):
    os.mkdir(new_path)

new_path = os.path.join(new_path, args.model)
if not os.path.exists(new_path):
    os.mkdir(new_path)

new_path = os.path.join(new_path, hyper)
if not os.path.exists(new_path):
    os.mkdir(new_path)

save_path = os.path.join(new_path, str(args.int)+str(args.seed))
if not os.path.exists(save_path):
    os.mkdir(save_path)

model_path = os.path.join(src_path, args.model, 'result_'+str(args.int),'model_'+str(args.seed)+'.pth')
attr_path = os.path.join(src_path, 'data', 'celebA', 'list_attr_celeba.txt')

if args.int == 'M':
    whole = ['Young', 'Male', 'Eyeglasses', 'Bald', 'Mustache', 'Smiling', \
             'Wearing_Lipstick', 'Mouth_Slightly_Open', 'Narrow_Eyes']
    sens = ['Mustache']
    des = []
elif args.int == 'S':
    whole = ['Young', 'Male', 'Eyeglasses', 'Bald', 'Mustache', 'Mustache', 'Wearing_Lipstick']
    sens = ['Smiling']
    des = ['Mouth_Slightly_Open', 'Narrow_Eyes']

test_loader = get_loader(data_df, attr_path, whole, sens, des, mode='test')

# Model Loading
test_model = torch.load(model_path, map_location=args.device)
test_model.to(args.device)
test_model.eval()

def save(file_name, img, folder):
    npimg = np.transpose(img.numpy(), (1, 2, 0))
    f = os.path.join(folder, "%s.jpg" % file_name)
    plt.imsave(f, npimg)

def draw_tSNE(input, a, save_path, latent_name):
    from sklearn.manifold import TSNE
    import time
    t0 = time.time()
    print('tSNE start for ' + latent_name)

    input = input.cpu().detach().numpy()
    a = a.cpu().detach().numpy()

    tsne = TSNE(n_components=2, verbose=1, perplexity=10, n_iter=300)
    tsne_result = tsne.fit_transform(input)

    colors = 'orange', 'b'  # , 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
    for c, label in zip(colors, [0.0, 1.0]):
        name = 'a=1' if label == 1.0 else 'a=0'
        index = np.where(a == label)
        print(index)
        plt.scatter(tsne_result[index, 0], tsne_result[index, 1], c=c, marker='.', label=name, alpha=1)
    plt.legend()
    figfile = os.path.join(save_path, 'tSNE_' + latent_name + '_wrt_A')
    plt.show()
    plt.savefig(figfile)

    plt.close()

    print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - t0))

itr = 0
for cur_data, cur_sens, cur_rest, cur_des in tqdm(test_loader):
    cur_data, cur_sens, cur_rest, cur_des = \
        cur_data.to(args.device), cur_sens.to(args.device), cur_rest.to(args.device), cur_des.to(args.device)
    if args.model == 'mCEVAE' or args.model == 'CEVAE':
        u_mu, u_logvar = test_model.q_u(cur_data, cur_sens, cur_rest, cur_des)
    elif args.model == 'DCEVAE' or args.model == 'CVAE':
        u_mu, u_logvar = test_model.q_u(cur_data, cur_sens)

    if itr == 0:
        u_mu_all = u_mu
        u_logvar_all = u_logvar
        a_all = cur_sens
    else:
        u_mu_all = torch.cat([u_mu_all, u_mu], 0)
        u_logvar_all = torch.cat([u_logvar_all, u_logvar], 0)
        a_all = torch.cat([a_all, cur_sens], 0)
    itr += 1
    if itr == 500:
        break

u = test_model.reparameterize(u_mu_all, u_logvar_all)

if args.model == 'DCEVAE':
    ur, ud = torch.split(u, [5, 5], 1)
    draw_tSNE(ur, a_all, save_path, 'Ur')
    draw_tSNE(ud, a_all, save_path, 'Ud')
else:
    draw_tSNE(u, a_all, save_path, 'U')