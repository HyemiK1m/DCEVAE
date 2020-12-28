import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
import os
import torchvision.utils as vutils
import torch

def show(file_name, img, dpi=300):
    npimg = np.transpose(img.numpy(), (1, 2, 0))
    plt.figure(dpi=dpi)
    plt.title(file_name, fontsize=14)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.imshow(npimg)
    plt.show()

def show_subplot(file_name, img):
    npimg = np.transpose(img.numpy(), (1, 2, 0))
    plt.title(file_name, fontsize=25)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.imshow(npimg)

def show_and_save(file_name, img, dpi=300):
    npimg = np.transpose(img.numpy(), (1, 2, 0))
    f = "./%s.png" % file_name
    fig = plt.figure(dpi=dpi)
    plt.title(file_name, fontsize=14)
    plt.xticks([], [])
    plt.yticks([], [])
    # plt.imshow(npimg)
    plt.imsave(f, npimg)

def coordinate(fixed_sens):
    fixed_sens = fixed_sens.squeeze(1).cpu().numpy()
    cdn = []
    for idx, sens in enumerate(fixed_sens):
        if sens == 1:
            row = int(idx / 8) + 1
            col = int(idx % 8) + 1
            cdn.append([row, col])
    return cdn

def plot_results(model, args, epoch, whole_losses, x_recon_losses, l_recon_losses, u_kl_losses, cf_recon_losses):

    plt.figure(figsize=(18, 5))

    plt.subplot(231)
    plt.title("whole loss")
    plt.plot(whole_losses, label="loss")
    plt.legend(loc="lower right")

    plt.subplot(232)
    plt.title("MSE(X)")
    plt.plot(x_recon_losses, label="MSE(X)")
    plt.legend(loc="lower right")

    plt.subplot(233)
    plt.title("MSE(L)")
    plt.plot(l_recon_losses, label="label_loss")
    plt.legend(loc="lower right")

    plt.subplot(234)
    plt.title("kl_u")
    plt.plot(u_kl_losses, label="u_kl")
    plt.legend(loc="lower right")

    plt.subplot(235)
    plt.title("cf recon")
    plt.plot(cf_recon_losses, label="MSE(X_cf)")
    plt.legend(loc="lower right")

    plt.savefig(os.path.join(args.save_path, 'losses_epoch_%d.png' % epoch))

    plt.figure(figsize=(20, 10), dpi=200)
    fixed_data, fixed_sens, _, _ = args.fixed_batch

    plt.subplot(121)
    #show_subplot('Real_epoch_%d' % epoch, make_grid((fixed_data[:16].data * 0.5 + 0.5).cpu(), 4))

    # here we show recovered imgs
    plt.subplot(122)
    rec_imgs = model.reconstruct_x(fixed_data.to(args.device), fixed_sens.to(args.device))
    vutils.save_image(rec_imgs.cpu().data * 0.5 + 0.5,
                      os.path.join(args.save_path, 'rec_img_epoch_%d.png' % epoch),
                      normalize=True)
    #show_subplot('Rec_epoch_%d' % epoch, make_grid((rec_imgs[:16].data * 0.5 + 0.5).cpu(), 4))
    #plt.show()

def test_results(args, model, fixed_data, fixed_sens, test=True, epoch=0):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fixed_data, fixed_sens = fixed_data.to(device), fixed_sens.to(device)
    rec_imgs = model.reconstruct_x(fixed_data, fixed_sens)
    int_imgs = model.sampling_intervention(fixed_sens)
    cond_imgs, a_cond = model.sampling_conditional(fixed_data)
    cf_imgs = model.sampling_counterfactual(fixed_data, fixed_sens)

    if test == True:
        png_real = 'test_real_img.png'
        png_rec = 'test_rec_img.png'
        png_int = 'test_int_img.png'
        png_cond = 'test_int_img.png'
        png_cf = 'test_cf_img.png'
        png_a0 = 'test_a0_img.png'
        png_a1 = 'test_a1_img.png'

        a0 = torch.zeros_like(fixed_sens).to(device)
        a1 = torch.ones_like(fixed_sens).to(device)
        a0_imgs = model.sampling_intervention(a0)
        a1_imgs = model.sampling_intervention(a1)

        cdn = coordinate(fixed_sens)
        args.logger.info('test intervention a: {:s}'.format(str(cdn)))
        cdn = coordinate(a_cond)
        args.logger.info('test cond a: {:s}'.format(str(cdn)))
    else:
        png_real = 'valid_real_img.png'
        png_rec = 'valid_rec_img_epoch_{:d}.png'.format(epoch)
        png_int = 'valid_int_img_epoch_{:d}.png'.format(epoch)
        png_cond = 'valid_cond_img_epoch_{:d}.png'.format(epoch)
        png_cf = 'valid_cf_img_epoch_{:d}.png'.format(epoch)
        png_a0 = 'valid_a0_img_epoch_{:d}.png'.format(epoch)
        png_a1 = 'valid_a1_img_epoch_{:d}.png'.format(epoch)

        a0 = torch.zeros_like(fixed_sens).to(device)
        a1 = torch.ones_like(fixed_sens).to(device)
        a0_imgs = model.sampling_intervention(a0)
        a1_imgs = model.sampling_intervention(a1)

        if epoch == 0:
            cdn = coordinate(fixed_sens)
            args.logger.info('valid epoch {:d}: {:s}'.format(epoch, str(cdn)))
        cdn = coordinate(a_cond)
        args.logger.info('valid cond a: {:s}'.format(str(cdn)))

    plt.figure(figsize=(18, 5))

    plt.subplot(121)
    #show_subplot('Real', make_grid((fixed_data[:16].data * 0.5 + 0.5).cpu(), 4))
    if epoch == 0:
        vutils.save_image(fixed_data.cpu().data * 0.5 + 0.5,
                          os.path.join(args.save_path, png_real),
                          normalize=True)

    plt.subplot(122)
    vutils.save_image(rec_imgs.cpu().data * 0.5 + 0.5,
                      os.path.join(args.save_path, png_rec),
                      normalize=True)
    #show_subplot('Rec', make_grid((rec_imgs[:16].data * 0.5 + 0.5).cpu(), 4))

    plt.figure(figsize=(18, 5))

    plt.subplot(131)
    vutils.save_image(cond_imgs.cpu().data * 0.5 + 0.5,
                      os.path.join(args.save_path, png_cond),
                      normalize=True)
    #show_subplot('conditional', make_grid((cond_imgs[:16].data * 0.5 + 0.5).cpu(), 4))

    plt.subplot(121)
    vutils.save_image(int_imgs.cpu().data * 0.5 + 0.5,
                      os.path.join(args.save_path, png_int),
                      normalize=True)
    #show_subplot('intervention', make_grid((int_imgs[:16].data * 0.5 + 0.5).cpu(), 4))

    plt.subplot(122)
    vutils.save_image(cf_imgs.cpu().data * 0.5 + 0.5,
                      os.path.join(args.save_path, png_cf),
                      normalize=True)
    #show_subplot('cf', make_grid((cf_imgs[:16].data * 0.5 + 0.5).cpu(), 4))

    plt.figure(figsize=(18, 5))
    plt.subplot(121)
    vutils.save_image(a0_imgs.cpu().data * 0.5 + 0.5,
                      os.path.join(args.save_path, png_a0),
                      normalize=True)
    #show_subplot('a0', make_grid((a0_imgs[:16].data * 0.5 + 0.5).cpu(), 4))

    plt.subplot(122)
    vutils.save_image(a1_imgs.cpu().data * 0.5 + 0.5,
                      os.path.join(args.save_path, png_a1),
                      normalize=True)
    #show_subplot('a1', make_grid((a1_imgs[:16].data * 0.5 + 0.5).cpu(), 4))