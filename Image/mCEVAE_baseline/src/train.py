import os
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
from IPython.display import clear_output

from show_img import plot_results, test_results

def train(args, model, train_loader, valid_loader):

    model = model.train()

    params_without_delta = [param for name, param in model.named_parameters()]
    opt_without_delta = optim.Adam(params_without_delta, lr=args.lr)
    scheduler_without_delta = optim.lr_scheduler.MultiStepLR(opt_without_delta, milestones=[pow(3, i) for i in range(7)], gamma=pow(0.1, 1/7))

    whole_losses = []
    x_recon_losses = []
    l_recon_losses = []
    u_kl_losses = []
    cf_recon_losses = []

    debug = True

    best_loss, best_epoch = 1e20, 0
    for epoch_i in range(args.max_epochs):
        model.train()
        itr = 0
        for cur_data, cur_sens, cur_rest, cur_des in tqdm(train_loader):

            cur_data, cur_sens, cur_rest, cur_des = \
                cur_data.to(args.device), cur_sens.to(args.device), cur_rest.to(args.device), cur_des.to(args.device)

            loss_val, x_recon_loss_val, l_recon_loss_val, u_kl_loss_val, x_cf_recon\
                = model.calculate_loss(cur_data, cur_sens, cur_rest, cur_des, \
                                       beta1=args.beta1, beta2=args.beta2, beta3=args.beta3, beta4=args.beta4)


            opt_without_delta.zero_grad()
            loss_val.backward()
            opt_without_delta.step()

            whole_losses.append(loss_val.item())
            x_recon_losses.append(x_recon_loss_val.item())
            l_recon_losses.append(l_recon_loss_val.item())
            u_kl_losses.append(u_kl_loss_val.item())
            cf_recon_losses.append(x_cf_recon.item())

            if debug == True and itr % 100 == 0:
                print('x_recon_loss_val: ', x_recon_loss_val.item(), '| scaled ', x_recon_loss_val.item() * args.beta1)
                print('l_recon_loss_val: ', l_recon_loss_val.item(), '| scaled ', l_recon_loss_val.item() * args.beta2)
                print('mmd: ', u_kl_loss_val.item(), '| scaled ', u_kl_loss_val.item() * args.beta3)
                print('mmd_A: ', x_cf_recon.item(), '| scaled ', x_cf_recon.item() * args.beta4)
            itr += 1

        scheduler_without_delta.step()

        clear_output(True)
        if (epoch_i % args.save_per_epoch == 0 and epoch_i < 100) or epoch_i % 10 == 0:
            plot_results(model, args, epoch_i, whole_losses, x_recon_losses, l_recon_losses, \
                         u_kl_losses, cf_recon_losses)

        print('Epoch {}'.format(epoch_i))
        mean_whole = np.array(whole_losses[-len(train_loader):]).mean()
        mean_x_recon = np.array(x_recon_losses[-len(train_loader):]).mean()
        mean_l_recon = np.array(l_recon_losses[-len(train_loader):]).mean()
        mean_u_kl = np.array(u_kl_losses[-len(train_loader):]).mean()

        print('Mean loss: {:.4f}'.format(mean_whole))
        print('Mean MSE(X): {:.4f}, scaled MSE(X): {:.4f}'.format(mean_x_recon, args.beta1 * mean_x_recon))
        print('Mean MSE(L): {:.4f}, scaled MSE(L): {:.4f}'.format(mean_l_recon, args.beta2 * mean_l_recon))
        print('Mean u_kl: {:.4f}, scaled u_kl: {:.4f}'.format(mean_u_kl, args.beta3 * mean_u_kl))
        print()

        model.eval()
        total_loss = 0
        #cnt = 0
        for cur_data, cur_sens, cur_rest, cur_des in tqdm(valid_loader):
            cur_data, cur_sens, cur_rest, cur_des = \
                cur_data.to(args.device), cur_sens.to(args.device), cur_rest.to(args.device), cur_des.to(args.device)

            with torch.no_grad():
                loss_val, x_recon_loss_val, l_recon_loss_val, u_kl_loss_val, x_cf_recon \
                    = model.calculate_loss(cur_data, cur_sens, cur_rest, cur_des, beta1=args.beta1, beta2=args.beta2,
                                                                              beta3=args.beta3, beta4=args.beta4)

                total_loss += loss_val

        if best_loss > total_loss:
            model_path = os.path.join(args.save_path, 'model.pth')
            torch.save(model, model_path)
            best_epoch = epoch_i
            best_loss = total_loss

        if (epoch_i % args.save_per_epoch == 0 and epoch_i < 100) or epoch_i % 10 == 0:
            cur_data_in, cur_sens_in, cur_rest_in, cur_des_in = args.valid_batch
            test_results(args, model, cur_data_in, cur_sens_in, cur_rest_in, cur_des_in, test=True, epoch=epoch_i)

        if epoch_i - best_epoch > args.early_stop:
            print('early_stop')
            break