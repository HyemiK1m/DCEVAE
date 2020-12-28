import os
import torch

from show_img import test_results

def test(args, test_loader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = os.path.join(args.save_path, 'model.pth')
    test_model = torch.load(model_path)
    test_model.to(device)
    test_model.eval()

    fixed_batch = next(iter(test_loader))
    cur_data_in, cur_sens_in, cur_rest_in, cur_des_in = fixed_batch
    cur_data_in, cur_sens_in, cur_rest_in, cur_des_in = \
        cur_data_in.to(device), cur_sens_in.to(device), cur_rest_in.to(device), cur_des_in.to(device)
    test_results(args, test_model, cur_data_in, cur_sens_in, cur_rest_in, cur_des_in, test=True, epoch=0)