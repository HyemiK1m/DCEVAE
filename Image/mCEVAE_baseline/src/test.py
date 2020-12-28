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
    fixed_data, fixed_sens, rest, des = fixed_batch
    fixed_data, fixed_sens, rest, des = \
    fixed_data.to(device), fixed_sens.to(device), rest.to(device), des.to(device)
    test_results(args, test_model, fixed_data, fixed_sens, rest, des, test=True, epoch=0)