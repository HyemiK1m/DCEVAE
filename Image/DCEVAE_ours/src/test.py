import os
import torch

from show_img import test_results, draw_cov

def test(args, test_loader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = os.path.join(args.save_path, 'model.pth')
    test_model = torch.load(model_path)
    test_model.to(device)
    test_model.eval()

    fixed_batch = next(iter(test_loader))
    fixed_data, fixed_sens, fixed_all, _, _, _ = fixed_batch
    fixed_data, fixed_sens, fixed_all = \
    fixed_data.to(device), fixed_sens.to(device), fixed_all.to(device)
    test_results(args, test_model, fixed_data, fixed_sens, test=True, epoch=0)

    draw_cov(args, test_model, test_loader, 0, test=True)