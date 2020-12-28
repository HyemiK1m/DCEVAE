# for seed in [1,2,3,4,5]:
#     for model in ['mCEVAE', 'DCEVAE', 'CVAE', 'CEVAE']:
#         for int in ['M', 'S']:
#             print('python3 tSNE.py --int %s --seed %d --model %s' %(int, seed, model))

for seed in [1]:
    for model in ['DCEVAE', 'CVAE', 'CEVAE']:
        for int in ['M', 'S']:
            print('python3 tSNE.py --int %s --seed %d --model %s' %(int, seed, model))