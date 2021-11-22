import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import numpy as np
import os
import random
from tqdm import tqdm
from torch.cuda.amp import autocast
from stats import calculate_stats
import warnings

warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def validate_epoch(dataloader, model):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    count = 0
    running_loss = 0.

    a_predictions = []
    a_targets = []
    for inputs, labels in tqdm(dataloader, total=dataloader.__len__(), leave=False):
        inputs = inputs.to(device).float()
        labels = labels.to(device).float()
        with autocast():
            with torch.no_grad():
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                loss = criterion(outputs, torch.argmax(labels, axis=1).long())
        a_predictions.append(outputs.to('cpu').detach())
        a_targets.append(labels.to('cpu').detach())
        # statistics
        running_loss += loss.item() * inputs.size(0)
        count += inputs.size(0)
    audio_output = torch.cat(a_predictions)
    target = torch.cat(a_targets)
    stats = calculate_stats(audio_output, target, save_steps=1)
    epoch_loss = running_loss / count

    return epoch_loss, stats


def validate_official(args):
    from dataset import RespiDataset_Diag
    val_dataset = RespiDataset_Diag(split='val', data_dir=args.data_path.replace('official', 'wav_16k'),
                                    initialize=True, num_mel=128, multi_label=args.multi_label,
                                    mean=None, std=None, fixed_length=args.data_size)

    val_loader = DataLoader(val_dataset, num_workers=16, batch_size=args.batch_size * 2, shuffle=False, drop_last=False)

    if args.model == 'resnet18':
        from models import ResidualNet
        model = ResidualNet(network_type='Sound', depth=18, num_classes=8, att_type=None, in_channel=1)
    elif args.model == 'cnnlstm':
        from models import CNNLSTM
        model = CNNLSTM(cnn_unit=32, lstm_unit=48, fc_unit=64, output_dim=8,
                        N_MELS=128, only_last=True, residual=True).cuda()
    elif args.model == 'lstm':
        from models import LSTM
        model = LSTM(input_dim=128, hidden_dim=48, batch_size=args.batch_size, output_dim=8,
                     only_last=False).cuda()
    elif args.model == 'ast':
        from models import ASTModel
        model = ASTModel(label_dim=8, fstride=10, tstride=10, input_fdim=128,
                         input_tdim=256, imagenet_pretrain=True,
                         audioset_pretrain=True, model_size='base384')
    else:
        raise AssertionError

    checkpoint = torch.load(f'checkpoints/{args.prefix}_official_model_best.pth')
    sd = checkpoint['state_dict']
    model.load_state_dict(sd)
    model.to(device)

    vl, stats = validate_epoch(val_loader, model)  # Set model to evaluate mode

    mAP = np.nanmean([stat['AP'] for stat in stats])

    import matplotlib.pyplot as plt
    import os
    os.makedirs(f'results/{args.prefix}_official', exist_ok=True)
    for i in range(8):
        fig = plt.figure()
        plt.plot(stats[i]['recalls'], stats[i]['precisions'], marker='.')
        plt.xlabel('Recall')
        plt.xlim(-0.05, 1.05)
        plt.ylabel('Precision')
        plt.ylim(-0.05, 1.05)
        plt.savefig(f'results/{args.prefix}_official/PRCURVE_{val_dataset.idx2cls[i]}')
        plt.close(fig)
        print(val_dataset.idx2cls[i], '\t', stats[i]['AP'])
    print(mAP)
    print(stats[0]['acc'])

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, required=True, metavar='PFX',
                        help='prefix for logging & checkpoint saving')
    parser.add_argument("--fixed_size", action='store_true')
    parser.add_argument("--multi_label", action='store_true')
    parser.add_argument("--data_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--model', type=str)
    parser.add_argument('--data_path',type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = args_parser()
    if args.seed != -1:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        torch.backends.cudnn.deterministic = True

    # train_official(args)
    validate_official(args)
