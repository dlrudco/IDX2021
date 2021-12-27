import torch
import time
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import numpy as np
import os
import shutil
from torch.optim import lr_scheduler
from .dataset import RespiDataset
import torchvision
import random
from tqdm import tqdm
from torch.cuda.amp import autocast,GradScaler
from .stats import calculate_stats
import warnings
warnings.filterwarnings("ignore")
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, required=True, metavar='PFX', help='prefix for logging & checkpoint saving')
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--validate", action='store_true')
    parser.add_argument("--fixed_size", action='store_true')
    parser.add_argument("--mixup", action='store_true')
    parser.add_argument("--multi_label", action='store_true')
    parser.add_argument("--data_size", type=int, default=None)
    parser.add_argument("--fold_N", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--model',type=str)
    parser.add_argument('--optimizer',type=str)
    parser.add_argument('--data_path',type=str)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()

    return args

def save_checkpoint(state, is_best, prefix):
    os.makedirs('./checkpoints/', exist_ok=True)
    filename='./checkpoints/%s_checkpoint.pth'%prefix
    torch.save(state, filename)
    if is_best:
        # print("\nSave new best model\n")
        shutil.copyfile(filename, './checkpoints/%s_model_best.pth'%prefix)

def validate_epoch(args, epoch, dataloader, model, writer, fold_idx=0):
    model.eval()
    criterion=nn.CrossEntropyLoss()
    count = 0
    running_loss = 0.

    A_predictions = []
    A_targets = []
    A_loss = []
    for inputs, labels in tqdm(dataloader, total=dataloader.__len__(), leave=False):
        inputs = inputs.to(device).float()
        labels = labels.to(device).float()
        with autocast():
            with torch.no_grad():
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                loss = criterion(outputs, torch.argmax(labels, axis=1).long())
        A_predictions.append(outputs.to('cpu').detach())
        A_targets.append(labels.to('cpu').detach())
        # statistics
        running_loss += loss.item() * inputs.size(0)
        count += inputs.size(0)
    audio_output = torch.cat(A_predictions)
    target = torch.cat(A_targets)
    stats = calculate_stats(audio_output, target, save_steps=1)
    epoch_loss = running_loss / count
    

    return epoch_loss, stats

def train_epoch(args, epoch, dataloader, model, optimizer, writer, fold_idx=0):
    model.train()
    criterion=nn.CrossEntropyLoss()
    scaler = GradScaler()
    count = 0
    running_loss = 0.

    for it, (inputs, labels) in tqdm(enumerate(dataloader), total=dataloader.__len__(), leave=False):
        # inputs=torch.unsqueeze(inputs,1)
        # breakpoint()
        inputs = inputs.to(device).float()
        labels = labels.to(device).float()
        
        # zero the parameter gradients
        optimizer.zero_grad()
        model.zero_grad()
        
        with autocast():
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = criterion(outputs, torch.argmax(labels, axis=1).long())
                    
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * inputs.size(0)
        
        count += inputs.size(0)
        writer.add_scalar(f'{fold_idx}/(Train)Loss', loss.item(), epoch*dataloader.__len__() + it)

    epoch_loss = running_loss / count

    return epoch_loss


def train_official(args):
    writer = SummaryWriter(f'diagnosis_runs/{args.prefix}')
    fold_idx = 'official'
    from .dataset import RespiDataset_Diag
    train_dataset = RespiDataset_Diag(split='train', data_dir=args.data_path+'/train', initialize=True, 
            num_mel=128, multi_label=args.multi_label, 
            mean=None, std=None, fixed_length=args.data_size)
    val_dataset = RespiDataset_Diag(split='val', data_dir=args.data_path.replace('official','wav_16k'), initialize=True, 
            num_mel=128, multi_label=args.multi_label, 
            mean=None, std=None, fixed_length=args.data_size)
    
    train_loader = DataLoader(train_dataset, num_workers=16, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, num_workers=16, batch_size=args.batch_size*2, shuffle=False, drop_last=False)
    best_acc = 0.
    if args.model == 'resnet18':
        from .models import ResidualNet
        model = ResidualNet(network_type='Sound', depth=18, num_classes=8, att_type=None, in_channel=1)
    elif args.model == 'cnnlstm':
        from .models import CNNLSTM
        model = CNNLSTM(cnn_unit=32, lstm_unit=48, fc_unit=64, output_dim=8, 
                N_MELS=128, only_last=True, residual=True).cuda()
    elif args.model == 'lstm':
        from .models import LSTM
        model = LSTM(input_dim=128, hidden_dim=48, batch_size=args.batch_size, output_dim=8, 
                only_last=False).cuda()
    elif args.model == 'ast':
        from .models import ASTModel
        model = ASTModel(label_dim=8, fstride=10, tstride=10, input_fdim=128,
                              input_tdim=256, imagenet_pretrain=True,
                              audioset_pretrain=True, model_size='base384')
    else:
        raise AssertionError

    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
    model.to(device)
    trainables = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(trainables, lr= args.lr, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(5,96,15)), gamma=0.85)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(trainables, lr= args.lr, weight_decay=5e-7, betas=(0.95, 0.999))
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(5,96,15)), gamma=0.85)
    else:
        pass
    

    for epoch in tqdm(range(args.num_epochs), leave=False):
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                tl= train_epoch(args, epoch, train_loader, model, optimizer, writer, fold_idx='official')  # Set model to training mode
            else:
                vl, stats = validate_epoch(args, epoch, val_loader, model, writer, fold_idx='official')   # Set model to evaluate mode

            if phase=="train":
                writer.add_scalar(f'{fold_idx}/(Train)Epoch Loss',tl,epoch)
            elif phase == "val":
                mAP = np.nanmean([stat['AP'] for stat in stats])
                # mAUC = np.mean([stat['auc'] for stat in stats])
                writer.add_scalar(f'{fold_idx}/0(Val)mAP',mAP,epoch)
                # writer.add_scalar(f'{fold_idx}/(Val)mAUC',mAUC,epoch)
                for i in range(8):
                    writer.add_scalar(f'{fold_idx}/(Val)AP-{train_dataset.idx2cls[i]}', stats[i]['AP'], epoch)
                writer.add_scalar(f'{fold_idx}/(Val)Loss',vl,epoch)
                writer.add_scalar(f'{fold_idx}/1(Val)Acc',stats[0]['acc'],epoch)
                middle_ps = [stat['precisions'][int(len(stat['precisions'])/2)] for stat in stats]
                average_precision = np.nanmean(middle_ps)
                writer.add_scalar(f'{fold_idx}/(Val)Precision-Average',average_precision,epoch)
                for i in range(8):
                    writer.add_scalar(f'{fold_idx}/(Val)Precision-{train_dataset.idx2cls[i]}', middle_ps[i], epoch)
                middle_rs = [stat['recalls'][int(len(stat['recalls'])/2)] for stat in stats]
                average_recall = np.nanmean(middle_rs)
                for i in range(8):
                    writer.add_scalar(f'{fold_idx}/(Val)Recall-{train_dataset.idx2cls[i]}', middle_rs[i], epoch)
                writer.add_scalar(f'{fold_idx}/(Val)Recall-Average', average_recall, epoch)
                
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_val_acc': best_acc,
                    'opts' : optimizer.state_dict(),
                    }, mAP > best_acc, args.prefix+f'_official')
                if mAP > best_acc:
                    best_acc = mAP
                writer.add_scalar(f'{fold_idx}/0(Val)mAP-best',best_acc,epoch)
                    
        scheduler.step()
    checkpoint = torch.load(f'checkpoints/{args.prefix}_official_model_best.pth')
    model.load_state_dict(checkpoint['state_dict'])
    return model


def validate_official(args, model=None):
    fold_idx = 'official'
    from .dataset import RespiDataset_Diag
    # val_dataset = RespiDataset_Diag(split='val', data_dir=args.data_path.replace('official','wav_16k'), initialize=True, 
    #         num_mel=128, multi_label=args.multi_label, 
    #         mean=None, std=None, fixed_length=args.data_size)
    val_dataset = RespiDataset_Diag(split='val', data_dir=args.data_path+'/val', initialize=True, 
            num_mel=128, multi_label=args.multi_label, 
            mean=None, std=None, fixed_length=args.data_size)
    
    val_loader = DataLoader(val_dataset, num_workers=16, batch_size=args.batch_size*2, shuffle=False, drop_last=False)        

        
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
    model.to(device)

    vl, stats = validate_epoch(args, 0, val_loader, model, None, fold_idx='official')   # Set model to evaluate mode

    mAP = np.nanmean([stat['AP'] for stat in stats])
    # mAUC = np.mean([stat['auc'] for stat in stats])
    middle_ps = [stat['precisions'][int(len(stat['precisions'])/2)] for stat in stats]
    average_precision = np.mean(middle_ps)
    middle_rs = [stat['recalls'][int(len(stat['recalls'])/2)] for stat in stats]
    average_recall = np.mean(middle_rs)
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
    print(f"mean Average Precision: {mAP}")
    print(f"Overall Accuracy Score: {stats[0]['acc']}")
            
class Trainer():
    def __init__(self):
        self.model=None
        self.args = args_parser()
        if self.args.seed != -1:
            torch.manual_seed(self.args.seed)
            torch.cuda.manual_seed_all(self.args.seed)
            random.seed(self.args.seed)
            np.random.seed(self.args.seed)
            os.environ['PYTHONHASHSEED'] = str(self.args.seed)            
            torch.backends.cudnn.deterministic = True
    
        if self.args.fixed_size:
            assert (self.args.data_size is not None) and (self.args.batch_size is not None), ("batch_size and data_size should be specified if fixed_size == True")
    
    def run(self):
        if self.args.train:
            self.train()
        if self.args.validate:
            if not self.model:
                if self.args.model == 'resnet18':
                    from .models import ResidualNet
                    model = ResidualNet(network_type='Sound', depth=18, num_classes=8, att_type=None, in_channel=1)
                elif self.args.model == 'cnnlstm':
                    from .models import CNNLSTM
                    model = CNNLSTM(cnn_unit=32, lstm_unit=48, fc_unit=64, output_dim=8, 
                            N_MELS=128, only_last=True, residual=True).cuda()
                elif self.args.model == 'lstm':
                    from .models import LSTM
                    model = LSTM(input_dim=128, hidden_dim=48, batch_size=args.batch_size, output_dim=8, 
                            only_last=False).cuda()
                elif self.args.model == 'ast':
                    from .models import ASTModel
                    model = ASTModel(label_dim=8, fstride=10, tstride=10, input_fdim=128,
                                  input_tdim=256, imagenet_pretrain=True,
                                  audioset_pretrain=True, model_size='base384')
                else:
                    raise AssertionError
                checkpoint = torch.load(f'checkpoints/{self.args.prefix}_official_model_best.pth')
                model.load_state_dict(checkpoint['state_dict'])
                self.model = model
            self.validate()

    def train(self):
        self.model = train_official(self.args)

    def validate(self):
        validate_official(self.args, self.model)

if __name__=="__main__":
    args = args_parser()
    if args.seed != -1:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        os.environ['PYTHONHASHSEED'] = str(args.seed)            
        torch.backends.cudnn.deterministic = True
    
    if args.fixed_size:
        assert (args.data_size is not None) and (args.batch_size is not None), ("batch_size and data_size should be specified if fixed_size == True")
    model = train_official(args)
    validate_official(args, model)
