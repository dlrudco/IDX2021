from torchvision.models import resnet18

import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim, num_layers=2, only_last=True):
        super(LSTM, self).__init__()
        self.only_last = only_last
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        # trying
        # setup LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=False)
        self.pool = nn.AdaptiveAvgPool1d(10)
        # setup output layer
        if only_last:
            self.linear1 = nn.Linear(self.hidden_dim, self.hidden_dim//2)
            self.linear2 = nn.Linear(self.hidden_dim//2, output_dim)
        else:
            self.linear1 = nn.Linear(10*self.hidden_dim, self.hidden_dim)
            self.linear2 = nn.Linear(self.hidden_dim, output_dim)

    def forward(self, input, hidden=None):
        # lstm step => then ONLY take the sequence's final timetep to pass into the linear/dense layer
        # Note: lstm_out contains outputs for every step of the sequence we are looping over (for BPTT)
        # but we just need the output of the last step of the sequence, aka lstm_out[-1]
        if len(input.shape) > 3:
            input = input.squeeze()
        input = input.permute(2,0,1)
        lstm_out, hidden = self.lstm(input, hidden)
        if self.only_last:
            lstm_out = lstm_out.squeeze()[-1]
        else:
            lstm_out= lstm_out.permute(1,2,0)
            lstm_out = self.pool(lstm_out).flatten(-2)
            # lstm_out = torch.max(lstm_out.squeeze(), dim=0)[0]
        logits = self.linear1(lstm_out)
        # logits = self.dropout(logits)
        logits = self.linear2(logits)
        return logits, hidden

    def get_accuracy(self, logits, target):
        """ compute accuracy for training round """
        corrects = (
                torch.max(logits, 1)[1].view(target.size()).data == target.data
        ).sum()
        accuracy = 100.0 * corrects / self.batch_size
        return accuracy.item()


class CNN_Block(nn.Module):
    def __init__(self, cnn_unit, residual=False):
        super().__init__()
        self.residual = residual
        self.conv1 = nn.Conv2d(1, cnn_unit, (3, 3), padding=1)
        # self.gn1 = nn.GroupNorm(cnn_unit,cnn_unit)
        self.gn1 = nn.BatchNorm2d(cnn_unit)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(cnn_unit, cnn_unit, (3, 3), padding=1)
        # self.gn2 = nn.GroupNorm(cnn_unit,cnn_unit)
        self.gn2 = nn.BatchNorm2d(cnn_unit)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d((2,2))

        self.conv3 = nn.Conv2d(cnn_unit, cnn_unit*2, (3, 3), padding=1)
        # self.gn3 = nn.GroupNorm(cnn_unit*2,cnn_unit*2)
        self.gn3 = nn.BatchNorm2d(cnn_unit*2)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(cnn_unit*2, cnn_unit*2, (3, 3), padding=1)
        # self.gn4 = nn.GroupNorm(cnn_unit*2,cnn_unit*2)
        self.gn4 = nn.BatchNorm2d(cnn_unit*2)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d((2,2))


        self.conv5 = nn.Conv2d(cnn_unit*2, cnn_unit*2, (3, 3), padding=1)
        # self.gn5 = nn.GroupNorm(cnn_unit*2,cnn_unit*2)
        self.gn5 = nn.BatchNorm2d(cnn_unit*2)
        self.relu5 = nn.ReLU()

        self.conv6 = nn.Conv2d(cnn_unit*2, cnn_unit*2, (3, 3), padding=1)
        # self.gn6 = nn.GroupNorm(cnn_unit*2,cnn_unit*2)
        self.gn6 = nn.BatchNorm2d(cnn_unit*2)
        self.relu6 = nn.ReLU()
        self.pool6 = nn.MaxPool2d((2,2))

        self.do = nn.Dropout(0.2)

        if self.residual:
            self.skip1 = nn.Sequential(
                nn.Conv2d(1, cnn_unit, (1,1)),
                nn.GroupNorm(cnn_unit,cnn_unit),
                nn.MaxPool2d((2,2))
            )
            self.skip2 = nn.Sequential(
                nn.Conv2d(cnn_unit, cnn_unit*2, (1,1)),
                nn.GroupNorm(cnn_unit*2,cnn_unit*2),
                nn.MaxPool2d((2,2))
            )
            self.skip3 = nn.Sequential(
                # nn.Conv2d(cnn_unit*2, cnn_unit*2, (1,1)),
                # nn.GroupNorm(cnn_unit*2,cnn_unit*2),
                nn.MaxPool2d((2,2))
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.gn2(out)
        out = self.pool2(out)
        if self.residual:
            identity = self.skip1(x)
            out += identity

        out = self.relu2(out)
        if self.residual:
            identity = out
        out = self.do(out)

        out = self.conv3(out)
        out = self.gn3(out)
        out = self.relu3(out)

        out = self.conv4(out)
        out = self.gn4(out)
        out = self.pool4(out)
        if self.residual:
            identity = self.skip2(identity)
            out += identity
        out = self.relu4(out)
        if self.residual:
            identity = out
        out = self.do(out)
        
        out = self.conv5(out)
        out = self.gn5(out)
        out = self.relu5(out)

        out = self.conv6(out)
        out = self.gn6(out)
        out = self.pool6(out)
        if self.residual:
            identity = self.skip3(identity)
            out += identity
        out = self.relu6(out)
        if self.residual:
            identity = out

        return out



class ConvStack(nn.Module):
    def __init__(self, n_mels, cnn_unit, fc_unit, mode='avg', residual=False):
        super().__init__()
        self.mode = mode
        # shape of input: (batch_size * 1 channel * frames * input_features)
        self.cnn = CNN_Block(cnn_unit, residual)
        # if self.mode == 'avg':
        #     self.gp = nn.AdaptiveAvgPool2d((16,48))
        # elif self.mode == 'max':
        #     self.gp = nn.AdaptiveMaxPool2d((16,48))
        # elif self.mode == 'nogp':
        #     self.gp = nn.Identity()
        # else:
        #     raise AssertionError('Mode should be one of ["avg", "max", "nogp"]')

        self.fc = nn.Sequential(
            nn.Linear(16*64, 16*8),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16*8, fc_unit),
            # nn.Dropout(0.1)
        )

    def forward(self, mel):
        x = self.cnn(mel)
        # x = self.gp(x)
        x = x.permute(0,3,1, 2).flatten(-2)
        x = self.fc(x)
        return x

class CNNLSTM(nn.Module):
    def __init__(self, cnn_unit, lstm_unit, fc_unit, output_dim, N_MELS=96, mode='max', only_last=False, residual=False):
        super(CNNLSTM, self).__init__()
        self.only_last=only_last
        self.fc_unit = fc_unit
        self.cnn_unit = cnn_unit
        self.lstm_unit = lstm_unit
        self.output_dim = output_dim
        self.conv_stack = ConvStack(N_MELS, self.cnn_unit, self.fc_unit, mode, residual=residual)
        self.lstm = nn.LSTM(input_size= self.fc_unit, hidden_size = self.lstm_unit, num_layers=2, 
                                batch_first=False)
        self.pool = nn.AdaptiveAvgPool1d(10)
        if self.only_last:
            self.linear1 = nn.Linear(self.lstm_unit, self.lstm_unit//2)
            self.linear2 = nn.Linear(self.lstm_unit//2, output_dim)
        else:
            self.linear1 = nn.Linear(10*self.lstm_unit, self.lstm_unit)
            self.linear2 = nn.Linear(self.lstm_unit, output_dim)


    def forward(self, input):
        x = self.conv_stack(input)  # (B x T x C)
        x = x.permute(1,0,2)
        x, hidden = self.lstm(x)
        if self.only_last:
            x = x.squeeze()[-1]
        else:
            breakpoint()
            # x = torch.max(x.squeeze(), dim=0)[0]
        x = self.linear1(x)
        x = self.linear2(x)
        return x, hidden

