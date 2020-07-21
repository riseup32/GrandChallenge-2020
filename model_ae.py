import torch
import torch.nn as nn



class Encoder(nn.Module):
    def __init__(self, conv_dim):
        super(Encoder, self).__init__()
        self.conv_dim = conv_dim
        if(conv_dim == '1d'):
            self.conv1 = nn.Sequential(
                nn.Conv1d(1, 4, (11, 1)), # (1, 40, 100) -> (4, 30, 100)
                nn.BatchNorm2d(4),
                nn.ReLU()
            )
            self.conv2 = nn.Sequential(
                nn.Conv1d(4, 4, (11, 1)), # (4, 30, 100) -> (4, 20, 100)
                nn.BatchNorm2d(4),
                nn.ReLU()
            )
            self.conv3 = nn.Sequential(
                nn.Conv1d(4, 8, (11, 1)), # (4, 20, 100) -> (8, 10, 100)
                nn.BatchNorm2d(8),
                nn.ReLU()
            )
            self.conv4 = nn.Sequential(
                nn.Conv1d(8, 8, (10, 1)), # (8, 10, 100) -> (8, 1, 100)
                nn.BatchNorm2d(8),
                nn.ReLU()
            )
        elif(conv_dim == '2d'):
            self.conv1 = nn.Sequential(
                nn.Conv2d(1, 4, (5, 3), padding=(0, 1)), # (1, 128, 50) -> (4, 124, 50)
                nn.BatchNorm2d(4),
                nn.ReLU(),
                nn.Conv2d(4, 4, (5, 3), padding=(0, 1)),  # (4, 124, 50) -> (4, 120, 50)
                nn.BatchNorm2d(4),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)  # (4, 120, 50) -> (4, 60, 25)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(4, 8, (5, 3), padding=(0, 1)), # (4, 60, 25) -> (8, 56, 25)
                nn.BatchNorm2d(8),
                nn.ReLU(),
                nn.Conv2d(8, 8, (5, 3), padding=(0, 1)),  # (8, 56, 25) -> (8, 52, 25)
                nn.BatchNorm2d(8),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)  # (8, 52, 25) -> (8, 26, 12)
            )
            self.conv3 = nn.Sequential(
                nn.Conv2d(8, 16, (5, 3), padding=0), # (8, 26, 12) -> (16, 22, 10)
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d((2, 1), (2, 1)),  # (16, 22, 10) -> (16, 11, 10)
                nn.Conv2d(16, 16, 3, padding=(1, 0)),  # (16, 11, 10) -> (16, 11, 8)
                nn.BatchNorm2d(16),
                nn.ReLU()
            )
        else:
            raise ValueError("Convolution dimension not found: %s" % (conv_dim))

    def forward(self, x):
        if(self.conv_dim == '1d'):
            out = self.conv1(x)
            out = self.conv2(out)
            out = self.conv3(out)
            out = self.conv4(out)
            # out = out.contiguous().view(x.size()[0], -1)  # (800,)
        elif(self.conv_dim == '2d'):
            out = self.conv1(x)
            out = self.conv2(out)
            out = self.conv3(out)
            # out = out.contiguous().view(x.size()[0], -1)  # (1408,)
        return out


class Decoder(nn.Module):
    def __init__(self, conv_dim):
        super(Decoder, self).__init__()
        self.conv_dim = conv_dim
        if(conv_dim == '1d'):
                self.conv_trans1 = nn.Sequential(
                    nn.ConvTranspose1d(8, 8, (10, 1)),  # (8, 1, 100) -> (8, 10, 100)
                    nn.BatchNorm2d(8),
                    nn.ReLU()
                )
                self.conv_trans2 = nn.Sequential(
                    nn.ConvTranspose1d(8, 4, (11, 1)),  # (8, 10, 100) -> (4, 20, 100)
                    nn.BatchNorm2d(4),
                    nn.ReLU()
                )
                self.conv_trans3 = nn.Sequential(
                    nn.ConvTranspose2d(4, 4, (11, 1)),  # (4, 20, 100) -> (4, 30, 100)
                    nn.BatchNorm2d(4),
                    nn.ReLU()
                )
                self.conv_trans4 = nn.Sequential(
                    nn.ConvTranspose2d(4, 1, (11, 1)),  # (4, 30, 100) -> (1, 40, 100)
                )
        elif(conv_dim == '2d'):
            self.conv_trans1 = nn.Sequential(
                    nn.ConvTranspose2d(16, 16, 3, 1, (1, 0)),  # (16, 11, 8) -> (16, 11, 10)
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
                    nn.ConvTranspose2d(16, 16, (5, 3), (2, 1), (0, 0), (1, 0)),  # (16, 11, 10) -> (16, 26, 12)
                    nn.BatchNorm2d(16),
                    nn.ReLU()
            )
            self.conv_trans2 = nn.Sequential(
                    nn.ConvTranspose2d(16, 8, (5, 3), 2, 0, (1, 0)),  # (16, 26, 12) -> (8, 56, 25)
                    nn.BatchNorm2d(8),
                    nn.ReLU(),
                    nn.ConvTranspose2d(8, 8, (5, 3), 1, (0, 1)),  # (8, 56, 25) -> (8, 60, 25)
                    nn.BatchNorm2d(8),
                    nn.ReLU()
            )
            self.conv_trans3 = nn.Sequential(
                    nn.ConvTranspose2d(8, 4, (5, 3), 2, 0, (1, 0)),  # (8, 60, 25) -> (4, 124, 51)
                    nn.BatchNorm2d(4),
                    nn.ReLU(),
                    nn.ConvTranspose2d(4, 1, (5, 2), 1, (0, 1))  # (4, 124, 51) -> (1, 128, 50)
            )
        else:
            raise ValueError("Convolution dimension not found: %s" % (conv_dim))

    def forward(self, x):
        if(self.conv_dim == '1d'):
            # out = x.contiguous().view(x.size()[0], 8, 1, 100)
            out = self.conv_trans1(x)
            out = self.conv_trans2(out)
            out = self.conv_trans3(out)
            out = self.conv_trans4(out)
        elif(self.conv_dim == '2d'):
            # out = x.contiguous().view(x.size()[0], 16, 11, 8)
            out = self.conv_trans1(x)
            out = self.conv_trans2(out)
            out = self.conv_trans3(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, conv_dim):
        super(Discriminator, self).__init__()
        if(conv_dim == '1d'):
            self.fc = nn.Sequential(
                nn.Linear(400, 256),
                nn.Dropout(0.2),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
        elif(conv_dim == '2d'):
            self.fc = nn.Sequential(
                nn.Linear(1408, 256),
                nn.Dropout(0.2),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
        else:
            raise ValueError("Convolution dimension not found: %s" % (conv_dim))

    def forward(self, x):
        out = self.fc(x)
        return out
