import os
import glob
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from preprocessing import convert_spectrograms, convert_tensor
from model_ser import CLDNN
from utils.optimization import WarmupLinearSchedule



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-conv_dim", default='1d', type=str,
                       help="Dimension of the Convolution kernel.{1d, 2d}")
    parser.add_argument("-checkpoint", default='', type=str,
                       help="pretrained model checkpoint.")
    parser.add_argument("-hidden_size", default=64, type=int,
                       help="The hidden size of LSTM.")
    parser.add_argument("-num_layers", default=2, type=int,
                       help="Number of LSTM layers.")
    parser.add_argument("-bidirectional", default='true', type=str,
                       help="Whether to use a bidirectional LSTM.")
    parser.add_argument("-with_focus_attn", default='false', type=str,
                       help="Whether to use a focus attention mechanism.")

    parser.add_argument("-batch_size", default=128, type=int,
                       help="Total batch size for training.")
    parser.add_argument("--learning_rate", default=0.001, type=int,
                       help="The initial learning rate for Adam.")
    parser.add_argument("-num_epochs", default=100, type=int,
                       help="Total number of training epochs to perform.")
    parser.add_argument("-multi_gpu", default='false', type=str,
                       help="Whether to use a multi-gpu.")

    parser.add_argument("-use_warmup", default='false', type=str,
                       help="Whether to use a warm-up.")
    parser.add_argument("-warmup_proportion", default=0.1, type=float,
                       help="Proportion of training to perform linear learning rate warmup for.")
    parser.add_argument("-gradient_accumulation_steps", default=1, type=int,
                       help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument("-normal_data_dir", default='./wav_data/normal', type=str,
                       help="The normal data directory.")
    parser.add_argument("-violence_data_dir", default='./wav_data/violence', type=str,
                       help="The violence data directory.")
    parser.add_argument("-output_dir", default='output', type=str,
                       help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("-save_checkpoint_steps", default=50, type=int)
    
    args = parser.parse_args()

    args.bidirectional = True if(args.bidirectional == 'true') else False
    args.with_focus_attn = True if(args.with_focus_attn == 'true') else False
    
    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if(args.multi_gpu == 'true'):
        args.batch_size = args.batch_size * torch.cuda.device_count()

    if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)


    normal_data = glob.glob(os.path.join(args.normal_data_dir, '**', '*wav'), recursive=True)
    normal_data = sorted(normal_data)

    violence_data = glob.glob(os.path.join(args.violence_data_dir, '**', '*wav'), recursive=True)
    violence_data = sorted(violence_data)

    X_normal = convert_spectrograms(normal_data, conv_dim=args.conv_dim)
    X_violence = convert_spectrograms(violence_data, conv_dim=args.conv_dim)

    X_normal = convert_tensor(X_normal)
    X_violence = convert_tensor(X_violence)

    y_normal = np.zeros(len(X_normal))
    y_violence = np.ones(len(X_violence))

    y_normal = torch.tensor(y_normal).float().view(-1, 1)
    y_violence = torch.tensor(y_violence).float().view(-1, 1)

    X = torch.cat((X_normal, X_violence), dim=0)
    y = torch.cat((y_normal, y_violence), dim=0)

    train_ds = TensorDataset(X, y)
    train_dataloader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)


    if(args.multi_gpu == 'true'):
        model = CLDNN(conv_dim=args.conv_dim, checkpoint=args.checkpoint, hidden_size=args.hidden_size,
                      num_layers=args.num_layers, bidirectional=args.bidirectional, with_focus_attn=args.with_focus_attn)
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = CLDNN(conv_dim=args.conv_dim, checkpoint=args.checkpoint, hidden_size=args.hidden_size,
                      num_layers=args.num_layers, bidirectional=args.bidirectional,
                      with_focus_attn=args.with_focus_attn).to(device)


    loss_func = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    if(args.use_warmup == 'true'):
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_epochs
        opt_scheduler = WarmupLinearSchedule(optimizer, warmup_steps=t_total * args.warmup_proportion, t_total=t_total)


    def train(dataloader, epochs):
        print('Start training')
        for epoch in range(epochs):
            train_loss = 0
            nb_train_steps = 0
            correct = 0
            num_samples = 0

            for X_batch, y_batch in dataloader:
                if cuda:
                    X_batch = X_batch.cuda()
                    y_batch = y_batch.cuda()

                optimizer.zero_grad()

                outputs = model(X_batch)

                loss = loss_func(outputs, y_batch)
                loss.backward()

                optimizer.step()
                if(args.use_warmup == 'true'):
                    opt_scheduler.step()

                train_loss += loss.mean().item()
                nb_train_steps += 1

                outputs = (outputs >= 0.5).float()
                correct += (outputs == y_batch).float().sum()
                num_samples += len(X_batch)

            train_loss = train_loss / nb_train_steps
            accuracy = correct / num_samples

            for param_group in optimizer.param_groups:
                lr = param_group['lr']
            print('epoch: {:3d},    lr={:6f},    loss={:5f},    accuracy={:5f}'
                  .format(epoch+1, lr, train_loss, accuracy))

            if((epoch+1) % args.save_checkpoint_steps == 0):
                model_checkpoint = "%s_%s_step_%d.pt" % ('CLDNN', args.conv_dim, epoch+1)
                output_model_file = os.path.join(args.output_dir, model_checkpoint)
                if(args.multi_gpu == 'true'):
                    torch.save(model.module.state_dict(), output_model_file)
                else:
                    torch.save(model.state_dict(), output_model_file)
                print("Saving checkpoint %s" % output_model_file)


    train(train_dataloader, args.num_epochs)


if __name__ == "__main__":
    main()

