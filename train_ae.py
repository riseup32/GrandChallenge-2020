import os
import glob
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from preprocessing import convert_spectrograms, convert_tensor
from model_ae import Encoder, Decoder, Discriminator
from utils.optimization import WarmupLinearSchedule


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-model", default='ae', type=str,
                       help="The name of model.{ae, aae}")
    parser.add_argument("-conv_dim", default='1d', type=str,
                       help="Dimension of the Convolution kernel.{1d, 2d}")
    
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
    
    parser.add_argument("-sample_data_dir", default='./wav_data/pretrain', type=str,
                       help="The sample data directory.")
    parser.add_argument("-output_dir", default='output', type=str,
                       help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("-save_checkpoint_steps", default=50, type=int)
    parser.add_argument("-eval", default='false', type=str,
                       help="Whether to evaluate data.")
    
    args = parser.parse_args()


    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    n_z = 400 if (args.conv_dim == '1d') else 1408
    if(args.multi_gpu == 'true'):
        args.batch_size = args.batch_size * n_gpu
        
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


    samples_data = glob.glob(os.path.join(args.sample_data_dir, '**', '*.wav'), recursive=True)
    samples_data = sorted(samples_data)

    if(args.eval == 'true'):
        np.random.seed(42)
        idx = np.random.permutation(len(samples_data))
        train_idx = idx[:int(len(samples_data)*0.8)]
        eval_idx = idx[int(len(samples_data)*0.8):]
        
        train_samples = list(np.array(samples_data)[train_idx])
        eval_samples = list(np.array(samples_data)[eval_idx])
        
        X_train = convert_spectrograms(train_samples, conv_dim=args.conv_dim)
        X_eval = convert_spectrograms(eval_samples, conv_dim=args.conv_dim)
        
        X_train = convert_tensor(X_train)
        X_eval = convert_tensor(X_eval)
        
        train_dataloader = DataLoader(X_train, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
        eval_dataloader = DataLoader(X_eval, batch_size=args.batch_size, num_workers=0, drop_last=True)
    else:
        X_train = convert_spectrograms(samples_data, conv_dim=args.conv_dim)
        X_train = convert_tensor(X_train)  
        train_dataloader = DataLoader(X_train, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
        eval_dataloader = None


    if(args.multi_gpu == 'true'):
        encoder = Encoder(conv_dim=args.conv_dim)
        decoder = Decoder(conv_dim=args.conv_dim)
        encoder = torch.nn.DataParallel(encoder).cuda()
        decoder = torch.nn.DataParallel(decoder).cuda()
        if(args.model == 'aae'):
            discriminator = Discriminator(conv_dim=args.conv_dim)
            discriminator = torch.nn.DataParallel(discriminator).cuda()
    else:
        encoder = Encoder(conv_dim=args.conv_dim).to(device)
        decoder = Decoder(conv_dim=args.conv_dim).to(device)
        if(args.model == 'aae'):
            discriminator = Discriminator(conv_dim=args.conv_dim).to(device)


    def reset_grad():
        encoder.zero_grad()
        decoder.zero_grad()
        if(args.model == 'aae'):
            discriminator.zero_grad()


    loss_func = nn.MSELoss()
    enc_opt = optim.Adam(encoder.parameters(), lr=args.learning_rate)
    dec_opt = optim.Adam(decoder.parameters(), lr=args.learning_rate)
    if(args.model == 'aae'):
        disc_opt = optim.Adam(discriminator.parameters(), lr=args.learning_rate*0.1)

    if(args.use_warmup == 'true'):
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_epochs
        enc_scheduler = WarmupLinearSchedule(enc_opt, warmup_steps=t_total * args.warmup_proportion, t_total=t_total)
        dec_scheduler = WarmupLinearSchedule(dec_opt, warmup_steps=t_total * args.warmup_proportion, t_total=t_total)
        disc_scheduler = WarmupLinearSchedule(disc_opt, warmup_steps=t_total * args.warmup_proportion, t_total=t_total)


    def train(train_dataloader, eval_dataloader, epochs):
        print('Start training')
        for epoch in range(epochs):
            ### Train step
            tr_recon_loss = 0
            nb_train_steps = 0

            for X_batch in train_dataloader:
                ## Reconstruction
                X = Variable(X_batch)
                if cuda:
                    X = X.cuda()

                z_sample = encoder(X)
                X_sample = decoder(z_sample)

                recon_loss = loss_func(X_sample, X)

                recon_loss.backward()
                dec_opt.step()
                enc_opt.step()
                reset_grad()

                tr_recon_loss += recon_loss.mean().item()
                nb_train_steps += 1

                if(args.model == 'aae'):
                    ## Discriminator
                    for _ in range(5):
                        z_real = Variable(torch.randn(args.batch_size, n_z))
                        if cuda:
                            z_real = z_real.cuda()

                        z_fake = encoder(X).view(args.batch_size, -1)

                        D_real = discriminator(z_real)
                        D_fake = discriminator(z_fake)

                        D_loss = -(torch.mean(D_real) - torch.mean(D_fake))

                        D_loss.backward()
                        disc_opt.step()

                        # Weight clipping
                        for p in discriminator.parameters():
                            p.data.clamp_(-0.01, 0.01)

                        reset_grad()

                    ## Generator
                    z_fake = encoder(X).view(args.batch_size, -1)
                    D_fake = discriminator(z_fake)

                    G_loss = -torch.mean(D_fake)

                    G_loss.backward()
                    enc_opt.step()
                    reset_grad()

                if(args.use_warmup == 'true'):
                    enc_scheduler.step()
                    dec_scheduler.step()
                    if(args.model == 'aae'):
                        disc_scheduler.step()

            tr_recon_loss = tr_recon_loss / nb_train_steps

            if(args.eval == 'true'):
                ### Evaluate step
                encoder.eval()
                decoder.eval()
                if(args.model == 'aae'):
                    discriminator.eval()
                eval_recon_loss = 0
                nb_eval_steps = 0

                for X_batch in eval_dataloader:
                    with torch.no_grad():
                        z_sample = encoder(X_batch)
                        X_sample = decoder(z_sample)

                    tmp_eval_loss = loss_func(X_sample, X_batch)
                    eval_recon_loss += tmp_eval_loss.mean().item()
                    nb_eval_steps += 1

                eval_recon_loss = eval_recon_loss / nb_eval_steps

                for param_group in enc_opt.param_groups:
                    lr = param_group['lr']
                print('epoch: {:3d},    lr={:6f},    loss={:5f},    eval_loss={:5f}'
                      .format(epoch+1, lr, tr_recon_loss, eval_recon_loss))
            else:
                for param_group in enc_opt.param_groups:
                    lr = param_group['lr']
                print('epoch: {:3d},    lr={:6f},    loss={:5f}'
                      .format(epoch+1, lr, tr_recon_loss))
            
            if((epoch+1) % args.save_checkpoint_steps == 0):
                model_checkpoint = "%s_%s_step_%d.pt" % (args.model, args.conv_dim, epoch+1)
                output_model_file = os.path.join(args.output_dir, model_checkpoint)
                if(args.multi_gpu == 'true'):
                    torch.save(encoder.module.state_dict(), output_model_file)
                else:
                    torch.save(encoder.state_dict(), output_model_file)
                print("Saving checkpoint %s" % output_model_file)
            
            
    train(train_dataloader, eval_dataloader, args.num_epochs)


if __name__ == "__main__":
    main()
