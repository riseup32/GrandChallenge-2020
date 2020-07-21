import os
import glob
import argparse

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from preprocessing import preprocessing, convert_spectrograms, convert_tensor
from model_ae import Encoder
from utils.optimization import WarmupLinearSchedule



class MultiHeadedAttention(nn.Module):
    def __init__(self, num_attn_heads, attn_hidden_size, dropout_prob, with_focus_attn):
        super(MultiHeadedAttention, self).__init__()
        self.num_attn_heads = num_attn_heads
        self.hidden_size = attn_hidden_size
        self.dropout_prob = dropout_prob
        self.with_focus_attn = with_focus_attn

        self.attn_head_size = int(self.hidden_size / self.num_attn_heads)
        self.all_head_size = self.num_attn_heads * self.attn_head_size

        self.query = nn.Linear(self.hidden_size, self.all_head_size)
        self.key = nn.Linear(self.hidden_size, self.all_head_size)
        self.value = nn.Linear(self.hidden_size, self.all_head_size)

        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_prob)

        self.softmax = nn.Softmax(dim=-1)

        if(with_focus_attn == True):
            self.tanh = nn.Tanh()
            self.sigmoid = nn.Sigmoid()

            self.linear_focus_query = nn.Linear(num_attn_heads * self.attn_head_size, 
                                                num_attn_heads * self.attn_head_size)
            self.linear_focus_global = nn.Linear(num_attn_heads * self.attn_head_size, 
                                                 num_attn_heads * self.attn_head_size)

            up = torch.randn(num_attn_heads, 1, self.attn_head_size)
            self.up = Variable(up, requires_grad=True).cuda()
            torch.nn.init.xavier_uniform_(self.up)

            uz = torch.randn(num_attn_heads, 1, self.attn_head_size)
            self.uz = Variable(uz, requires_grad=True).cuda()
            torch.nn.init.xavier_uniform_(self.uz)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attn_heads, self.attn_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        key_len = hidden_states.size(1)

        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        if(self.with_focus_attn == True):
            glo = torch.mean(mixed_query_layer, dim=1, keepdim=True)

            c = self.tanh(self.linear_focus_query(mixed_query_layer) + self.linear_focus_global(glo))
            c = self.transpose_for_scores(c)

            p = c * self.up
            p = p.sum(3).squeeze()
            z = c * self.uz
            z = z.sum(3).squeeze()

            P = self.sigmoid(p) * key_len
            Z = self.sigmoid(z) * key_len

            j = torch.arange(start=0, end=key_len, dtype=P.dtype).unsqueeze(0).unsqueeze(0).unsqueeze(0).to('cuda')
            P = P.unsqueeze(-1)
            Z = Z.unsqueeze(-1)

            G = -(j - P)**2 * 2 / (Z**2)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attn_head_size)

        if(self.with_focus_attn == True):
            attention_scores = attention_scores + G

        attention_probs = self.softmax(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.o_proj(context_layer)
        return attention_output


class CLDNN(nn.Module):
    def __init__(self, conv_dim, checkpoint=None, hidden_size=64, num_layers=2,
                 bidirectional=True, with_focus_attn=False):
        super(CLDNN, self).__init__()
        self.conv_dim = conv_dim
        if(conv_dim == '1d'):
            self.encoder = Encoder(conv_dim)
            if checkpoint:
                self.encoder.load_state_dict(torch.load(checkpoint))
            self.attn = MultiHeadedAttention(num_attn_heads=4, attn_hidden_size=8, dropout_prob=0.1,
                                             with_focus_attn=with_focus_attn)
            self.lstm = nn.LSTM(8, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional)
            self.fc = nn.Sequential(
                nn.Linear(hidden_size*2 if bidirectional else hidden_size, 8),
            )
        elif(conv_dim == '2d'):
            self.encoder = Encoder(conv_dim)
            if checkpoint:
                self.encoder.load_state_dict(torch.load(checkpoint))
            self.attn = MultiHeadedAttention(num_attn_heads=4, attn_hidden_size=176, dropout_prob=0.1, 
                                             with_focus_attn=with_focus_attn)
            self.gap = nn.AdaptiveAvgPool2d((1, 11))
            self.lstm = nn.LSTM(11, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional)
            self.fc = nn.Sequential(
                nn.Linear(hidden_size*2 if bidirectional else hidden_size, 8),
            )
        else:
            raise ValueError("Convolution dimension not found: %s" % (conv_dim))

    def forward(self, x):
        if(self.conv_dim == '1d'):
            out = self.encoder(x)
            out = torch.squeeze(out, 2)
            out = out.permute(0, 2, 1)
            h = out
            out = self.attn(out)
            out = h + out
            out = out.permute(1, 0, 2)
            self.lstm.flatten_parameters()
            out, _ = self.lstm(out)
            out = out[-1]
            out = self.fc(out)
        elif(self.conv_dim == '2d'):
            out = self.encoder(x)
            out = out.permute(0, 3, 1, 2)
            h = out
            new_out_shape = out.size()[:2] + (out.size()[2] * out.size()[3],)
            out = out.view(*new_out_shape)
            out = self.attn(out)
            out = out.view(h.size())
            out = h + out
            out = self.gap(out)
            out = torch.squeeze(out, 2)
            out = out.permute(1, 0, 2)
            self.lstm.flatten_parameters()
            out, _ = self.lstm(out)
            out = out[-1]
            out = self.fc(out)
        return out


class CLDNN_G(nn.Module):
    def __init__(self, conv_dim, checkpoint=None, hidden_size=64, num_layers=2,
                 bidirectional=True, with_focus_attn=False):
        super(CLDNN_G, self).__init__()
        self.conv_dim = conv_dim
        if(conv_dim == '1d'):
            self.encoder = Encoder(conv_dim)
            if checkpoint:
                self.encoder.load_state_dict(torch.load(checkpoint))
            self.attn = MultiHeadedAttention(num_attn_heads=4, attn_hidden_size=8, dropout_prob=0.1,
                                             with_focus_attn=with_focus_attn)
            self.lstm = nn.LSTM(8, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional)
            self.fc = nn.Sequential(
                nn.Linear(hidden_size*2 if bidirectional else hidden_size, 1),
                nn.Sigmoid()
            )
        elif(conv_dim == '2d'):
            self.encoder = Encoder(conv_dim)
            if checkpoint:
                self.encoder.load_state_dict(torch.load(checkpoint))
            self.attn = MultiHeadedAttention(num_attn_heads=4, attn_hidden_size=176, dropout_prob=0.1, 
                                             with_focus_attn=with_focus_attn)
            self.gap = nn.AdaptiveAvgPool2d((1, 11))
            self.lstm = nn.LSTM(11, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional)
            self.fc = nn.Sequential(
                nn.Linear(hidden_size*2 if bidirectional else hidden_size, 1),
                nn.Sigmoid()
            )
        else:
            raise ValueError("Convolution dimension not found: %s" % (conv_dim))

    def forward(self, x):
        if(self.conv_dim == '1d'):
            out = self.encoder(x)
            out = torch.squeeze(out, 2)
            out = out.permute(0, 2, 1)
            h = out
            out = self.attn(out)
            out = h + out
            out = out.permute(1, 0, 2)
            self.lstm.flatten_parameters()
            out, _ = self.lstm(out)
            out = out[-1]
            out = self.fc(out)
        elif(self.conv_dim == '2d'):
            out = self.encoder(x)
            out = out.permute(0, 3, 1, 2)
            h = out
            new_out_shape = out.size()[:2] + (out.size()[2] * out.size()[3],)
            out = out.view(*new_out_shape)
            out = self.attn(out)
            out = out.view(h.size())
            out = h + out
            out = self.gap(out)
            out = torch.squeeze(out, 2)
            out = out.permute(1, 0, 2)
            self.lstm.flatten_parameters()
            out, _ = self.lstm(out)
            out = out[-1]
            out = self.fc(out)  
        return out
    
    
def train(train_dataloader, eval_dataloader, epochs):
        print('Start training')
        softmax = nn.Softmax(dim=1)
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            nb_train_steps = 0
            correct = 0
            num_samples = 0

            if(args.multi_task == 'true'):
                for X_batch, y_batch, y_g_batch in train_dataloader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    y_g_batch = y_g_batch.to(device)

                    optimizer.zero_grad()

                    outputs = model(X_batch)
                    outputs_g = model_g(X_batch)

                    loss_1 = loss_func(outputs, y_batch)
                    loss_2 = loss_func_g(outputs_g, y_g_batch)
                    loss = loss_1 + 0.8 * loss_2
                    loss.backward(retain_graph=True)

                    optimizer.step()
                    opt_scheduler.step()

                    train_loss += loss.mean().item()
                    nb_train_steps += 1

                    outputs = softmax(outputs)
                    outputs = torch.argmax(outputs, dim=1)
                    correct += (outputs == y_batch).float().sum()
                    num_samples += len(X_batch)

                train_loss = train_loss / nb_train_steps
                train_accuracy = correct / num_samples

                model.eval()
                eval_loss = 0
                nb_eval_steps = 0
                correct = 0
                num_samples = 0

                for X_batch, y_batch, y_g_batch in eval_dataloader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    y_g_batch = y_g_batch.to(device)
                    with torch.no_grad():
                        outputs = model(X_batch)
                        outputs_g = model_g(X_batch)

                    tmp_eval_loss_1 = loss_func(outputs, y_batch)
                    tmp_eval_loss_2 = loss_func_g(outputs_g, y_g_batch)
                    tmp_eval_loss = tmp_eval_loss_1 + 0.8 * tmp_eval_loss_2
                    eval_loss += tmp_eval_loss.mean().item()
                    nb_eval_steps += 1

                    outputs = softmax(outputs)
                    outputs = torch.argmax(outputs, dim=1)
                    correct += (outputs == y_batch).float().sum()
                    num_samples += len(X_batch)

                eval_loss = eval_loss / nb_eval_steps
                eval_accuracy = correct / num_samples
            else:
                for X_batch, y_batch in train_dataloader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)

                    optimizer.zero_grad()

                    outputs = model(X_batch)

                    loss = loss_func(outputs, y_batch)
                    loss.backward()

                    optimizer.step()
                    opt_scheduler.step()

                    train_loss += loss.mean().item()
                    nb_train_steps += 1

                    outputs = softmax(outputs)
                    outputs = torch.argmax(outputs, dim=1)
                    correct += (outputs == y_batch).float().sum()
                    num_samples += len(X_batch)

                train_loss = train_loss / nb_train_steps
                train_accuracy = correct / num_samples

                model.eval()
                eval_loss = 0
                nb_eval_steps = 0
                correct = 0
                num_samples = 0

                for X_batch, y_batch in eval_dataloader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    with torch.no_grad():
                        outputs = model(X_batch)

                    tmp_eval_loss = loss_func(outputs, y_batch)
                    eval_loss += tmp_eval_loss.mean().item()
                    nb_eval_steps += 1

                    outputs = softmax(outputs)
                    outputs = torch.argmax(outputs, dim=1)
                    correct += (outputs == y_batch).float().sum()
                    num_samples += len(X_batch)

                eval_loss = eval_loss / nb_eval_steps
                eval_accuracy = correct / num_samples

            for param_group in optimizer.param_groups:
                lr = param_group['lr']
            print('epoch: {:3d},    lr={:6f},    loss={:5f},    train_acc={:5f},    eval_loss={:5f},    eval_acc={:5f}'
                  .format(epoch+1, lr, train_loss, train_accuracy, eval_loss, eval_accuracy))

            '''
            if((epoch+1) % args.save_checkpoint_steps == 0):
                model_checkpoint = "%s_%s_step_%d.pt" % ('CLDNN', args.conv_dim, epoch+1)
                output_model_file = os.path.join(args.output_dir, model_checkpoint)
                if(args.multi_gpu == 'true'):
                    torch.save(model.module.state_dict(), output_model_file)
                else:
                    torch.save(model.state_dict(), output_model_file)
                print("Saving checkpoint %s" % output_model_file)
            '''
        
        
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-conv_dim", default='1d', type=str,
                        help="Dimension of the Convolution kernel.{1d, 2d}")
    parser.add_argument("-checkpoint", default='', type=str,
                        help="pretrained model checkpoint.")
    parser.add_argument("-hidden_size", default=128, type=int,
                        help="The hidden size of LSTM.")
    parser.add_argument("-num_layers", default=2, type=int,
                        help="Number of LSTM layers.")
    parser.add_argument("-bidirectional", default='true', type=str,
                        help="Whether to use a bidirectional LSTM.")
    parser.add_argument("-with_focus_attn", default='false', type=str,
                        help="Whether to use a focus attention mechanism.")

    parser.add_argument("-batch_size", default=128, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate", default=0.0001, type=int,
                        help="The initial learning rate for Adam.")
    parser.add_argument("-num_epochs", default=300, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("-multi_gpu", default='false', type=str,
                        help="Whether to use a multi-gpu.")

    parser.add_argument("-use_warmup", default='false', type=str,
                        help="Whether to use a warm-up.")
    parser.add_argument("-warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for.")
    parser.add_argument("-gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument("-data_dir", default='./wav_data/pretrain/RAVDESS_resample/', type=str,
                        help="The wav data directory.")    
    parser.add_argument("multi_task", default='false', type=str,
                        help="Whether to use a gender information.")

    args = parser.parse_args()

    args.bidirectional = True if(args.bidirectional == 'true') else False
    args.with_focus_attn = True if(args.with_focus_attn == 'true') else False
    n_mfcc = 40 if(args.conv_dim == '1d') else 128

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    

    sample_datas = glob.glob(os.path.join(args.data_dir, '**', '*wav'), recursive=True)
    sample_datas = sorted(sample_datas)


    np.random.seed(42)
    idx = np.random.permutation(len(sample_datas))
    train_idx = idx[:int(len(sample_datas)*0.75)]
    eval_idx = idx[int(len(sample_datas)*0.75):]

    train_samples = list(np.array(sample_datas)[train_idx])
    eval_samples = list(np.array(sample_datas)[eval_idx])


    y = np.array(list(map(lambda x: int(x.split('/')[-1].split('-')[2]) - 1, sample_datas)))
    y_train = y[train_idx]
    y_eval = y[eval_idx]


    if(args.multi_task == 'true'):
        speaker = np.array(list(map(lambda x: int(x.split('/')[-1].split('-')[-1].split('.')[0]), sample_datas)))
        y_gender = np.array(list(map(lambda x: 1 if x % 2 ==0 else 0, speaker)))

        y_g_train = y_gender[train_idx]
        y_g_eval = y_gender[eval_idx]


    X_train, y_train = convert_spectrograms(train_samples, conv_dim=args.conv_dim, sr=16000, labels=y_train)
    X_eval, y_eval = convert_spectrograms(eval_samples, conv_dim=args.conv_dim, sr=16000, labels=y_eval)

    X_train, y_train = convert_tensor(X_train, y_train)
    X_eval, y_eval = convert_tensor(X_eval, y_eval)

    y_train = y_train.long()
    y_eval = y_eval.long()


    if(args.multi_task == 'true'):
        _, y_g_train = convert_spectrograms(train_samples, conv_dim=args.conv_dim, sr=16000, labels=y_g_train)
        _, y_g_eval = convert_spectrograms(eval_samples, conv_dim=args.conv_dim, sr=16000, labels=y_g_eval)

        y_g_train = torch.tensor(y_g_train).float()
        y_g_eval = torch.tensor(y_g_eval).float()

        y_g_train = y_g_train.unsqueeze(-1)
        y_g_eval = y_g_eval.unsqueeze(-1)


    if(args.multi_task == 'true'):
        train_ds = TensorDataset(X_train, y_train, y_g_train)
        eval_ds = TensorDataset(X_eval, y_eval, y_g_eval)
    else:
        train_ds = TensorDataset(X_train, y_train)
        eval_ds = TensorDataset(X_eval, y_eval)

    train_dataloader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    eval_dataloader = DataLoader(eval_ds, batch_size=args.batch_size, num_workers=0, drop_last=True)


    model = CLDNN(conv_dim=args.conv_dim, checkpoint=args.checkpoint, hidden_size=args.hidden_size,
                  num_layers=args.num_layers, bidirectional=args.bidirectional,
                  with_focus_attn=args.with_focus_attn).to(device)

    if(args.multi_task == 'true'):
        model_g = CLDNN_G(conv_dim=args.conv_dim, checkpoint=args.checkpoint, hidden_size=args.hidden_size,
                            num_layers=args.num_layers, bidirectional=args.bidirectional,
                            with_focus_attn=args.with_focus_attn).to(device)


    if(args.multi_task == 'true'):
        loss_func = nn.CrossEntropyLoss()
        loss_func_g = nn.BCELoss()
        optimizer = optim.Adam(list(model.parameters()) + list(model_g.parameters()), lr=args.learning_rate)
    else:
        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)


    if(args.use_warmup == 'true'):
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_epochs
        opt_scheduler = WarmupLinearSchedule(optimizer, warmup_steps=t_total * args.warmup_proportion, t_total=t_total)


    train(train_dataloader, eval_dataloader, args.num_epochs)

    model.eval()
    if(args.multi_task == 'true'):
        model_g.eval()

    correct = 0
    n = 0
    for i in range(len(eval_samples)):
        try:
            X_new = preprocessing(eval_samples[i], method='mfcc', sr=16000, n_mfcc=n_mfcc)
            X_new = convert_tensor(X_new).to(device)
            y_new = model(X_new)
            y_new = torch.argmax(nn.Softmax(dim=-1)(torch.mean(y_new, dim=0)))
            #y_new = sorted(dict(collections.Counter(torch.argmax(nn.Softmax(dim=-1)(y_new), dim=1).cpu().numpy()))
            #               .items(), key=(lambda x: x[1]), reverse=True)[0][0]
            y_new = 1 if (y_new.item() == y[eval_idx][i].item()) else 0
            correct += y_new
            n += 1
        except:
            pass

    acc = correct / n
    print('Test accuray:', round(acc, 5))

    
    
if __name__ == "__main__":
    main()