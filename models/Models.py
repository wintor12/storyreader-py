import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import dataset


class Fc(nn.Module):
    def __init__(self, feature, opt):
        super(Fc, self).__init__()
        self.opt = opt
        self.fc1 = nn.Linear(feature, opt.hidden1)
        self.fc2 = nn.Linear(opt.hidden1, opt.hidden2)
        self.fc3 = nn.Linear(opt.hidden2, 1)
        self.dropout = nn.Dropout(opt.dropout)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        if self.opt.text:
            x = self.fc3(x)
        else:
            x = F.relu(self.fc3(x))
        return x


class RegionalCNN(nn.Module):
    def __init__(self, opt):
        super(RegionalCNN, self).__init__()
        self.opt = opt
        self.conv1_out = 32
        self.conv2_out = 32
        self.conv3_out = 16
        self.conv1_kernel = (3, 5)
        self.conv2_kernel = (2, 3)
        self.conv3_kernel = (1, 3)
        self.conv1_stride = (1, 3)
        self.conv2_stride = (1, 2)
        self.conv3_stride = (1, 1)
        self.conv1 = nn.Conv2d(1, self.conv1_out, self.conv1_kernel, self.conv1_stride)
        self.conv2 = nn.Conv2d(self.conv1_out, self.conv2_out,
                               self.conv2_kernel, self.conv2_stride)
        self.conv3 = nn.Conv2d(self.conv2_out, self.conv3_out,
                               self.conv3_kernel, self.conv3_stride)
        self.dropout = nn.Dropout(self.opt.dropout)
        # self.batch_norm1 = nn.BatchNorm2d(self.conv1_out)
        # self.batch_norm2 = nn.BatchNorm2d(self.conv2_out)
        # self.batch_norm3 = nn.BatchNorm2d(self.conv3_out)

    def forward(self, input):
        batch_size, _, emb_size = input.size()
        region_input = input.view(batch_size, -1, 1,
                                  self.opt.region_words, emb_size)
        outputs = []
        for region in region_input.split(1, dim=1):
            region = region.squeeze(1)
            # x = F.relu(self.batch_norm1(self.conv1(region)))
            x = F.relu(self.conv1(region))
            x = F.max_pool2d(x, 2)
            # x = F.relu(self.batch_norm2(self.conv2(x)))
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            # x = F.relu(self.batch_norm3(self.conv3(x)))
            x = F.relu(self.conv3(x))
            x = F.max_pool2d(x, 2)
            x = x.view(batch_size, -1)
            x = self.dropout(x)
            outputs.append(x)

        outputs = torch.stack(outputs, 1)
        return outputs


class RegionalReader(nn.Module):

    def __init__(self, vocab, embed_size, s_rcnn, q_rcnn, fc, opt):
        super(RegionalReader, self).__init__()
        self.embed = nn.Embedding(len(vocab), embed_size,
                                  padding_idx=vocab.stoi[dataset.PAD_WORD])
        self.s_rcnn = s_rcnn
        self.q_rcnn = q_rcnn
        self.r_fc = nn.Linear(320, opt.r_emb)
        self.fc = fc
        self.opt = opt
        self.dropout = nn.Dropout(self.opt.dropout)

    def load_pretrained_vectors(self, wv):
        if wv is not None:
            self.embed.weight.data.copy_(wv)

    def compute_regional_emb(self, batch):
        """
        Args:
          input: batch
        returns:
          batch x regions x 10
        """
        # preprocess text and question
        if self.opt.region_nums == 0:
            length = batch.src[0].size(0)
            length = length - length % self.opt.region_words
            text = batch.src[0][:length]
        else:
            text = batch.src[0]
        question = batch.question[0][:36]
        s_embs = self.embed(text)
        q_embs = self.embed(question)

        # use regional cnn to get region embedding
        s_r_emb = self.s_rcnn(s_embs.transpose(0, 1).contiguous())
        # batch x regions x 320
        q_r_emb = self.q_rcnn(q_embs.transpose(0, 1).contiguous())
        r_emb = torch.cat([q_r_emb, s_r_emb], 1)
        # batch x (sregions + qregions) x 320

        return r_emb

    def forward(self, batch):
        r_emb = self.compute_regional_emb(batch)
        outputs = []
        for emb_t in r_emb.split(1, dim=1):
            emb_t = emb_t.squeeze(1)
            x = self.r_fc(emb_t)
            outputs.append(x)
        r_emb = torch.stack(outputs, 1)  # batch x regions x r_emb
        if self.opt.region_nums == 0:
            # r_emb = r_emb.max(dim=1)[0]
            r_emb = r_emb.mean(1)
        r_emb = r_emb.view(r_emb.size(0), -1)
        # r_emb = self.dropout(r_emb)
        if not self.opt.text:
            fc_input = torch.cat([r_emb, batch.feature], 1)
        else:
            fc_input = r_emb
        output = self.fc(fc_input)
        return output


class SequentialReader(RegionalReader):
    def __init__(self, vocab, embed_size, s_rcnn, q_rcnn, fc, opt):
        super(SequentialReader, self).__init__(vocab, embed_size,
                                               s_rcnn, q_rcnn, fc, opt)
        self.rnn_cell = nn.LSTMCell(320, opt.r_emb)
        self.r_w = nn.Linear(320, opt.r_emb)
        self.h_w = nn.Linear(opt.r_emb, opt.r_emb)

    def forward(self, batch):
        r_emb = self.compute_regional_emb(batch)
        batch_size = r_emb.size(0)
        weight = next(self.parameters()).data
        h, c = (Variable(weight.new(batch_size, self.opt.r_emb).normal_(0, 1)),
                Variable(weight.new(batch_size, self.opt.r_emb).normal_(0, 1)))

        outputs = []
        for emb_t in r_emb.split(1, dim=1):
            emb_t = emb_t.squeeze(1)
            h, c = self.rnn_cell(emb_t, (h, c))
            gate = F.sigmoid(self.r_w(emb_t) + self.h_w(h))
            outputs.append(torch.mul(h, gate))
        r_emb = torch.stack(outputs, 1)
        if self.opt.region_nums == 0:
            # r_emb = r_emb.max(dim=1)[0]
            r_emb = r_emb.mean(dim=1)
        r_emb = r_emb.view(r_emb.size(0), -1)
        # r_emb = self.dropout(r_emb)
        if not self.opt.text:
            fc_input = torch.cat([r_emb, batch.feature], 1)
        else:
            fc_input = r_emb
        output = self.fc(fc_input)
        return output


class HolisticReader(RegionalReader):
    def __init__(self, vocab, embed_size, s_rcnn, q_rcnn, fc, opt):
        super(HolisticReader, self).__init__(vocab, embed_size,
                                             s_rcnn, q_rcnn, fc, opt)
        self.rnn = nn.LSTM(320, opt.r_emb, batch_first=True)
        self.r_w = nn.Linear(320, opt.r_emb)
        self.h_w = nn.Linear(opt.r_emb, opt.r_emb)

    def forward(self, batch):
        r_emb = self.compute_regional_emb(batch)
        batch_size = r_emb.size(0)

        weight = next(self.parameters()).data
        h, c = (Variable(weight.new(1, batch_size, self.opt.r_emb).normal_(0, 1)),
                Variable(weight.new(1, batch_size, self.opt.r_emb).normal_(0, 1)))

        h_output, _ = self.rnn(r_emb, (h, c))
        # h_output = h_output.max(dim=1)[0]
        # r_output = r_emb.max(dim=1)[0]
        h_output = h_output.mean(1)
        r_output = r_emb.mean(1)
        gate = F.sigmoid(self.r_w(r_output) + self.h_w(h_output))
        r_emb = torch.mul(gate, h_output)
        r_emb = r_emb.view(r_emb.size(0), -1)
        # r_emb = self.dropout(r_emb)
        if not self.opt.text:
            fc_input = torch.cat([r_emb, batch.feature], 1)
        else:
            fc_input = r_emb
        output = self.fc(fc_input)
        return output
