import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable


class Fc(nn.Module):
    def __init__(self, feature, opt):
        super(Fc, self).__init__()
        self.fc1 = nn.Linear(feature, opt.hidden1)
        self.fc2 = nn.Linear(opt.hidden1, opt.hidden2)
        self.fc3 = nn.Linear(opt.hidden2, 1)
        self.dropout = nn.Dropout(opt.dropout)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        return x


class RegionalCNN(nn.Module):
    def __init__(self, opt, regions):
        super(RegionalCNN, self).__init__()
        self.opt = opt
        self.regions = regions
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

    def forward(self, input):
        batch_size, _, emb_size = input.size()
        region_input = input.view(batch_size, self.regions, 1,
                                  self.opt.region_words, emb_size)

        outputs = []
        for region in region_input.split(1, dim=1):
            region = region.squeeze(1)
            x = F.relu(self.conv1(region))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv3(x))
            x = F.max_pool2d(x, 2)
            x = x.view(batch_size, -1)
            x = self.dropout(x)
            outputs.append(x)

        outputs = torch.stack(outputs, 1)
        return outputs


class RegionalReader(nn.Module):

    def __init__(self, vocab_size, embed_size, s_rcnn, q_rcnn, fc):
        super(RegionalReader, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.s_rcnn = s_rcnn
        self.q_rcnn = q_rcnn
        self.r_fc = nn.Linear(320, 10)
        self.fc = fc

    def load_pretrained_vectors(self, wv):
        if wv is not None:
            self.embed.weight.data.copy_(wv)

    def forward(self, batch):
        """
        Args:
          input: batch
        returns:
          log upvote prediction
        """
        # src_len x batch x emb_size
        s_embs = self.embed(batch.src[0])
        q_embs = self.embed(batch.question[0][:36])

        # use regional cnn to get region embedding
        s_r_emb = self.s_rcnn(s_embs.transpose(0, 1).contiguous())
        # batch x regions x 320
        q_r_emb = self.q_rcnn(q_embs.transpose(0, 1).contiguous())
        r_emb = torch.cat([q_r_emb, s_r_emb], 1)
        # batch x (sregions + qregions) x 320

        # r_emb = q_r_emb
        outputs = []
        for emb_t in r_emb.split(1, dim=1):
            emb_t = emb_t.squeeze(1)
            x = self.r_fc(emb_t)
            outputs.append(x)
        r_emb = torch.stack(outputs, 1)

        r_emb = r_emb.view(r_emb.size(0), -1)
        fc_input = torch.cat([r_emb, batch.feature], 1)
        output = self.fc(fc_input)
        return output


class SequentialReader(RegionalReader):
    def __init__(self, vocab_size, embed_size, s_rcnn, q_rcnn, fc):
        super(SequentialReader, self).__init__(vocab_size,
                                               embed_size, s_rcnn, q_rcnn, fc)
        self.rnn_cell = nn.LSTMCell(320, 10)
        self.h_fc = nn.Linear(10, 10)

    def forward(self, batch):
        """
        Args:
          input: batch
        returns:
          log upvote prediction
        """
        # src_len x batch x emb_size
        s_embs = self.embed(batch.src[0])
        q_embs = self.embed(batch.question[0][:36])

        # use regional cnn to get region embedding
        s_r_emb = self.s_rcnn(s_embs.transpose(0, 1).contiguous())
        # batch x regions x 320
        q_r_emb = self.q_rcnn(q_embs.transpose(0, 1).contiguous())
        r_emb = torch.cat([q_r_emb, s_r_emb], 1)
        # batch x (sregions + qregions) x 320

        batch_size = r_emb.size(0)
        weight = next(self.parameters()).data
        h, c = (Variable(weight.new(batch_size, 10).normal_(0, 1)),
                Variable(weight.new(batch_size, 10).normal_(0, 1)))

        outputs = []
        for emb_t in r_emb.split(1, dim=1):
            emb_t = emb_t.squeeze(1)
            h, c = self.rnn_cell(emb_t, (h, c))
            h = F.sigmoid(self.r_fc(emb_t) + self.h_fc(h))
            outputs.append(h)
        r_emb = torch.stack(outputs, 1)

        r_emb = r_emb.view(r_emb.size(0), -1)
        fc_input = torch.cat([r_emb, batch.feature], 1)
        output = self.fc(fc_input)
        return output


class HolisticReader(RegionalReader):
    def __init__(self, vocab_size, embed_size, s_rcnn, q_rcnn, fc):
        super(HolisticReader, self).__init__(vocab_size,
                                             embed_size, s_rcnn, q_rcnn, fc)
        self.rnn = nn.LSTM(320, 10, batch_first=True)
        self.h_fc = nn.Linear(10, 10)

    def forward(self, batch):
        """
        Args:
          input: batch
        returns:
          log upvote prediction
        """
        # src_len x batch x emb_size
        s_embs = self.embed(batch.src[0])
        q_embs = self.embed(batch.question[0][:36])

        # use regional cnn to get region embedding
        s_r_emb = self.s_rcnn(s_embs.transpose(0, 1).contiguous())
        # batch x regions x 320
        q_r_emb = self.q_rcnn(q_embs.transpose(0, 1).contiguous())
        r_emb = torch.cat([q_r_emb, s_r_emb], 1)
        # batch x (sregions + qregions) x 320

        batch_size = r_emb.size(0)
        weight = next(self.parameters()).data
        h, c = (Variable(weight.new(1, batch_size, 10).normal_(0, 1)),
                Variable(weight.new(1, batch_size, 10).normal_(0, 1)))

        r_ouput = self.rnn(r_emb, (h, c))
        print(r_ouput[0].size())
        weight = next(self.parameters()).data
        W = Variable(weight.new(r_emb.size(1), r_emb.size(1)).uniform_(-1, 1),
                     requires_grad=True)
        # W = Variable(W.unsqueeze(-1).expand_as(r_emb))
        print(W)
        print(W.requires_grad)
        U = Variable(weight.new(1, batch_size, 10))

        import sys
        sys.exit()
        r_emb = r_emb.view(r_emb.size(0), -1)
        fc_input = torch.cat([r_emb, batch.feature], 1)
        output = self.fc(fc_input)
        return output
