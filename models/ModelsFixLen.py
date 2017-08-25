import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import dataset


class RegionalReader(nn.Module):

    def __init__(self, vocab, embed_size, s_rcnn, q_rcnn, fc, opt):
        super(RegionalReader, self).__init__()
        self.embed = nn.Embedding(len(vocab), embed_size,
                                  padding_idx=vocab.stoi[dataset.PAD_WORD])
        self.s_rcnn = s_rcnn
        self.q_rcnn = q_rcnn
        self.r_fc = nn.Linear(320, 10)
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
        # src_len x batch x emb_size
        s_embs = self.embed(batch.src[0])
        q_embs = self.embed(batch.question[0][:36])

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
        r_emb = torch.stack(outputs, 1)
        r_emb = self.dropout(r_emb.view(r_emb.size(0), -1))
        if self.opt.text:
            fc_input = torch.cat([r_emb, batch.feature], 1)
        else:
            fc_input = r_emb
        output = self.fc(fc_input)
        return output


class SequentialReader(RegionalReader):
    def __init__(self, vocab, embed_size, s_rcnn, q_rcnn, fc, opt):
        super(SequentialReader, self).__init__(vocab, embed_size,
                                               s_rcnn, q_rcnn, fc, opt)
        self.rnn_cell = nn.LSTMCell(320, 10)
        self.r_w = nn.Linear(320, 10)
        self.h_w = nn.Linear(10, 10)

    def forward(self, batch):
        r_emb = self.compute_regional_emb(batch)
        batch_size = r_emb.size(0)
        weight = next(self.parameters()).data
        h, c = (Variable(weight.new(batch_size, 10).normal_(0, 1)),
                Variable(weight.new(batch_size, 10).normal_(0, 1)))

        outputs = []
        for emb_t in r_emb.split(1, dim=1):
            emb_t = emb_t.squeeze(1)
            h, c = self.rnn_cell(emb_t, (h, c))
            gate = F.sigmoid(self.r_w(emb_t) + self.h_w(h))
            outputs.append(torch.mul(h, gate))
        r_emb = torch.stack(outputs, 1)

        r_emb = self.dropout(r_emb.view(r_emb.size(0), -1))
        if self.opt.text:
            fc_input = torch.cat([r_emb, batch.feature], 1)
        else:
            fc_input = r_emb
        output = self.fc(fc_input)
        return output


class HolisticReader(RegionalReader):
    def __init__(self, vocab, embed_size, s_rcnn, q_rcnn, fc, opt):
        super(HolisticReader, self).__init__(vocab, embed_size,
                                             s_rcnn, q_rcnn, fc, opt)
        self.rnn = nn.LSTM(320, 10, batch_first=True)
        self.h_conv = nn.Conv2d(1, 11, (11, 1))
        self.r_conv = nn.Conv2d(1, 11, (11, 1))

    def forward(self, batch):
        r_emb = self.compute_regional_emb(batch)
        batch_size = r_emb.size(0)

        weight = next(self.parameters()).data
        h, c = (Variable(weight.new(1, batch_size, 10).normal_(0, 1)),
                Variable(weight.new(1, batch_size, 10).normal_(0, 1)))

        r_output, _ = self.rnn(r_emb, (h, c))

        gate_h_input = self.h_conv(r_output.unsqueeze(1)).squeeze(2)
        gate_r_input = self.r_conv(self.r_fc(r_emb).unsqueeze(1)).squeeze(2)
        gate = F.sigmoid(gate_r_input + gate_h_input)  # batch x regions x 10
        r_emb = torch.mul(gate, r_output)
        r_emb = self.dropout(r_emb.view(r_emb.size(0), -1))
        if self.opt.text:
            fc_input = torch.cat([r_emb, batch.feature], 1)
        else:
            fc_input = r_emb
        output = self.fc(fc_input)
        return output
