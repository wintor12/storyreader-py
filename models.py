import torch.nn as nn
import torch.nn.functional as F
import torch


class Fc(nn.Module):
    def __init__(self, feature, hidden1, hidden2, dropout):
        super(Fc, self).__init__()
        self.fc1 = nn.Linear(feature, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.dropout = nn.Dropout(dropout)

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
        self.fc = nn.Linear(320, 10)
        

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
            x = self.fc(x)
            outputs.append(x)

        outputs = torch.stack(outputs, 1)
        return outputs


class RegionalReader(nn.Module):

    def __init__(self, vocab_size, embed_size, s_rcnn, q_rcnn):
        super(RegionalReader, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.s_rcnn = s_rcnn
        self.q_rcnn = q_rcnn


    def forward(self, story, question):
        """
        Args:
        input: src_len x batch
        returns:
        """
        # src_len x batch x emb_size
        s_embs = self.embed(story)
        q_embs = self.embed(question[:36])
        print('q_embs', q_embs.size())

        # use regional cnn to get region embedding
        s_r_emb = self.s_rcnn(s_embs.transpose(0, 1).contiguous()) # batch x regions x 10
        print(s_r_emb.size())
        q_r_emb = self.q_rcnn(q_embs.transpose(0, 1).contiguous())
        print('q_r', q_r_emb.size())
        r_emb = torch.cat([q_r_emb, s_r_emb], 1)
        print(r_emb.size())
        return r_emb
