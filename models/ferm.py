from models import common

import torch.nn as nn
import torch


def make_model(args):
    return FERM(args)


## Residual Block (RB)
class RB(nn.Module):
    def __init__(
            self, conv, n_feat, bn=False, act=nn.ReLU(True)):
        super(RB, self).__init__()

        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


## Residual Group (RG)
class RG(nn.Module):
    def __init__(self, conv, n_feat, bn, act, n_resblocks):
        super(RG, self).__init__()

        modules_body = [
            RB(conv, n_feat, bn=bn, act=act) for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat))

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class FERM(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(FERM, self).__init__()

        in_channel = 3
        out_channel = 3
        n_feats = 64
        act = nn.ReLU(True)
        n_resblocks = 20
        n_resgroups = 10
        bn = False

        # RGB mean for DIV2K
        self.sub_mean = common.MeanShift(1)
        self.add_mean = common.MeanShift(1, sign=1)

        head = [conv(in_channel, n_feats)]

        body = [
            RG(conv, n_feats, bn=bn, act=act, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]
        body.append(conv(n_feats, n_feats))

        tail = [
            conv(n_feats, n_feats),
            conv(n_feats, out_channel)]

        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)

    def forward(self, x):
        x = self.sub_mean(x)
        global_res = x

        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)

        x += global_res
        x = self.add_mean(x)

        return x

