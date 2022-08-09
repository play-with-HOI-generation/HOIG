import torch
import torch.nn as nn
import torch.nn.functional as F
from thirdparty.block_extractor.block_extractor import BlockExtractor
from thirdparty.local_attn_reshape.local_attn_reshape import LocalAttnReshape


class ExtractorAttn(nn.Module):
    def __init__(self, feature_nc, kernel_size=4, nonlinearity=nn.LeakyReLU(), softmax=None):
        super(ExtractorAttn, self).__init__()
        self.kernel_size=kernel_size
        hidden_nc = 128
        softmax = nonlinearity if softmax is None else nn.Softmax(dim=1)

        self.extractor = BlockExtractor(kernel_size=kernel_size)
        self.reshape = LocalAttnReshape()
        self.fully_connect_layer = nn.Sequential(
                nn.Conv2d(2*feature_nc, hidden_nc, kernel_size=kernel_size, stride=kernel_size, padding=0),
                nonlinearity,
                nn.Conv2d(hidden_nc, kernel_size*kernel_size, kernel_size=1, stride=1, padding=0),
                softmax,)

    def forward(self, source, target, flow_field):
        block_source = self.extractor(source, flow_field)
        block_target = self.extractor(target, torch.zeros_like(flow_field))
        attn_param = self.fully_connect_layer(torch.cat((block_target, block_source), 1))
        attn_param = self.reshape(attn_param, self.kernel_size)
        result = torch.nn.functional.avg_pool2d(attn_param*block_source, self.kernel_size, self.kernel_size)
        return result

    def hook_attn_param(self, source, target, flow_field):
        block_source = self.extractor(source, flow_field)
        block_target = self.extractor(target, torch.zeros_like(flow_field))
        attn_param_ = self.fully_connect_layer(torch.cat((block_target, block_source), 1))
        attn_param = self.reshape(attn_param_, self.kernel_size)
        result = torch.nn.functional.avg_pool2d(attn_param*block_source, self.kernel_size, self.kernel_size)
        return attn_param_, result
