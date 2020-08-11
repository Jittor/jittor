import jittor as jt
from jittor import nn
from jittor import Module
from jittor import init
from jittor.contrib import concat, argmax_pool
import math

class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, baseWidth=26, scale = 4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth/64.0)))
        self.conv1 = nn.Conv(inplanes, width*scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm(width*scale)
        assert scale > 1, 'Res2Net degenerates to ResNet when scales = 1.'
        if scale == 1:
          self.nums = 1
        else:
          self.nums = scale -1
        if stype == 'stage':
            self.pool = nn.Pool(kernel_size=3, stride = stride, padding=1, op='mean')
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(self.nums):
          self.convs.append(nn.Conv(width, width, kernel_size=3, stride = stride, dilation=dilation, padding=dilation, bias=False))
          self.bns.append(nn.BatchNorm(width))

        self.conv3 = nn.Conv(width*scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm(planes * self.expansion)

        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width  = width
        self.stride = stride
        self.dilation = dilation

    def execute(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        spx = out
        
        outs = []
        for i in range(self.nums):
            if i==0 or self.stype=='stage':
                sp = spx[:, i*self.width: (i+1)*self.width]
            else:
                sp = sp + spx[:, i*self.width: (i+1)*self.width]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            outs.append(sp)
        if self.stype=='normal' or self.stride==1:
            outs.append(spx[:, self.nums*self.width: (self.nums+1)*self.width])
        elif self.stype=='stage':
            outs.append(self.pool(spx[:, self.nums*self.width: (self.nums+1)*self.width]))
        out = concat(outs, 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Res2Net(Module):
    def __init__(self, block, layers, output_stride, baseWidth = 26, scale = 4):
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.inplanes = 64
        blocks = [1, 2, 4]
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError

        # Modules
        self.conv1 = nn.Sequential(
            nn.Conv(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm(32),
            nn.ReLU(),
            nn.Conv(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm(32),
            nn.ReLU(),
            nn.Conv(32, 64, 3, 1, 1, bias=False)
        )
        self.bn1 = nn.BatchNorm(64)
        self.relu = nn.ReLU()
        # self.maxpool = nn.Pool(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2])
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], dilation=dilations[3])


    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Pool(kernel_size=stride, stride=stride, 
                    ceil_mode=True, op='mean'),
                nn.Conv(self.inplanes, planes * block.expansion, 
                    kernel_size=1, stride=1, bias=False),
                nn.BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample,
                        stype='stage', baseWidth = self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, baseWidth = self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Pool(kernel_size=stride, stride=stride, 
                    ceil_mode=True, op='mean'),
                nn.Conv(self.inplanes, planes * block.expansion, 
                    kernel_size=1, stride=1, bias=False),
                nn.BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=blocks[0]*dilation,
                            downsample=downsample, stype='stage', baseWidth = self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1,
                                dilation=blocks[i]*dilation, baseWidth = self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)

    def execute(self, input):

        x = self.conv1(input)
        x = self.bn1(x)

        x = self.relu(x)
        x = argmax_pool(x, 2, 2)
        x = self.layer1(x)
        low_level_feat = x
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.layer4(x)
        return x, low_level_feat
    
def res2net50(output_stride):
    model = Res2Net(Bottle2neck, [3,4,6,3], output_stride)
    return model

def res2net101(output_stride):
    model = Res2Net(Bottle2neck, [3,4,23,3], output_stride)
    return model
