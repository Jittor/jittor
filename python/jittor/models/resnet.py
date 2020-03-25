# ***************************************************************
# Copyright (c) 2020 Jittor. Authors: 
#     Guowei Yang <471184555@qq.com>
#     Wenyang Zhou <576825820@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import jittor as jt
from jittor import nn
from jittor import Module

@jt.var_scope('basic_block')
def basic_block(x, is_train, in_planes, out_planes, stride = 1):
    identity = x
    x = nn.conv(x, in_planes, out_planes, 3, 1, stride)
    x = nn.batch_norm(x, is_train)
    x = nn.relu(x)
    x = nn.conv(x, out_planes, out_planes, 3, 1)
    x = nn.batch_norm(x, is_train)
    if in_planes!=out_planes:
        identity = nn.conv(identity, in_planes, out_planes, 1, 0, stride)
        identity = nn.batch_norm(identity, is_train)
    x = x+identity
    x = nn.relu(x)
    return x

@jt.var_scope('make_layer')
def make_layer(x, is_train, out_planes, blocks, layer_in_planes, stride = 1):
    x = basic_block(x, is_train, layer_in_planes, out_planes, stride)
    layer_in_planes = out_planes

    for i in range(1, blocks):
        x = basic_block(x, is_train, layer_in_planes, out_planes)
    return x, layer_in_planes

@jt.var_scope('bottleneck_block')
def bottleneck_block(x, is_train, in_planes, out_planes, stride = 1):
    expansion = 4 
    width = out_planes
    identity = x

    x = nn.conv(x, in_planes, width, 1, 0)
    x = nn.batch_norm(x, is_train)
    x = nn.relu(x)

    x = nn.conv(x, width, width, 3, 1, stride)
    x = nn.batch_norm(x, is_train)
    x = nn.relu(x)

    x = nn.conv(x, width, out_planes * expansion, 1, 0)
    x = nn.batch_norm(x, is_train)

    if in_planes != out_planes * expansion:
        identity = nn.conv(identity, in_planes, out_planes * expansion, 1, 0, stride)
        identity = nn.batch_norm(identity, is_train)
    
    x = x+identity
    x = nn.relu(x)
    return x

@jt.var_scope('make_layer_bottleneck')
def make_layer_bottleneck(x, is_train, out_planes, blocks, layer_in_planes, stride = 1):
    expansion = 4
    x = bottleneck_block(x, is_train, layer_in_planes, out_planes, stride)
    layer_in_planes = out_planes * expansion
    for i in range(1, blocks):
        x = bottleneck_block(x, is_train, layer_in_planes, out_planes)
    return x, layer_in_planes

@jt.var_scope('resnet')
def resnet(x, is_train, block, layers, num_classes = 1000):
    layer_in_planes = 64
    x = nn.conv(x, 3, layer_in_planes, 7, 3, 2)
    x = nn.batch_norm(x, is_train)
    x = nn.relu(x)
    x = nn.pool(x, 3, "maximum", 1, 2)
    x, layer_in_planes = block(x, is_train, 64, layers[0], layer_in_planes)
    x, layer_in_planes = block(x, is_train, 128, layers[1], layer_in_planes, 2)
    x, layer_in_planes = block(x, is_train, 256, layers[2], layer_in_planes, 2)
    x, layer_in_planes = block(x, is_train, 512, layers[3], layer_in_planes, 2)

    x = x.reindex_reduce("add", [x.shape[0],x.shape[1]], ["i0","i1"])/x.shape[2]/x.shape[3]
    x = nn.linear(x, num_classes)

    return x

@jt.var_scope('resnet18', unique=True)
def resnet18(x, is_train):
    return resnet(x, is_train, make_layer, [2, 2, 2, 2])

@jt.var_scope('resnet34', unique=True)
def resnet34(x, is_train):
    return resnet(x, is_train, make_layer, [3, 4, 6, 3])

@jt.var_scope('resnet50', unique=True)
def resnet50(x, is_train):
    return resnet(x, is_train, make_layer_bottleneck, [3, 4, 6, 3])

@jt.var_scope('resnet101', unique=True)
def resnet101(x, is_train):
    return resnet(x, is_train, make_layer_bottleneck, [3, 4, 23, 3])

@jt.var_scope('resnet152', unique=True)
def resnet152(x, is_train):
    return resnet(x, is_train, make_layer_bottleneck, [3, 8, 36, 3])

class BasicBlock(Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, base_width=64):
        self.conv1 = nn.Conv(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm(planes)
        self.relu = nn.Relu()
        self.conv2 = nn.Conv(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm(planes)
        self.downsample = downsample
        self.stride = stride
        self.planes = planes

    def execute(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        return out

class Bottleneck(Module):
    expansion = 4
 
    def __init__(self, inplanes, planes, stride=1, downsample=None, base_width=64):
        width = int((planes * (base_width / 64.0)))
        self.conv1 = nn.Conv(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm(width)
        self.conv2 = nn.Conv(width, width, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm(width)
        self.conv3 = nn.Conv(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm(planes * self.expansion)
        self.relu = nn.Relu()
        self.downsample = downsample
        self.stride = stride
 
    def execute(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
 
        out = self.conv3(out)
        out = self.bn3(out)
 
        if self.downsample is not None:
            residual = self.downsample(x)
 
        out += residual
        out = self.relu(out)
        return out

class ResNet(Module):
    def __init__(self, block, layers, num_classes=1000, width_per_group=64):
        self.inplanes = 64
        self.base_width = width_per_group
        self.conv1 = nn.Conv(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm(64)
        self.relu = nn.Relu()
        self.maxpool = nn.Pool(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.Pool(7, stride=1, op="mean")
        self.fc = nn.Linear(512 * block.expansion, num_classes)
 
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm(planes * block.expansion),
            )
 
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, base_width=self.base_width))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, base_width=self.base_width))
 
        return nn.Sequential(*layers)
 
    def execute(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = jt.reshape(x, [x.shape[0],-1])
        x = self.fc(x)
 
        return x

def Resnet18():
    model = ResNet(BasicBlock, [2,2,2,2])
    return model

def Resnet34():
    model = ResNet(BasicBlock, [3,4,6,3])
    return model

def Resnet50():
    model = ResNet(Bottleneck, [3,4,6,3])
    return model

def Resnet101():
    model = ResNet(Bottleneck, [3,4,23,3])
    return model

def Resnet152():
    model = ResNet(Bottleneck, [3,8,36,3])
    return model

def wide_resnet50_2():
    model = ResNet(Bottleneck, [3, 4, 6, 3], width_per_group=128)
    return model

def wide_resnet101_2():
    model = ResNet(Bottleneck, [3, 4, 23, 3], width_per_group=128)
    return model