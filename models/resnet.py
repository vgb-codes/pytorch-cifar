'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def create_squared_sparsity_matrix(planes, m, p):
    M = torch.rand(planes*m, planes*m, 1,1) < p
    a = torch.eye(planes*m).reshape(planes*m, planes*m,1,1)
    M = nn.Parameter((M+a)>0, requires_grad=False)
    del a
    return M

def create_rect_sparsity_matrix(in_planes, planes, m, p):
    M = torch.rand(planes*m, in_planes*m, 1, 1) < p
    a = np.zeros((planes*m, in_planes*m, 1, 1))
    i,j,_,_ = np.indices(a.shape)
    
    for z in range(in_planes*m):
        a[i==j+z] = 1

    M = nn.Parameter((M+a)>0, requires_grad=False)
    del a
    return M

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, m=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes*m)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes*m)

        # Code modified here: Converting to functional code
        
        # Code modified here: Adding masking
        self.m = m
        self.p = 1/(self.m**2)
        self.W1 = nn.Parameter(torch.zeros(size=(planes*self.m, in_planes*self.m, 3, 3)))
        self.W2 = nn.Parameter(torch.zeros(size=(planes*self.m, planes*self.m, 3, 3)))
        nn.init.kaiming_normal_(self.W1)
        nn.init.kaiming_normal_(self.W2)


        # Simple Sparsity
        self.M1 = nn.Parameter(torch.rand(planes*self.m, in_planes*self.m, 1, 1) < self.p, requires_grad=False)
        self.M2 = nn.Parameter(torch.rand(planes*self.m, planes*self.m, 1, 1) < self.p, requires_grad=False)


        # Squared or Rectangular Sparsity for M1 (in_channels != out_channels)
        if in_planes != planes:
            self.M1 = create_rect_sparsity_matrix(in_planes, planes, self.m, self.p)
        else:
            self.M1 = create_squared_sparsity_matrix(planes, self.m, self.p)

        # Squared Sparsity for M2 (in_channels == out_channels)
        self.M2 = create_squared_sparsity_matrix(planes, self.m, self.p)


        self.stride = stride
        self.shortcut = nn.Sequential()
        self.planes=planes

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes*self.m, self.expansion*planes*self.m,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes*self.m)
            )


    def forward(self, x):
        out = F.relu(self.bn1(F.conv2d(x, self.W1*self.M1, stride=self.stride, padding=1)))
        out = self.bn2(F.conv2d(out, self.W2*self.M2, stride=1, padding=1))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, m=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes*m)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes*m)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes*m)

        # Code modified here: Adding Masking
        self.m = m
        self.p = 1/(self.m**2)

        self.M1 = nn.Parameter(torch.rand(planes*self.m, in_planes*self.m, 1, 1) < self.p, requires_grad=False)
        self.M2 = nn.Parameter(torch.rand(planes*self.m, planes*self.m, 3, 3) < self.p, requires_grad=False)
        self.M3 = nn.Parameter(torch.rand(self.expansion*planes*self.m, planes*self.m,1, 1) < self.p, requires_grad=False)


        # Changes made: converting to functional code
        self.W1 = nn.Parameter(torch.zeros(size=(planes*self.m, in_planes*self.m, 1, 1)))
        self.W2 = nn.Parameter(torch.zeros(size=(planes*self.m, planes*self.m, 3, 3)))
        self.W3 = nn.Parameter(torch.zeros(size=(self.expansion*planes*self.m, planes*self.m, 1, 1)))
        nn.init.kaiming_normal_(self.W1)
        nn.init.kaiming_normal_(self.W2)
        nn.init.kaiming_normal_(self.W3)
        self.stride = stride

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes*self.m, self.expansion*planes*self.m,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes*self.m)
            )
        
        if in_planes != planes:
            self.M1 = create_rect_sparsity_matrix(in_planes, planes, self.m, self.p)
        else:
            self.M1 = create_squared_sparsity_matrix(planes, self.m, self.p)

        self.M2 = create_squared_sparsity_matrix(planes, self.m, self.p)
        
        if planes != self.expansion*planes:
            self.M3 = create_rect_sparsity_matrix(planes, self.expansion*planes, self.m, self.p)
        else:
            self.M3 = create_squared_sparsity_matrix(planes, self.m, self.p)
        
        

    def forward(self, x):
        # Changes made: converting to functional code
        out = F.relu(self.bn1(F.conv2d(x, self.W1*self.M1)))
        out = F.relu(self.bn2(F.conv2d(out, self.W2*self.M2, padding=1, stride=self.stride)))
        out = self.bn3(F.conv2d(out, self.W3*self.M3))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, m=1):
        super(ResNet, self).__init__()
        #self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               #stride=1, padding=1, bias=False)

        # Change made here: Converting conv1 to functional
        self.in_planes = 64
        self.W1 = nn.Parameter(torch.zeros(size=(64*m,3,3,3)))
        nn.init.kaiming_normal_(self.W1)
    

        self.bn1 = nn.BatchNorm2d(64*m)
        self.layer1 = self._make_layer(block, m, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, m, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, m, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, m, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion*m, num_classes)

    def _make_layer(self, block, m, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, m))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(F.conv2d(x, self.W1, stride=1, padding=1)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2], m=1)


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3], m=1)


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3], m=4)


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
