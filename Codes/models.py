import torch
import torch.nn as nn 
from collections import namedtuple
from torch import Tensor
from typing import Optional, Tuple, List, Callable, Any
import torch.nn.functional as F
import torchvision 
from torchvision.models import resnet18
from torchvision.models import resnet34
from torchvision.models import resnet50
from torchvision.models import resnet101
from torchvision.models import resnet152
from torchvision.models import googlenet
from torchvision import models
import pdb
import math
#from torchsummary import summary

#import resnet18
#import resnet34
#import resnet50
#import resnet101
#import resnet152
#import googlenet






class GoogLeNet(nn.Module):
    """
    
    """

    __constants__ = ['aux_logits', 'transform_input']

    def __init__(
        self,
        num_classes: int = 1000,
        aux_logits: bool = True,
        transform_input: bool = False,
        init_weights: Optional[bool] = None,
        blocks: Optional[List[Callable[..., nn.Module]]] = None
    ) -> None:
        super(GoogLeNet, self).__init__()
        self.googlenet = googlenet(pretrained=True)
        if blocks is None:
            blocks = [BasicConv2d, Inception, InceptionAux]
        if init_weights is None:
            init_weights = False
        assert len(blocks) == 3
        conv_block = blocks[0]
        inception_block = blocks[1]
        inception_aux_block = blocks[2]

        self.aux_logits = aux_logits
        self.transform_input = transform_input

        self.conv1 = conv_block(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = conv_block(64, 64, kernel_size=1)
        self.conv3 = conv_block(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = inception_block(256, 128, 128, 192, 32, 96, 64)
        # self.inception3c = inception_block(480, 192, 160, 192, 32, 96, 64) # Module Step 1
        # self.inception3d = inception_block(544, 192, 160, 128, 48, 128, 32) # Module Step 1b
        # self.inception3e = inception_block(480, 192, 160, 256, 48, 160, 96) # Module Step 1b        
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = inception_block(528, 256, 160, 320, 32, 128, 128)
        #self.inception4f = inception_block(528, 256, 160, 320, 32, 128, 128) # Module Step 2
        #self.inception4f = inception_block(528, 256, 160, 320, 32, 128, 128) # Module Step 2b
        #self.inception4f = inception_block(528, 256, 160, 320, 32, 128, 128) # Module Step 2b
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = inception_block(832, 384, 192, 384, 48, 128, 128)
        #self.inception5c = inception_block(832, 384, 192, 384, 48, 128, 128) # Module Step 3
        #self.inception5c = inception_block(832, 384, 192, 384, 48, 128, 128) # Module Step 3b
        #self.inception5c = inception_block(832, 384, 192, 384, 48, 128, 128) # Module Step 3b

        if aux_logits:
            self.aux1 = inception_aux_block(512, num_classes)
            self.aux2 = inception_aux_block(528, num_classes)
        else:
            self.aux1 = None  # type: ignore[assignment]
            self.aux2 = None  # type: ignore[assignment]

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1024, num_classes)

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                X = stats.truncnorm(-2, 2, scale=0.01)
                values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _transform_input(self, x: Tensor) -> Tensor:
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # # N x 320 x 28 x 28

        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        aux1: Optional[Tensor] = None
        if self.aux1 is not None:
            if self.training:
                aux1 = self.aux1(x)

        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14
        aux2: Optional[Tensor] = None
        if self.aux2 is not None:
            if self.training:
                aux2 = self.aux2(x)

        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # N x 1000 (num_classes)
        return x



class GoogLeNetPlus(nn.Module):
    """
    
    """

    __constants__ = ['aux_logits', 'transform_input']

    def __init__(
        self,
        num_classes: int = 1000,
        aux_logits: bool = True,
        transform_input: bool = False,
        init_weights: Optional[bool] = None,
        blocks: Optional[List[Callable[..., nn.Module]]] = None
    ) -> None:
        super(GoogLeNetPlus, self).__init__()
        self.googlenet = googlenet(pretrained=True)
        if blocks is None:
            blocks = [BasicConv2d, Inception, InceptionAux]
        if init_weights is None:
            init_weights = False
        assert len(blocks) == 3
        conv_block = blocks[0]
        inception_block = blocks[1]
        inception_aux_block = blocks[2]

        self.aux_logits = aux_logits
        self.transform_input = transform_input

        self.conv1 = conv_block(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = conv_block(64, 64, kernel_size=1)
        self.conv3 = conv_block(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = inception_block(256, 128, 128, 192, 32, 96, 64)
        self.inception3c = inception_block(480, 192, 160, 192, 32, 96, 64) # Module Step 1
        self.inception3d = inception_block(544, 192, 160, 128, 48, 128, 32) # Module Step 1b
        self.inception3e = inception_block(480, 128, 160, 192, 48, 64, 96) # Module Step 1b
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = inception_block(528, 256, 160, 320, 32, 128, 128)
        self.inception4f = inception_block(832, 192, 160, 256, 32, 64, 64) # Module Step 2
        self.inception4g = inception_block(576, 256, 160, 320, 32, 128, 128) # Module Update 2
        self.inception4h = inception_block(832, 256, 160, 320, 32, 128, 128) # Module Update 22
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = inception_block(832, 384, 192, 384, 48, 128, 128)
        self.inception5c = inception_block(1024, 384, 192, 448, 48, 128, 128) # Module Step 3
        self.inception5d = inception_block(1088, 384, 192, 384, 48, 128, 128) # Module Update 3
        self.inception5e = inception_block(1024, 384, 192, 384, 48, 128, 128) # Module Update 33
        

        
        if aux_logits:
            self.aux1 = inception_aux_block(512, num_classes)
            self.aux2 = inception_aux_block(528, num_classes)
        else:
            self.aux1 = None  # type: ignore[assignment]
            self.aux2 = None  # type: ignore[assignment]

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1024, num_classes)

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                X = stats.truncnorm(-2, 2, scale=0.01)
                values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _transform_input(self, x: Tensor) -> Tensor:
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)

        #x= self.inception3c(x) # add in

        #x = self.inception3d(x) # add in

        #x = self.inception3e(x) # add in


        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        aux1: Optional[Tensor] = None
        if self.aux1 is not None:
            if self.training:
                aux1 = self.aux1(x)

        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14

       
        aux2: Optional[Tensor] = None
        if self.aux2 is not None:
            if self.training:
                aux2 = self.aux2(x)

        x = self.inception4e(x)
        #x = self.inception4f(x) # add in 
        #x = self.inception4g(x) # add in 
        #x = self.inception4h(x) # add in 
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        #x = self.inception5c(x) # add in
        #x = self.inception5d(x) # add in
        #x = self.inception5e(x) # add in
        
        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # N x 1000 (num_classes)
        return x




class Inception(nn.Module):

    def __init__(
        self,
        in_channels: int,
        ch1x1: int,
        ch3x3red: int,
        ch3x3: int,
        ch5x5red: int,
        ch5x5: int,
        pool_proj: int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Inception, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1 = conv_block(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            conv_block(in_channels, ch3x3red, kernel_size=1),
            conv_block(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, ch5x5red, kernel_size=1),
            # Here, kernel_size=3 instead of kernel_size=5 is a known bug.
            # Please see https://github.com/pytorch/vision/issues/906 for details.
            conv_block(ch5x5red, ch5x5, kernel_size=3, padding=1)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            conv_block(in_channels, pool_proj, kernel_size=1)
        )

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InceptionAux, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.conv = conv_block(in_channels, 128, kernel_size=1)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = F.adaptive_avg_pool2d(x, (4, 4))
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = torch.flatten(x, 1)
        # N x 2048
        x = F.relu(self.fc1(x), inplace=True)
        # N x 1024
        x = F.dropout(x, 0.7, training=self.training)
        # N x 1024
        x = self.fc2(x)
        # N x 1000 (num_classes)
        return x


class BasicConv2d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        **kwargs: Any
    ) -> None:
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
  

class TruckResnet18(nn.Module):
    """
    
    """

    def __init__(self):
        super(TruckResnet18, self).__init__()

        self.resnet18 = resnet18(pretrained=True)
        self.freeze_params(self.resnet18)
        self.resnet18.fc = nn.Identity()                            # N x 3 x 224 x 224 -> N x 2048

        self.fc = nn.Sequential(
            nn.Linear(512, 256),                                   # N x 2048 -> N x 512 /  N x 512 -> N x 256
            nn.ELU(),
            nn.Linear(256, 64),                                    # N x 512 -> N x 256 / N x 256 -> N x 64 
            nn.ELU(),
            nn.Linear(64, 32),                                     # N x 256 -> N x 64 / N x 64 -> N x 32 
            nn.ELU()
        )

        self.out = nn.Linear(32, 1)                                 # N x 64 -> N x 1 / N x 32 -> N x 1 

    def freeze_params(self, model):
        count = 0
        for param in model.parameters():
            count += 1
            if count <= 141:
                param.requires_grad = False

    def forward(self, x):
        
        x = x.view(x.size(0), 3, 224, 224)                          # N x 3 x H x W, H = 224, W = 224

        x = self.resnet18(x)                                        # N x 2048

        # input dimension needs to be monitored
        x = self.fc(x)                                              # N x 64

        x = self.out(x)                                             # N x 1

        return x
    
    
    
class TruckResnet34(nn.Module):
    """
    
    """

    def __init__(self):
        super(TruckResnet34, self).__init__()

        self.resnet34 = resnet34(pretrained=True)
        self.freeze_params(self.resnet34)
        self.resnet34.fc = nn.Identity()                            # N x 3 x 224 x 224 -> N x 2048

        self.fc = nn.Sequential(
            nn.Linear(512, 256),                                   # N x 2048 -> N x 512 /  N x 512 -> N x 256
            nn.ELU(),
            nn.Linear(256, 64),                                    # N x 512 -> N x 256 / N x 256 -> N x 64 
            nn.ELU(),
            nn.Linear(64, 32),                                     # N x 256 -> N x 64 / N x 64 -> N x 32 
            nn.ELU()
        )

        self.out = nn.Linear(32, 1)                                 # N x 64 -> N x 1 / N x 32 -> N x 1 

    def freeze_params(self, model):
        count = 0
        for param in model.parameters():
            count += 1
            if count <= 141:
                param.requires_grad = False

    def forward(self, x):
        
        x = x.view(x.size(0), 3, 224, 224)                          # N x 3 x H x W, H = 224, W = 224

        x = self.resnet34(x)                                        # N x 2048

        # input dimension needs to be monitored
        x = self.fc(x)                                              # N x 64

        x = self.out(x)                                             # N x 1

        return x    

class TruckResnet50(nn.Module):
    """
    A modified CNN model, leverages the pretrained resnet50 for features extraction https://arxiv.org/abs/1512.00567
    Transfer Learning from pretrained Resnet-50, connected with 3 dense layers. 
    Total params: 24.7M (24704961), pretrained 14.5M (14582848), trainable 10.1M (10122113)
 
    """

    def __init__(self):
        super(TruckResnet50, self).__init__()

        self.resnet50 = resnet50(pretrained=True)
        self.freeze_params(self.resnet50)
        self.resnet50.fc = nn.Identity()                            # N x 3 x 224 x 224 -> N x 2048

        self.fc = nn.Sequential(
            nn.Linear(2048, 512),                                   # N x 2048 -> N x 512
            nn.ELU(),
            nn.Linear(512, 256),                                    # N x 512 -> N x 256
            nn.ELU(),
            nn.Linear(256, 64),                                     # N x 256 -> N x 64
            nn.ELU()
        )

        self.out = nn.Linear(64, 1)                                 # N x 64 -> N x 1

    def freeze_params(self, model):
        count = 0
        for param in model.parameters():
            count += 1
            if count <= 141:
                param.requires_grad = False

    def forward(self, x):
        
        x = x.view(x.size(0), 3, 224, 224)                          # N x 3 x H x W, H = 224, W = 224

        x = self.resnet50(x)                                        # N x 2048

        # input dimension needs to be monitored
        x = self.fc(x)                                              # N x 64

        x = self.out(x)                                             # N x 1

        return x

    
    
class TruckResnet101(nn.Module):
    """
    A modified CNN model, leverages the pretrained resnet101 for features extraction https://arxiv.org/abs/1512.00567
    Transfer Learning from pretrained Resnet-101, connected with 3 dense layers. 
    Total params: 24.7M (24704961), pretrained 14.5M (14582848), trainable 10.1M (10122113)
 
    """

    def __init__(self):
        super(TruckResnet101, self).__init__()

        self.resnet101 = resnet101(pretrained=True)
        self.freeze_params(self.resnet101)
        self.resnet101.fc = nn.Identity()                            # N x 3 x 224 x 224 -> N x 2048

        self.fc = nn.Sequential(
            nn.Linear(2048, 512),                                   # N x 2048 -> N x 512
            nn.ELU(),
            nn.Linear(512, 256),                                    # N x 512 -> N x 256
            nn.ELU(),
            nn.Linear(256, 64),                                     # N x 256 -> N x 64
            nn.ELU()
        )

        self.out = nn.Linear(64, 1)                                 # N x 64 -> N x 1

    def freeze_params(self, model):
        count = 0
        for param in model.parameters():
            count += 1
            if count <= 141:
                param.requires_grad = False

    def forward(self, x):
        
        x = x.view(x.size(0), 3, 224, 224)                          # N x 3 x H x W, H = 224, W = 224

        x = self.resnet101(x)                                        # N x 2048

        # input dimension needs to be monitored
        x = self.fc(x)                                              # N x 64

        x = self.out(x)                                             # N x 1

        return x    

class TruckResnet152(nn.Module):
    """
    A modified CNN model, leverages the pretrained resnet151 for features extraction https://arxiv.org/abs/1512.00567
    Transfer Learning from pretrained Resnet-151, connected with 3 dense layers. 
    Total params: 24.7M (24704961), pretrained 14.5M (14582848), trainable 10.1M (10122113)
 
    """

    def __init__(self):
        super(TruckResnet152, self).__init__()

        self.resnet152 = resnet152(pretrained=True)
        self.freeze_params(self.resnet152)
        self.resnet152.fc = nn.Identity()                            # N x 3 x 224 x 224 -> N x 2048

        self.fc = nn.Sequential(
            nn.Linear(2048, 512),                                   # N x 2048 -> N x 512
            nn.ELU(),
            nn.Linear(512, 256),                                    # N x 512 -> N x 256
            nn.ELU(),
            nn.Linear(256, 64),                                     # N x 256 -> N x 64
            nn.ELU()
        )

        self.out = nn.Linear(64, 1)                                 # N x 64 -> N x 1

    def freeze_params(self, model):
        count = 0
        for param in model.parameters():
            count += 1
            if count <= 141:
                param.requires_grad = False

    def forward(self, x):
        
        x = x.view(x.size(0), 3, 224, 224)                          # N x 3 x H x W, H = 224, W = 224

        x = self.resnet152(x)                                        # N x 2048

        # input dimension needs to be monitored
        x = self.fc(x)                                              # N x 64

        x = self.out(x)                                             # N x 1

        return x 



### Added in ResNet 34 layers or less


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        super().__init__()
        
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 , num_classes)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None  
   
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        
        self.inplanes = planes
        
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        x = self.conv1(x)           # 224x224
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)         # 112x112

        x = self.layer1(x)          # 56x56
        x = self.layer2(x)          # 28x28
        x = self.layer3(x)          # 14x14
        x = self.layer4(x)          # 7x7

        x = self.avgpool(x)         # 1x1
        x = torch.flatten(x, 1)     # remove 1 X 1 grid and make vector of tensor shape 
        x = self.fc(x)

        return x




# Add in Larger ResNet larger than 34 layers, most well known - ResNet50







#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model18 = TruckResnet18().to(device)
# model34 = TruckResnet34().to(device)
# model50 = TruckResnet50().to(device)
# model101 = TruckResnet101().to(device)
#model152 = TruckResnet152().to(device)
#modelGoogLeNet = GoogLeNet().to(device)
#modelGoogLeNetPlus = GoogLeNetPlus().to(device)

# model18_summary = summary(model18,input_size=(3,224,224))
# model34_summary = summary(model34,input_size=(3,224,224))
# model50_summary = summary(model50,input_size=(3,224,224))
# model101_summary = summary(model101,input_size=(3,224,224))
# model152_summary = summary(model152,input_size=(3,224,224))
#GoogLeNet_summary = summary(modelGoogLeNet,input_size=(3,224,224))
#GoogLeNetPlus_summary = summary(modelGoogLeNetPlus,input_size=(3,224,224))

# import torchvision.models as models
# resnet50 = models.resnet50(pretrained=True)
# features = nn.Sequential(*(list(resnet50.children())[0:10]))
# features

class BasicConv2(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv2, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv2(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


#######Attention!!!!!
import torch.nn.functional as F

#"""The Attention Module is built by pre-activation Residual Unit [11] with the 
#number of channels in each stage is the same as ResNet [10]."""

class PreActResidualUnit(nn.Module):
    """PreAct Residual Unit
    Args:
        in_channels: residual unit input channel number
        out_channels: residual unit output channel numebr
        stride: stride of residual unit when stride = 2, downsample the featuremap
    """

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        bottleneck_channels = int(out_channels / 4)
        self.residual_function = nn.Sequential(
            #1x1 conv
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, bottleneck_channels, 1, stride),

            #3x3 conv
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channels, bottleneck_channels, 3, padding=1),

            #1x1 conv
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channels, out_channels, 1)
        )

        self.shortcut = nn.Sequential()
        if stride != 2 or (in_channels != out_channels):
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride=stride)
    
    def forward(self, x):

        res = self.residual_function(x)
        shortcut = self.shortcut(x)
        return res + shortcut
#class Block(nn.Module):
    #def __init__(self, in_channels, out_channels,stride=1):
     #   super(Block, self).__init__()
      #  self.expansion = 1
        #self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        #self.bn1 = nn.BatchNorm2d(out_channels)
       # self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        #self.bn2 = nn.BatchNorm2d(out_channels)
        #self.conv3 = nn.Conv2d(out_channels,out_channels*self.expansion,kernel_size=3, stride=1, padding=0)
        #self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        #self.relu = nn.ReLU()
        #if stride !=1:
        #self.identity_downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels*1, kernel_size=1, stride=stride), nn.BatchNorm2d(out_channels*1))
        #self.stride = stride

    #def forward(self,x):
     #   identity = x

        #x = self.conv1(x)
        #x = self.bn1(x)
        #x = self.relu(x)
      #  x = self.conv2(x)
       # x = self.bn2(x)
        #x = self.relu(x)
        #x = self.conv3(x)
        #x = self.bn3(x)

        #if self.identity_downsample is not None:
       # identity = self.identity_downsample(identity)
        
        #x += identity
        #x = self.relu(x)
        #return x


class Block(nn.Module):
    '''
    实现子module: Residual Block
    '''
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(Block, self).__init__()
        self.left = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
                nn.BatchNorm2d(outchannel),
                nn.ReLU(inplace=True),
                nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
                nn.BatchNorm2d(outchannel) )
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        #out += residual
        return F.relu(out)



class AttentionModule1(nn.Module):
    
    def __init__(self, in_channels, out_channels, p=1, t=2, r=1):
        super().__init__()
        #"""The hyperparameter p denotes the number of preprocessing Residual 
        #Units before splitting into trunk branch and mask branch. t denotes 
        #the number of Residual Units in trunk branch. r denotes the number of 
        #Residual Units between adjacent pooling layer in the mask branch."""
        assert in_channels == out_channels

        self.pre = self._make_residual(in_channels, out_channels, p)
        self.trunk = self._make_residual(in_channels, out_channels, t)#downsample
        self.soft_resdown1 = self._make_residual(in_channels, out_channels, r)
        self.soft_resdown2 = self._make_residual(in_channels, out_channels, r)
        self.soft_resdown3 = self._make_residual(in_channels, out_channels, r)
        self.soft_resdown4 = self._make_residual(in_channels, out_channels, r)

        self.soft_resup1 = self._make_residual(in_channels, out_channels, r)
        self.soft_resup2 = self._make_residual(in_channels, out_channels, r)
        self.soft_resup3 = self._make_residual(in_channels, out_channels, r)
        self.soft_resup4 = self._make_residual(in_channels, out_channels, r)

        self.shortcut_short = block(in_channels, out_channels, 1)
        self.shortcut_long = block(in_channels, out_channels, 1)

        self.sigmoid = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            #nn.Sigmoid()
            nn.ReLU()
        ) 
        
        self.last = self._make_residual(in_channels, out_channels, p)
    
    def forward(self, x):
        ###We make the size of the smallest output map in each mask branch 7*7 to be consistent
        #with the smallest trunk output map size.
        ###Thus 3,2,1 max-pooling layers are used in mask branch with input size 56 * 56, 28 * 28, 14 * 14 respectively.
        x = self.pre(x)
        input_size = (x.size(2), x.size(3))

        x_t = self.trunk(x)

        #first downsample out 28
        x_s = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x_s = self.soft_resdown1(x_s)

        #28 shortcut
        shape1 = (x_s.size(2), x_s.size(3))
        shortcut_long = self.shortcut_long(x_s)

        #seccond downsample out 14
        x_s = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x_s = self.soft_resdown2(x_s)

        #14 shortcut
        shape2 = (x_s.size(2), x_s.size(3))
        shortcut_short = self.soft_resdown3(x_s)

        #third downsample out 7
        x_s = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x_s = self.soft_resdown3(x_s)

        #mid
        x_s = self.soft_resdown4(x_s)
        x_s = self.soft_resup1(x_s)

        #first upsample out 14
        x_s = self.soft_resup2(x_s)
        x_s = F.interpolate(x_s, size=shape2)
        x_s += shortcut_short

        #second upsample out 28
        x_s = self.soft_resup3(x_s)
        x_s = F.interpolate(x_s, size=shape1)
        x_s += shortcut_long

        #thrid upsample out 54
        x_s = self.soft_resup4(x_s)
        x_s = F.interpolate(x_s, size=input_size)

        x_s = self.sigmoid(x_s)
        x = (1 + x_s) * x_t
        x = self.last(x)

        return x

    def _make_residual(self, in_channels, out_channels, p):

        layers = []
        for _ in range(p):
            layers.append(Block(in_channels, out_channels, 1))

        return nn.Sequential(*layers)

class AttentionModule2(nn.Module):
    
    def __init__(self, in_channels, out_channels, p=1, t=2, r=1):
        super().__init__()
        #"""The hyperparameter p denotes the number of preprocessing Residual 
        #Units before splitting into trunk branch and mask branch. t denotes 
        #the number of Residual Units in trunk branch. r denotes the number of 
        #Residual Units between adjacent pooling layer in the mask branch."""
        assert in_channels == out_channels

        self.pre = self._make_residual(in_channels, out_channels, p)
        self.trunk = self._make_residual(in_channels, out_channels, t)
        self.soft_resdown1 = self._make_residual(in_channels, out_channels, r)
        self.soft_resdown2 = self._make_residual(in_channels, out_channels, r)
        self.soft_resdown3 = self._make_residual(in_channels, out_channels, r)

        self.soft_resup1 = self._make_residual(in_channels, out_channels, r)
        self.soft_resup2 = self._make_residual(in_channels, out_channels, r)
        self.soft_resup3 = self._make_residual(in_channels, out_channels, r)

        self.shortcut = block(in_channels, out_channels, 1)

        self.sigmoid = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            #nn.Sigmoid()
            nn.ReLU()
        ) 
        
        self.last = self._make_residual(in_channels, out_channels, p)
    
    def forward(self, x):
        x = self.pre(x)
        input_size = (x.size(2), x.size(3))

        x_t = self.trunk(x)

        #first downsample out 14
        x_s = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x_s = self.soft_resdown1(x_s)

        #14 shortcut
        shape1 = (x_s.size(2), x_s.size(3))
        shortcut = self.shortcut(x_s)

        #seccond downsample out 7 
        x_s = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x_s = self.soft_resdown2(x_s)

        #mid
        x_s = self.soft_resdown3(x_s)
        x_s = self.soft_resup1(x_s)

        #first upsample out 14
        x_s = self.soft_resup2(x_s)
        x_s = F.interpolate(x_s, size=shape1)
        x_s += shortcut

        #second upsample out 28
        x_s = self.soft_resup3(x_s)
        x_s = F.interpolate(x_s, size=input_size)

        x_s = self.sigmoid(x_s)
        x = (1 + x_s) * x_t
        x = self.last(x)

        return x

    def _make_residual(self, in_channels, out_channels, p):

        layers = []
        for _ in range(p):
            layers.append(Block(in_channels, out_channels, 1))

        return nn.Sequential(*layers)

class AttentionModule3(nn.Module):
    
    def __init__(self, in_channels, out_channels, p=1, t=2, r=1):
        super().__init__()

        assert in_channels == out_channels

        self.pre = self._make_residual(in_channels, out_channels, p)
        self.trunk = self._make_residual(in_channels, out_channels, t)
        self.soft_resdown1 = self._make_residual(in_channels, out_channels, r)
        self.soft_resdown2 = self._make_residual(in_channels, out_channels, r)

        self.soft_resup1 = self._make_residual(in_channels, out_channels, r)
        self.soft_resup2 = self._make_residual(in_channels, out_channels, r)

        self.shortcut = block(in_channels, out_channels, 1)

        self.sigmoid = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            #nn.Sigmoid()
            nn.ReLU()
        ) 
        
        self.last = self._make_residual(in_channels, out_channels, p)
    
    def forward(self, x):
        x = self.pre(x)
        input_size = (x.size(2), x.size(3))

        x_t = self.trunk(x)

        #first downsample out 14
        x_s = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x_s = self.soft_resdown1(x_s)

        #mid
        x_s = self.soft_resdown2(x_s)
        x_s = self.soft_resup1(x_s)

        #first upsample out 14
        x_s = self.soft_resup2(x_s)
        x_s = F.interpolate(x_s, size=input_size)

        x_s = self.sigmoid(x_s)
        x = (1 + x_s) * x_t
        x = self.last(x)

        return x

    def _make_residual(self, in_channels, out_channels, p):

        layers = []
        for _ in range(p):
            layers.append(Block(in_channels, out_channels, 1))

        return nn.Sequential(*layers)

class Attention(nn.Module):
    """residual attention netowrk
    Args:
        block_num: attention module number for each stage
    """

    def __init__(self, block_num, class_num=1000):
        
        super().__init__()
        self.pre_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.stage1 = self._make_stage(64, 128, block_num[0], AttentionModule1)
        self.stage2 = self._make_stage(128, 256, block_num[1], AttentionModule2)
        self.stage3 = self._make_stage(256, 512, block_num[2], AttentionModule3)
        self.stage4 = nn.Sequential(
            PreActResidualUnit(512, 1024, 2),
            PreActResidualUnit(1024, 2048, 1),
            PreActResidualUnit(2048, 2048, 1)
        )
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(2048, 1000)
    
    def forward(self, x):
        x = self.pre_conv(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x

    def _make_stage(self, in_channels, out_channels, num, block):

        layers = []
        layers.append(PreActResidualUnit(in_channels, out_channels, 2))

        for _ in range(num):
            layers.append(Block(out_channels, out_channels))

        return nn.Sequential(*layers)
    
def attention56():
    return Attention([1, 1, 1])

def attention92():
    return Attention([1, 2, 3])
    
class ResNetLarge(nn.Module):
    def __init__(self,block,layers, image_channels=3, num_classes=1000):
        super(ResNetLarge, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        self.layer1 = self._make_layer(AttentionModule1, layers[0], in_channels=64, out_channels=128, stride=1)
        self.layer2 = self._make_layer(AttentionModule2, layers[1], in_channels=128, out_channels=256, stride=2)
        self.layer3 = self._make_layer(AttentionModule3, layers[2], in_channels=256, out_channels=512, stride=2)
        self.layer4 = self._make_layer(Block, layers[3], in_channels=512, out_channels=2048, stride=2) # 2048 out_channels at the end

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fc(x)
        return x

    def _make_layer(self,block, num_residual_blocks, in_channels, out_channels,stride):
        #identity_downsample = None
        layers = []
        layers.append(Block(in_channels, out_channels,stride))
        #in_channels = out_channels*4

        for i in range(num_residual_blocks -1):
            layers.append(Block(out_channels, out_channels)) # 256 -> 64, 64*4

        return nn.Sequential(*layers)