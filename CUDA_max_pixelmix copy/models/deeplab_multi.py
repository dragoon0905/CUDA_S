# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

affine_par = True


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, bn_momentum=0.1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=1, stride=stride, bias=False)  # change
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,  # change
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
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


class Classifier_Module(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
            return out


class TM(nn.Module):
    def __init__(self, inplanes, num_classes):
        super(TM, self).__init__()
        self.num_classes = num_classes
        self.module_list = nn.ModuleList()
        self.module_list.append(conv1x1(inplanes, num_classes))
        self.module_list.append(nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                              nn.Conv2d(inplanes, num_classes, 1, bias=False),
                                              nn.ReLU()))

        for i, m in enumerate(self.module_list):
            if i == 0:
                m.weight.data.normal_(0, 0.01)
            else:
                for n in m:
                    if isinstance(n, nn.Conv2d):
                        n.weight.data.normal_(0, 0.01)
                    elif isinstance(n, nn.BatchNorm2d):
                        n.weight.data.fill_(1)
                        n.bias.data.zero_()

    def forward(self, x):
        out1 = self.module_list[0](x)
        out2 = self.module_list[1](x)
        return out1, out2



class ResNetMulti(nn.Module):
    def __init__(self, block, layers, num_classes, tm, num_target, eval_target):
        
        super(ResNetMulti, self).__init__()
        self.tm=tm     #changed
        self.num_target=num_target#changed
        self.eval_target=eval_target#changed


        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=1, dilation=4)
        self.layer5 = self._make_pred_layer(Classifier_Module, 1024, [6, 12, 18, 24], [
                                            6, 12, 18, 24], num_classes)
        self.layer6 = self._make_pred_layer(Classifier_Module, 2048, [6, 12, 18, 24], [
                                            6, 12, 18, 24], num_classes)

        if self.tm:
            for num in range(self.num_target):
                TM_name1 = 'TM' + str(num + 1) + '_1'
                TM_name2 = 'TM' + str(num + 1) + '_2'
                setattr(self, TM_name1, TM(1024, num_classes))
                setattr(self, TM_name2, TM(2048, num_classes))


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # New! Rotation prediction head
        # self.rotation_prediction_head = nn.Identity()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par))
        layers = []
        layers.append(block(self.inplanes, planes, stride,
                            dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _make_pred_layer(self, block, inplanes, dilation_series, padding_series, num_classes):
        return block(inplanes, dilation_series, padding_series, num_classes)

    def forward(self, x, forward_target=0):
        input_size = x.shape[-1], x.shape[-2] #x.size()[2:] [1024,512].

        # if self.tm:
        #     if self.eval_target == -1:
        #         TM_name1 = 'TM' + str(self.num_target) + '_1'
        #         TM_name2 = 'TM' + str(self.num_target) + '_2'
        #     else:
        #         TM_name1 = 'TM' + str(self.eval_target) + '_1'
        #         TM_name2 = 'TM' + str(self.eval_target) + '_2'
        if forward_target > 0 :
            # if self.eval_target == -1:
            TM_name1 = 'TM' + str(forward_target) + '_1'
            TM_name2 = 'TM' + str(forward_target) + '_2'
        elif forward_target == 0:
            pass


        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x1 = self.layer5(x)
        pred_ori1 = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear', align_corners=True)(x1)  # ResNet + ASPP1

        if self.tm:
            x3_1, x3_2 = getattr(self, TM_name1)(x)
            new_x = x1 + x3_1 + x3_2
            pred1 = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear', align_corners=True)(new_x)  # ResNet + (ASPP1+TM1)

        x = self.layer4(x)
        x2 = self.layer6(x)
        pred_ori2 = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear', align_corners=True)(x2)  # ResNet + ASPP2

        if self.tm:
            x3_1, x3_2 = getattr(self, TM_name2)(x)
            new_x = x2 + x3_1 + x3_2
            pred2 = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear', align_corners=True)(new_x)  # ResNet + (ASPP2+TM2)
        
        if self.tm:
            return pred2, pred1  # changed!
        else:
            return pred_ori2, pred_ori1

    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []

        b.append(self.conv1)
        b.append(self.bn1)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        b.append(self.layer5.parameters())
        b.append(self.layer6.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def TM_params(self):
        TM_name1 = 'TM' + str(self.num_target) + '_1'
        TM_name2 = 'TM' + str(self.num_target) + '_2'

        b = []
        b.append(getattr(self, TM_name1).parameters())
        b.append(getattr(self, TM_name2).parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i


    def optim_parameters(self, lr, tm):
        optim_parameters = [{'params': self.get_1x_lr_params_NOscale(), 'lr': lr},
                            {'params': self.get_10x_lr_params(), 'lr': 10 * lr}]
        if tm:
            optim_parameters += [{'params': self.TM_params(), 'lr': 10 * lr}]
        return optim_parameters


def DeeplabMulti(tm, num_target, eval_target, num_classes=21, init=None):

    # Create model
    model = ResNetMulti(Bottleneck, [3, 4, 23, 3], num_classes, tm, num_target, eval_target)

    # Standard DeepLabv2 initialization
    # if init:
    #     saved_state_dict = torch.load(init)
    #     new_params = model.state_dict().copy()
    #     for i in saved_state_dict:
    #         i_parts = i.split('.')
    #         if not i_parts[1] == 'layer5':
    #             new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
    #     model.load_state_dict(new_params)

    return model
