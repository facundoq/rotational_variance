import torch.nn as nn
import torch.nn.functional as F
from pytorch.model.util import SequentialWithIntermediates


class ConvBN(nn.Module):
    def __init__(self,in_filters,out_filters,kernel_size,stride=1):
        super(ConvBN, self).__init__()
        if kernel_size==0:
            padding=0
        else:
            padding=1
        self.model=nn.Sequential(
            nn.Conv2d(in_filters, out_filters, kernel_size=kernel_size, padding=padding,stride=stride),
            nn.BatchNorm2d(out_filters),
        )

    def forward(self,x):
        return self.model.forward(x)


class AllConvolutional(nn.Module):
    def __init__(self, input_shape, num_classes=10,filters=96,dropout=False):
        super(AllConvolutional, self).__init__()
        self.name = self.__class__.__name__
        h,w,c=input_shape
        filters2=filters*2
        self.dropout=dropout
        self.conv1 = ConvBN(c, filters, 3)
        self.conv2 = ConvBN(filters, filters, 3)
        self.conv3 = ConvBN(filters, filters, 3,stride=2)
        self.conv4 = ConvBN(filters, filters2, 3)
        self.conv5 = ConvBN(filters2, filters2, 3)
        self.conv6 = ConvBN(filters2, filters2, 3, stride=2)
        self.conv7 = ConvBN(filters2, filters2, 3)
        self.conv8 = ConvBN(filters2, filters2, 1)
        self.class_conv = nn.Conv2d(filters2, num_classes, 1)

        self.layers= [self.conv1, self.conv2, self.conv3, self.conv4,
                         self.conv5, self.conv6, self.conv7, self.conv8,
                         self.class_conv]

    def forward(self, x):
        out,intermediates=self.forward_intermediates(x)
        return out

    def forward_intermediates(self, x):

        intermediates=[]

        for layer in self.layers:
                x=layer(x)
                intermediates.append(x)
                x=F.relu(x)
                intermediates.append(x)
        pool_out = x.reshape(x.size(0), x.size(1), -1).mean(-1)
        intermediates.append(pool_out)

        log_probabilities=F.log_softmax(pool_out,dim=1)
        intermediates.append(log_probabilities)

        return log_probabilities,intermediates

    def layer_names(self):
        return [f"c{i}" for i in range(8)]+["cc"]

    def get_layer(self,layer_name):

        layer_names=self.layer_names()
        index=layer_names.index(layer_name)
        return self.layers[index]

    def n_intermediates(self):
        return len(self.intermediates_names())

    def intermediates_names(self):
        names=[ [x,x+"act"] for x in self.layer_names()]
        names = sum(names,[])
        names+=["ap","smax"]
        return names



# class ConvBNAct(nn.Module):
#     def __init__(self,in_filters,out_filters,stride=1,kernel_size=3):
#         super(ConvBNAct, self).__init__()
#         if kernel_size==1:
#             padding=0
#         else:
#             padding=1
#         self.model=nn.Sequential(
#             nn.Conv2d(in_filters, out_filters, kernel_size=kernel_size, padding=padding,stride=stride),
#             nn.BatchNorm2d(out_filters),
#             nn.ReLU()
#         )
#
#     def forward(self,x):
#         return self.model.forward(x)
#
# # class AllConv(nn.Module):
# #     def __init__(self, input_shape, num_classes, filters=96):
# #         super(AllConv, self).__init__()
# #         self.name = self.__class__.__name__
# #         filters2 = filters * 2
# #         h, w, channels = input_shape
# #
# #         self.conv = nn.Sequential(
# #             nn.Conv2d(channels, filters, kernel_size=3, padding=1),
# #             nn.BatchNorm2d(filters),
# #             nn.ReLU(),
# #             ConvBNAct(filters, filters),
# #             ConvBNAct(filters, filters, stride=2),
# #             ConvBNAct(filters, filters2, ),
# #             ConvBNAct(filters2, filters2),
# #             ConvBNAct(filters2, filters2, stride=2),
# #             ConvBNAct(filters2, filters2),
# #             ConvBNAct(filters2, filters2, kernel_size=1),
# #
# #         )
# #
# #         final_channels = filters2
# #
# #         self.class_conv = nn.Conv2d(final_channels, num_classes, 1)
# #
# #     def forward(self, x):
# #         # # print(x.shape)
# #         # x = F.relu(self.conv1(x))
# #         # # print(x.shape)
# #         # x = F.relu(self.conv2(x))
# #         # # print(x.shape)
# #         # x = F.relu(self.conv3(x))
# #         # # print(x.shape)
# #         # x = F.relu(self.conv4(x))
# #         # # print(x.shape)
# #         # # print(x.shape)
# #         # x = F.relu(self.conv5(x))
# #         x = self.conv(x)
# #
# #         class_out = F.relu(self.class_conv(x))
# #         pool_out = class_out.reshape(class_out.size(0), class_out.size(1), -1).mean(-1)
# #         # pool_out = F.adaptive_avg_pool2d(class_out, 1)
# #         # pool_out.squeeze_(-1)
# #         # pool_out.squeeze_(-1)
# #
# #         log_probabilities = F.log_softmax(pool_out, dim=1)
# #         return log_probabilities
# #
# #     def layer_names(self):
# #         return [f"conv{i}" for i in range(8)] + ["class_conv"]
# #
# #     def layers(self):
# #         convs=list(self.conv.children())
# #         return [convs[0]]+[list(convs[i].model.children())[0] for i in range(3,10)]
# #
# #
# #         # return x
