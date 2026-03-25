"""
Densely Connected Convolutional Networks.
"""

import math
import torch
import torch.nn as nn
from collections import OrderedDict
import warnings

from . import pytorchmodelbase as dmspytorchmodelbase
from ...errors import modelerrors as dmsmodelerrors

#----------------------------------------------------------------------------------------------------

class Bottleneck(nn.Module):
    def __init__(self, channels, growth_rate, dropout_rate, bottleneck_rate, padding):
        super(Bottleneck, self).__init__()
        
        self.layers = nn.Sequential(OrderedDict([
            ('bn0',   nn.BatchNorm2d(channels)), 
            ('act0',  nn.ReLU()),
            ('conv0', nn.Conv2d(channels, bottleneck_rate * growth_rate, kernel_size=1, padding=padding, bias=False)),
            #('drop0', nn.Dropout(p=dropout_rate)),
            ('bn1',   nn.BatchNorm2d(bottleneck_rate * growth_rate)),
            ('act1',  nn.ReLU()),
            ('conv1', nn.Conv2d(bottleneck_rate * growth_rate, growth_rate, kernel_size=3, padding=padding, bias=False)),
            ('drop1', nn.Dropout2d(p=dropout_rate))
        ]))

    def forward(self, x):
        return torch.cat((torch.nn.functional.pad(x, (-1, -1, -1, -1)), self.layers(x)), 1)

    
class SingleLayer(nn.Module):
    def __init__(self, channels, growth_rate, dropout_rate, padding):
        super(SingleLayer, self).__init__()
        
        self.layers = nn.Sequential(OrderedDict([
            ('bn0',   nn.BatchNorm2d(channels)), 
            ('act0',  nn.ReLU()),
            ('conv0', nn.Conv2d(channels, growth_rate, kernel_size=3, padding=padding, bias=False)),
            ('drop0', nn.Dropout(p=dropout_rate))
        ]))

    def forward(self, x):
        return torch.cat((x, self.layers(x)), 1)

    
class Transition(nn.Module):
    def __init__(self, channels, out_channels, padding):
        super(Transition, self).__init__()
        
        self.layers = nn.Sequential(OrderedDict([
            ('bn0',   nn.BatchNorm2d(channels)), 
            ('act0',  nn.ReLU()),
            ('conv0', nn.Conv2d(channels, out_channels, kernel_size=1, padding=padding, bias=False)),
            ('pool0', nn.AvgPool2d(2))
        ]))

    def forward(self, x):
        return self.layers(x)


def make_dense_block(channels, growth_rate, n_blocks, bottleneck, dropout_rate, bottleneck_rate, padding):
    layers = nn.Sequential()
    for i in range(int(n_blocks)):
        if bottleneck:
            layers.add_module('bottle{}'.format(i), Bottleneck(channels, growth_rate, dropout_rate, bottleneck_rate, padding))
        else:
            layers.add_module('single{}'.format(i), SingleLayer(channels, growth_rate, dropout_rate, padding))
        channels += growth_rate
    return layers


class DenseNetBase(nn.Module):
    def __init__(self, input_shape, layers_per_block, growth_rate, reduction, num_classes, bottleneck, bottleneck_rate, valid_padding, dropout_rate=0.0, pool_initial=False, pool_final=True):
        super(DenseNetBase, self).__init__()
            
        channels = 2 * growth_rate
        padding = 0 if valid_padding else 1

        # Initial convolution.
        #
        if pool_initial:
            initial_padding = 0 if valid_padding else 3
            self.layers = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(input_shape[0], channels, kernel_size=7, stride=2, padding=initial_padding, bias=False)),
                ('bn0', nn.BatchNorm2d(channels)),
                ('act0', nn.ReLU()),
                ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=padding, ceil_mode=False))
            ]))
        else:
            self.layers = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(input_shape[0], channels, kernel_size=3, padding=padding, bias=False))
            ]))
        
        for block_index, n_blocks in enumerate(layers_per_block):
            
            self.layers.add_module('dense{}'.format(block_index), make_dense_block(channels, growth_rate, n_blocks, bottleneck, dropout_rate, bottleneck_rate, padding))        
            channels += n_blocks * growth_rate
            
            if block_index == len(layers_per_block) - 1:
                break 
            
            out_channels = int(math.floor(channels * reduction))
            self.layers.add_module('trans{}'.format(block_index), Transition(channels, out_channels, padding))
            channels = out_channels
            

        self.layers.add_module('bn{}'.format(block_index), nn.BatchNorm2d(channels))
        self.layers.add_module('act{}'.format(block_index), nn.ReLU())
        
        final_conv_size = self.layers(torch.rand(*input_shape).unsqueeze(0)).shape[-1]
        if pool_final:
            self.layers.add_module('pool{}'.format(block_index), nn.AvgPool2d(kernel_size=final_conv_size, stride=1, padding=0))
            final_conv_size = 1
        self.layers.add_module('fconv', nn.Conv2d(channels, num_classes, kernel_size=final_conv_size, stride=1, padding=0, bias=True))
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.reset_parameters()
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.layers(x)
        
        # Remove spatial dimensions when none are left.
        #
        return out.squeeze(-1).squeeze(-1)


class DenseNet(dmspytorchmodelbase.PytorchModelBase):
    """DenseNet model implementation in PyTorch."""

    def __init__(self, name=None, description=None):
        """
        Initialize model parameters.

        Args:
            name (str): Name of the model.
            description (str): Description of the model.
        """

        # Initialize base class.
        #
        super().__init__(name, description)

        # Initialize members.
        #
        self.__input_shape = None      # Input shape: (channels, rows, cols).
        self.__valid_padding = False   # Use valid padding only.
        self.__pool_initial = False    # Pool initial convolutional layer.
        self.__pool_final = False      # Use global average pooling before the last convolutional layer.
        self.__growth_rate = 0         # Number of filters to add per convolutional block.
        self.__num_classes = 0         # Number of output classes.
        self.__layers_per_block = []   # Number of layers in each block.
        self.__bottleneck = False      # Determine to use Bottleneck layers.
        self.__bottleneck_rate = 4     # Determine the factor used in Bottleneck layers.
        self.__reduction = 0.0         # Reduction of the number of filters to the transition block.
        self.__dropout_rate = 0.0      # Probability of dropout on the dropout layers.
        
    def configure(self, input_shape, valid_padding, pool_initial, pool_final, growth_rate, num_classes, 
                  layers_per_block, bottleneck, reduction, dropout_rate, bottleneck_rate):
        """
        Configure the network.

        Args:
            input_shape (tuple): Input shape (height, width, channels).
            valid_padding (bool): Only use valid padding. Introduces cropping layers.
            pool_initial (bool): As a first layer use a 7x7 convolution with 2x2 stride, and a 3x3 max pooling with 2x2 stride, if true. A single 3x3 convolution with 1x1 stride otherwise.
            pool_final (bool): Add global average pooling before the final convolutional layer.
            growth_rate (int): Filter count growth rate. The initial convolution produces 2 * growth_rate filters, all other 3x3 convolutions in the dense blocks produces this amount.
            classes (int): Number of classes.
            layers_per_block (list): Number of 3x3 convolutions in a dense block.
            bottleneck (bool): The bottleneck 1x1 convolution layers produce growth_rate * bottleneck_rate filters. (In the article it is fixed to 4.)
            weight_decay (float): L2 regularization weight.
            compression (float): Reduction of the number of filters in the transition blocks.
            dropout_rate (float): Dropout probability. If the probability is larger than 0 than the dropout layers are added.
        """

        self.__input_shape = input_shape
        self.__valid_padding = valid_padding
        self.__pool_initial = pool_initial
        self.__pool_final = pool_final
        self.__growth_rate = growth_rate
        self.__num_classes = num_classes
        self.__layers_per_block = layers_per_block
        self.__bottleneck = bottleneck
        self.__reduction = reduction
        self.__dropout_rate = dropout_rate
        self.__bottleneck_rate = bottleneck_rate

    def _customdata(self):
        """
        Get a map of custom data structure to add to the dictionary of exported values.

        Returns:
            dict: Dictionary of values to add to the exported data.
        """

        return {'input_shape': self.__input_shape,
                'valid_padding': self.__valid_padding,
                'pool_initial': self.__pool_initial,
                'pool_final': self.__pool_final,
                'growth_rate': self.__growth_rate,
                'num_classes': self.__num_classes,
                'layers_per_block': self.__layers_per_block,
                'bottleneck': self.__bottleneck,
                'weight_decay': self.__weight_decay,
                'reduction': self.__reduction,
                'dropout_rate': self.__dropout_rate,
                'bottleneck_rate': self.__bottleneck_rate}

    def _setcustomdata(self, data_maps):
        """
        Set the class specific data.

        Args:
            data_maps (dict): Custom data map.
        """

        self.__input_shape = data_maps.get('input_shape', None)
        self.__valid_padding = data_maps.get('valid_padding', False)
        self.__pool_initial = data_maps.get('pool_initial', False)
        self.__pool_final = data_maps.get('pool_final', False)
        self.__growth_rate = data_maps.get('growth_rate', 0)
        self.__num_classes = data_maps.get('num_classes', 0)
        self.__layers_per_block = data_maps.get('layers_per_block', [])
        self.__bottleneck = data_maps.get('bottleneck', False)
        self.__weight_decay = data_maps.get('weight_decay', 0.0)
        self.__reduction = data_maps.get('reduction', 0.0)
        self.__dropout_rate = data_maps.get('dropout_rate', 0.0)
        self.__bottleneck_rate = data_maps.get('bottleneck_rate', 4)

    def build(self):
        """
        Build the network instance with the pre-configured parameters.
        
        TODO: 
          - Make it fully convolutional.
          - Make the final_pool argument functional.
          - Make it work with final padding.
        """
        
        self._model_instance = DenseNetBase(input_shape=self.__input_shape,
                                            layers_per_block=self.__layers_per_block,
                                            growth_rate=self.__growth_rate,
                                            reduction=self.__reduction,
                                            num_classes=self.__num_classes,
                                            bottleneck=self.__bottleneck,
                                            dropout_rate=self.__dropout_rate,
                                            pool_initial=self.__pool_initial,
                                            pool_final=self.__pool_final,
                                            bottleneck_rate=self.__bottleneck_rate,
                                            valid_padding=self.__valid_padding)
        self._model_instance.cuda()
        self._optimizer = torch.optim.SGD(self._model_instance.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=1e-4)

    def calculate_loss(self, logits, targets, sample_weight=None, class_weight=None, *args, **kwargs):
        
        # TODO: Make compatible with non-one-hot encodings.
        targets = torch.argmax(targets, dim=1)
        
        if sample_weight is not None:
            crit = torch.nn.NLLLoss(reduction='none', weight=class_weight)
            sample_weight = sample_weight.cuda().squeeze(1)
            total_loss = 0
            logits = F.log_softmax(logits, dim=1)
            per_sample_loss = crit(logits, targets.squeeze(1))
            for i, loss in enumerate(per_sample_loss):
                sample_loss = loss
                sample_loss *= sample_weight[i]
                total_loss += sample_loss[sample_weight[i] > 0].mean()
            total_loss /= len(sample_weight)
            return total_loss
        return F.cross_entropy(logits.squeeze(), targets, weight=class_weight)

    def calculate_metric(self, name, logit, target, sample_weight):
        logit = F.softmax(logit, dim=1)
        preds = torch.argmax(logit, dim=1).cpu()
        y_class = torch.argmax(target, dim=1).cpu()
        if name == 'accuracy':
            if sample_weight is not None:
                metric = (preds[sample_weight.squeeze(1) > 0] == y_class[sample_weight.squeeze(1) > 0]).sum()
                metric /= (sample_weight.squeeze(1) > 0).sum()
            else:
                metric = (preds == y_class).sum() / preds.numel()
        elif name == 'errors':
            weight_sum = np.sum(sample_weight.numpy(), axis=(1, 2, 3)).reshape(sample_weight.shape[0], 1, 1)
            error_flat = torch.max((target - logit), dim=1)[0].cpu() * sample_weight.squeeze(1)
            error_flat = error_flat.detach().numpy()
            error_flat /= weight_sum
            metric = np.sum(error_flat, axis=(1, 2))
        return metric

    def getreconstructioninformation(self, input_shape=None):
        """
        Calculate the scale factor and padding to reconstruct the input shape.
        This function calculates the information needed to reconstruct the input image shape given the output of a layer. For each layer leading up to the
        output it will return the number of pixels lost/gained on all image edges.
        For transposed convolutions the function checks the stride and cropping method to compute the correct upsampling factor.
        Args:
            input_shape (sequence of ints): input_shape to calculate the reconstruction information for. Order should be (width, height)
                set of layers.
        Returns:
            np.array: lost pixels
            np.array: downsample factor
            np.array: interpolation factor
        Raises:
            MissingNetworkError: The network is not defined.
        """

        # PyTorch tactic:
        # 1. Add hooks to all modules
        # 2. Do forward pass with fake data
        # 3. Follow linear path back from prediction via grad_fn
        # 4. Calculate recon-information with this linear path
        #

        # Make compatible with library methods for inference.
        #
        if len(input_shape) != 2:
            input_shape = input_shape[:2]
        
        if self._model_instance is None:
            raise dmsmodelerrors.MissingNetworkError()

        # 1. Add hooks to all relevant modules
        #
        def forw_lambda(module, inpt, outpt):
            self._forwardstatshook(module, inpt, outpt)

        hooks = []

        supported_layers = (torch.nn.Conv2d,
                            torch.nn.MaxPool2d,
                            torch.nn.AvgPool2d,
                            torch.nn.AdaptiveAvgPool2d,
                            torch.nn.AdaptiveMaxPool2d,
                            torch.nn.ConvTranspose2d,
                            torch.nn.UpsamplingBilinear2d,
                            torch.nn.UpsamplingNearest2d)

        # TODO: check whether this works for all pytorch models.
        #
        modules = []
        for mod in self._model_instance.modules():
            if isinstance(mod, supported_layers):
                modules.append(mod)
        
        # Do actual reconstruction information calculation with the linear path
        #
        return self._getreconstructioninformationforlayers(input_shape, modules)

