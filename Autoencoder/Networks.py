import math
import torch.nn as nn
from .Resamplers import InterpolativeUpsampler, InterpolativeDownsampler
from .FusedOperators import BiasedActivation

def MSRInitializer(Layer, ActivationGain=1):
    FanIn = Layer.weight.data.size(1) * Layer.weight.data[0][0].numel()
    Layer.weight.data.normal_(0,  ActivationGain / math.sqrt(FanIn))

    if Layer.bias is not None:
        Layer.bias.data.zero_()
    
    return Layer

class ResidualBlock(nn.Module):
    def __init__(self, InputChannels, CompressionFactor, ReceptiveField):
        super(ResidualBlock, self).__init__()
        
        CompressedChannels = InputChannels // CompressionFactor
        
        self.LinearLayer1 = MSRInitializer(nn.Conv2d(InputChannels, CompressedChannels, kernel_size=1, stride=1, padding=0, bias=False), ActivationGain=BiasedActivation.Gain)
        self.LinearLayer2 = MSRInitializer(nn.Conv2d(CompressedChannels, CompressedChannels, kernel_size=ReceptiveField, stride=1, padding=(ReceptiveField - 1) // 2, padding_mode='reflect', bias=False), ActivationGain=BiasedActivation.Gain)
        self.LinearLayer3 = MSRInitializer(nn.Conv2d(CompressedChannels, InputChannels, kernel_size=1, stride=1, padding=0, bias=False), ActivationGain=0)
        
        self.NonLinearity1 = BiasedActivation(CompressedChannels)
        self.NonLinearity2 = BiasedActivation(CompressedChannels)
        
    def forward(self, x):
        y = self.LinearLayer1(x)
        y = self.LinearLayer2(self.NonLinearity1(y))
        y = self.LinearLayer3(self.NonLinearity2(y))
        
        return x + y
    
def ChannelMixer(InputChannels, OutputChannels):
    return MSRInitializer(nn.Conv2d(InputChannels, OutputChannels, kernel_size=1, stride=1, padding=0, bias=False)) if InputChannels != OutputChannels else nn.Identity()
    
class UpsampleLayer(nn.Module):
    def __init__(self, InputChannels, OutputChannels, ResamplingFilter):
        super(UpsampleLayer, self).__init__()
        
        self.LinearLayer = ChannelMixer(InputChannels, OutputChannels)
        self.Resampler = InterpolativeUpsampler(ResamplingFilter)
        
    def forward(self, x):
        return self.Resampler(self.LinearLayer(x))
    
class DownsampleLayer(nn.Module):
    def __init__(self, InputChannels, OutputChannels, ResamplingFilter):
        super(DownsampleLayer, self).__init__()
        
        self.LinearLayer = ChannelMixer(InputChannels, OutputChannels)
        self.Resampler = InterpolativeDownsampler(ResamplingFilter)
        
    def forward(self, x):
        return self.LinearLayer(self.Resampler(x))
    
class EncoderStage(nn.Module):
    def __init__(self, InputChannels, OutputChannels, NumberOfBlocks, CompressionFactor, ReceptiveField, ResamplingFilter=None):
        super(EncoderStage, self).__init__()
        
        TransitionLayer = ChannelMixer(InputChannels, OutputChannels) if ResamplingFilter is None else DownsampleLayer(InputChannels, OutputChannels, ResamplingFilter)
        self.Layers = nn.ModuleList([ResidualBlock(InputChannels, CompressionFactor, ReceptiveField) for _ in range(NumberOfBlocks)] + [TransitionLayer])
        
    def forward(self, x):
        for Layer in self.Layers:
            x = Layer(x)
        
        return x
        
class DecoderStage(nn.Module):
    def __init__(self, InputChannels, OutputChannels, NumberOfBlocks, CompressionFactor, ReceptiveField, ResamplingFilter=None):
        super(DecoderStage, self).__init__()
        
        TransitionLayer = ChannelMixer(InputChannels, OutputChannels) if ResamplingFilter is None else UpsampleLayer(InputChannels, OutputChannels, ResamplingFilter)
        self.Layers = nn.ModuleList([TransitionLayer] + [ResidualBlock(OutputChannels, CompressionFactor, ReceptiveField) for _ in range(NumberOfBlocks)])
        
    def forward(self, x):
        for Layer in self.Layers:
            x = Layer(x)
        
        return x
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# import torch

# ResamplingFilter = [1, 2, 1]

# x = torch.zeros(1, 32, 128, 128)
# m = EncoderStage(32, 64, 2, 4, 3, ResamplingFilter)
# n = DecoderStage(32, 16, 2, 4, 3, ResamplingFilter)

# print(m(x).shape)
# print(n(x).shape)