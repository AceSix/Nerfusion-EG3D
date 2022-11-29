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
    
class Encoder(nn.Module):
    def __init__(self, InputChannels, OutputChannels, StageWidths=[192, 384, 768], BlocksPerStage=[2, 2, 2], CompressionFactor=4, ReceptiveField=3, ResamplingFilter=[1, 2, 1]):
        super(Encoder, self).__init__()
        
        MainLayers = [EncoderStage(StageWidths[x], StageWidths[x + 1], BlocksPerStage[x], CompressionFactor, ReceptiveField, ResamplingFilter) for x in range(len(StageWidths) - 1)]
        MainLayers += [EncoderStage(StageWidths[-1], OutputChannels, BlocksPerStage[-1], CompressionFactor, ReceptiveField)]
        
        self.Head = ChannelMixer(InputChannels, StageWidths[0])
        self.MainLayers = nn.ModuleList(MainLayers)
        
    def forward(self, x):
        x = self.Head(x)
        
        for Layer in self.MainLayers:
            x = Layer(x)
            
        return x
    
class Decoder(nn.Module):
    def __init__(self, InputChannels, OutputChannels, StageWidths=[768, 384, 192], BlocksPerStage=[2, 2, 2], CompressionFactor=4, ReceptiveField=3, ResamplingFilter=[1, 2, 1]):
        super(Decoder, self).__init__()
        
        MainLayers = [DecoderStage(InputChannels, StageWidths[0], BlocksPerStage[0], CompressionFactor, ReceptiveField)]
        MainLayers += [DecoderStage(StageWidths[x], StageWidths[x + 1], BlocksPerStage[x + 1], CompressionFactor, ReceptiveField, ResamplingFilter) for x in range(len(StageWidths) - 1)]
        
        self.MainLayers = nn.ModuleList(MainLayers)
        self.Tail = ChannelMixer(StageWidths[-1], OutputChannels)
        
    def forward(self, x):
        for Layer in self.MainLayers:
            x = Layer(x)
            
        return self.Tail(x)
    
class Autoencoder(nn.Module):
    def __init__(self, InputChannels, BottleneckChannels, StageWidths=[192, 384, 768], BlocksPerStage=[2, 2, 2], CompressionFactor=4, ReceptiveField=3, ResamplingFilter=[1, 2, 1]):
        super(Autoencoder, self).__init__()
        
        self.EncoderLayer = Encoder(InputChannels, BottleneckChannels, StageWidths, BlocksPerStage, CompressionFactor, ReceptiveField, ResamplingFilter)
        self.DecoderLayer = Decoder(BottleneckChannels, InputChannels, [*reversed(StageWidths)], [*reversed(BlocksPerStage)], CompressionFactor, ReceptiveField, ResamplingFilter)
        
    def forward(self, x):
        return self.DecoderLayer(self.EncoderLayer(x))