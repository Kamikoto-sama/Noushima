using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;

namespace Noushima.Farm;

public sealed class MlpModule : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor>[] layers;

    public MlpModule(BotMlpModel dto) : base(string.Empty)
    {
        layers = new Module<Tensor, Tensor>[dto.Layers.Length];
        var inputSize = dto.InputSize;
        var layersCount = dto.Layers.Length;

        for (var layerIndex = 0; layerIndex < layersCount; layerIndex++)
        {
            var layer = dto.Layers[layerIndex];
            if (layer.Bias.Length != layer.Weights.Length)
                throw new InvalidOperationException($"Layer {layerIndex}: bias size ({layer.Bias.Length}), " +
                                                    $"doesnt match weights count ({layer.Weights.Length})");

            var layerModule = new MlpLayerModule(layer, inputSize);
            layers[layerIndex] = layerModule;
            inputSize = layer.Bias.Length;
        }

        RegisterComponents();
    }

    public override Tensor forward(Tensor input) => Sequential(layers).forward(input);
}

public sealed class MlpLayerModule : Module<Tensor, Tensor>
{
    private readonly Tensor weights;
    private readonly Tensor bias;

    public MlpLayerModule(BotMlpModel.Layer dto, int inputSize) : base(string.Empty)
    {
        var outputSize = dto.Bias.Length;
        var flatWeights = dto.Weights.SelectMany(w => w).ToArray();
        weights = tensor(flatWeights, outputSize, inputSize, device: CPU).transpose(0, 1);
        bias = tensor(dto.Bias, device: CPU);

        RegisterComponents();
    }

    public override Tensor forward(Tensor input) => relu(matmul(input, weights) + bias);
}