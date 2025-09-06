using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;

namespace FarmApi;

public sealed class MlpModule : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor>[] layers;

    public MlpModule(MlpDto dto) : base(string.Empty)
    {
        layers = new Module<Tensor, Tensor>[dto.Layers.Length];
        var inputSize = dto.Input;
        var layersCount = dto.Layers.Length;

        for (var layerIndex = 0; layerIndex < layersCount; layerIndex++)
        {
            var layer = new MlpLayerModule(dto.Layers[layerIndex], inputSize);
            layers[layerIndex] = layer;
            inputSize = dto.Layers[layerIndex].Bias.Length;
        }

        RegisterComponents();
    }

    public override Tensor forward(Tensor input) => Sequential(layers).forward(input);
}

public sealed class MlpLayerModule : Module<Tensor, Tensor>
{
    private readonly Tensor weights;
    private readonly Tensor bias;

    public MlpLayerModule(MlpLayerDto dto, int inputSize) : base(string.Empty)
    {
        var outputSize = dto.Bias.Length;
        var flatWeights = dto.Weights.SelectMany(w => w).ToArray();
        weights = tensor(flatWeights, outputSize, inputSize, device: CUDA).transpose(0, 1);
        bias = tensor(dto.Bias, device: CUDA);

        RegisterComponents();
    }

    public override Tensor forward(Tensor input) => relu(matmul(input, weights) + bias);
}