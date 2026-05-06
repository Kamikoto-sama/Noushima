using Noushima.Island.Config;

namespace Noushima.Island.Genetics;

public sealed class GenomeFactory(IslandConfig config, Random random)
{
    public Genome CreateRandom(int inputSize, int outputSize)
    {
        var layers = new List<GenomeLayer>();
        var currentInput = inputSize;
        for (var layerIndex = 0; layerIndex < config.InitialHiddenLayerCount; layerIndex++)
        {
            layers.Add(CreateLayer(currentInput, config.InitialHiddenLayerSize));
            currentInput = config.InitialHiddenLayerSize;
        }

        layers.Add(CreateLayer(currentInput, outputSize));
        return new Genome(Guid.NewGuid(), inputSize, layers);
    }

    internal GenomeLayer CreateLayer(int inputSize, int outputSize)
    {
        var weights = new float[outputSize][];
        for (var outputIndex = 0; outputIndex < outputSize; outputIndex++)
        {
            weights[outputIndex] = new float[inputSize];
            for (var inputIndex = 0; inputIndex < inputSize; inputIndex++)
                weights[outputIndex][inputIndex] = NextWeight();
        }

        var bias = new float[outputSize];
        for (var index = 0; index < outputSize; index++)
            bias[index] = NextWeight();

        return new GenomeLayer(weights, bias);
    }

    internal float NextWeight() => (float)(random.NextDouble() * 2d - 1d);
}
