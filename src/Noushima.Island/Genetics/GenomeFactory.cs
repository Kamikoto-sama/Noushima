using Noushima.Island.Simulation;

namespace Noushima.Island.Genetics;

public sealed class GenomeFactory(IslandConfig config, Random random, GenomeComplexityCalculator complexityCalculator)
{
    public Genome CreateNewGenome(int inputSize, int neuronsCount)
    {
        var layers = new List<GenomeLayer>();
        var currentInput = inputSize;
        var layersCount = random.Next(config.MinHiddenLayersCount, config.MaxHiddenLayersCount + 1);
        for (var layerIndex = 0; layerIndex < layersCount; layerIndex++)
        {
            var layerSize = random.Next(config.MinHiddenLayerSize, config.MaxHiddenLayerSize + 1);
            var genomeLayer = CreateLayer(currentInput, layerSize, layerIndex == 0);
            layers.Add(genomeLayer);
            currentInput = genomeLayer.Size;
        }

        layers.Add(CreateLayer(currentInput, neuronsCount, true));
        var complexity = complexityCalculator.Calculate(layers);
        return new Genome(Guid.NewGuid(), layers, complexity);
    }

    public GenomeLayer CreateLayer(int inputSize, int neuronsCount, bool full)
    {
        var neurons = new GenomeNeuron[neuronsCount];
        for (var neuronIndex = 0; neuronIndex < neuronsCount; neuronIndex++)
            neurons[neuronIndex] = CreateNeuron(inputSize, full);
        return new GenomeLayer(neurons);
    }

    public GenomeNeuron CreateNeuron(int inputSize, bool full)
    {
        var weights = new float[inputSize];
        for (var w = 0; w < inputSize; w++)
            weights[w] = random.CheckChance(config.AddLinkChance) || full ? NextWeight() : 0;
        if (weights.All(w => w == 0))
            weights[random.Next(0, inputSize)] = NextWeight();
        return new GenomeNeuron(weights, 0);
    }

    public float NextWeight() => (float)(random.NextDouble() * 2d - 1d);
}