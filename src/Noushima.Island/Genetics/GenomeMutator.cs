using Noushima.Island.Config;

namespace Noushima.Island.Genetics;

public sealed class GenomeMutator(IslandConfig config, Random random, GenomeFactory genomeFactory)
{
    public Genome Mutate(Genome source, bool stronger)
    {
        var layers = source.Layers.Select(layer => layer.Clone()).ToList();
        var multiplier = stronger ? config.StrongMutationMultiplier : 1f;

        MutateWeights(layers, multiplier);
        if (random.NextDouble() < config.AddNeuronChance * multiplier)
            AddNeuron(layers);
        if (random.NextDouble() < config.RemoveNeuronChance * multiplier)
            RemoveNeuron(layers);
        if (random.NextDouble() < config.AddLayerChance * multiplier)
            AddLayer(layers);
        if (random.NextDouble() < config.RemoveLayerChance * multiplier)
            RemoveLayer(layers);

        return new Genome(Guid.NewGuid(), source.InputSize, layers);
    }

    private void MutateWeights(IList<GenomeLayer> layers, float multiplier)
    {
        foreach (var layer in layers)
        {
            for (var outputIndex = 0; outputIndex < layer.OutputSize; outputIndex++)
            {
                layer.Bias[outputIndex] = Clamp(layer.Bias[outputIndex] + NextDelta(config.BiasMutationScale * multiplier));
                for (var inputIndex = 0; inputIndex < layer.InputSize; inputIndex++)
                    layer.Weights[outputIndex][inputIndex] = Clamp(layer.Weights[outputIndex][inputIndex] + NextDelta(config.WeightMutationScale * multiplier));
            }
        }
    }

    private void AddNeuron(IList<GenomeLayer> layers)
    {
        if (layers.Count < 2)
            return;

        var layerIndex = random.Next(layers.Count - 1);
        var layer = layers[layerIndex];
        if (layer.OutputSize >= config.MaxHiddenLayerSize)
            return;

        var bias = layer.Bias;
        Array.Resize(ref bias, layer.OutputSize + 1);
        bias[^1] = genomeFactory.NextWeight();
        layer.Bias = bias;

        var weights = layer.Weights;
        Array.Resize(ref weights, layer.OutputSize + 1);
        weights[^1] = new float[layer.InputSize];
        for (var inputIndex = 0; inputIndex < layer.InputSize; inputIndex++)
            weights[^1][inputIndex] = genomeFactory.NextWeight();
        layer.Weights = weights;

        var nextLayer = layers[layerIndex + 1];
        for (var outputIndex = 0; outputIndex < nextLayer.OutputSize; outputIndex++)
        {
            var nextWeights = nextLayer.Weights[outputIndex];
            Array.Resize(ref nextWeights, nextLayer.InputSize + 1);
            nextWeights[^1] = genomeFactory.NextWeight();
            nextLayer.Weights[outputIndex] = nextWeights;
        }
    }

    private void RemoveNeuron(IList<GenomeLayer> layers)
    {
        if (layers.Count < 2)
            return;

        var candidates = Enumerable.Range(0, layers.Count - 1)
            .Where(index => layers[index].OutputSize > config.MinHiddenLayerSize)
            .ToArray();
        if (candidates.Length == 0)
            return;

        var layerIndex = candidates[random.Next(candidates.Length)];
        var layer = layers[layerIndex];
        var neuronIndex = random.Next(layer.OutputSize);
        layer.Bias = RemoveAt(layer.Bias, neuronIndex);
        layer.Weights = RemoveAt(layer.Weights, neuronIndex);

        var nextLayer = layers[layerIndex + 1];
        for (var outputIndex = 0; outputIndex < nextLayer.OutputSize; outputIndex++)
            nextLayer.Weights[outputIndex] = RemoveAt(nextLayer.Weights[outputIndex], neuronIndex);
    }

    private void AddLayer(IList<GenomeLayer> layers)
    {
        var hiddenCount = layers.Count - 1;
        if (hiddenCount >= config.MaxHiddenLayerCount)
            return;

        var insertIndex = random.Next(layers.Count);
        var previousOutputSize = insertIndex == 0 ? layers[0].InputSize : layers[insertIndex - 1].OutputSize;
        var nextOutputSize = layers[insertIndex].OutputSize;
        var newLayerSize = random.Next(config.MinHiddenLayerSize, config.InitialHiddenLayerSize + 1);
        layers.Insert(insertIndex, genomeFactory.CreateLayer(previousOutputSize, newLayerSize));
        layers[insertIndex + 1] = genomeFactory.CreateLayer(newLayerSize, nextOutputSize);
    }

    private void RemoveLayer(IList<GenomeLayer> layers)
    {
        var hiddenCount = layers.Count - 1;
        if (hiddenCount <= 1)
            return;

        var removeIndex = random.Next(hiddenCount);
        var previousOutputSize = removeIndex == 0 ? layers[0].InputSize : layers[removeIndex - 1].OutputSize;
        var nextOutputSize = layers[removeIndex + 1].OutputSize;
        layers.RemoveAt(removeIndex);
        layers[removeIndex] = genomeFactory.CreateLayer(previousOutputSize, nextOutputSize);
    }

    private float NextDelta(float scale) => (float)(random.NextDouble() * 2d - 1d) * scale;

    private static float Clamp(float value) => Math.Clamp(value, -1f, 1f);

    private static float[] RemoveAt(float[] values, int index) => values.Where((_, i) => i != index).ToArray();

    private static float[][] RemoveAt(float[][] values, int index) => values.Where((_, i) => i != index).ToArray();
}
