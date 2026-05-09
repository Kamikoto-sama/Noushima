using Noushima.Island.Simulation;

namespace Noushima.Island.Genetics;

public sealed class GenomeMutator(
    IslandConfig config,
    Random random,
    GenomeFactory genomeFactory,
    GenomeComplexityCalculator complexityCalculator)
{
    public Genome Mutate(Genome source, bool stronger)
    {
        var layers = source.Layers.Select(layer => layer.Clone()).ToList();
        var multiplier = stronger ? config.StrongMutationMultiplier : 1f;

        MutateWeights(layers, multiplier);
        if (random.CheckChance(config.AddNeuronChance * multiplier) && layers.Count > 1)
            AddNeuron(layers.Take(layers.Count - 1).ToArray());
        else if (random.CheckChance(config.RemoveNeuronChance * multiplier))
            RemoveNeuron(layers);
        if (random.CheckChance(config.AddLayerChance * multiplier))
            AddLayer(layers);
        else if (random.CheckChance(config.RemoveLayerChance * multiplier))
            RemoveLayer(layers);

        var complexity = complexityCalculator.Calculate(layers);
        return new Genome(Guid.NewGuid(), layers, complexity);
    }

    private void MutateWeights(IList<GenomeLayer> layers, float multiplier)
    {
        foreach (var neuron in layers.SelectMany(layer => layer.Neurons))
        {
            if (random.CheckChance(config.BiasMutationChance * multiplier))
                neuron.Bias = MutateWeight(neuron.Bias, multiplier);
            for (var w = 0; w < neuron.Weights.Length; w++)
                if (random.CheckChance(config.WeightMutationChance * multiplier))
                    neuron.Weights[w] = MutateWeight(neuron.Weights[w], multiplier);
                else if (random.CheckChance(config.RemoveLinkChance * multiplier))
                    neuron.Weights[w] = 0;
        }
    }

    private void AddNeuron(IList<GenomeLayer> layers)
    {
        var layerIndex = random.Next(layers.Count - 1);
        var layer = layers[layerIndex];
        if (layer.Size >= config.MaxHiddenLayerSize)
            return;

        var neurons = layer.Neurons.ToList();
        neurons.Add(genomeFactory.CreateNeuron(layer.InputSize, false));
        layers[layerIndex] = new GenomeLayer(neurons);

        var nextLayer = layers[layerIndex + 1];
        layers[layerIndex + 1] = new GenomeLayer(nextLayer.Neurons.Select(AppendWeight));
    }

    private void RemoveNeuron(IList<GenomeLayer> layers)
    {
        if (layers.Count < 2)
            return;

        var candidates = Enumerable.Range(0, layers.Count - 1)
            .Where(index => layers[index].Size > config.MinHiddenLayerSize)
            .ToArray();
        if (candidates.Length == 0)
            return;

        var layerIndex = candidates[random.Next(candidates.Length)];
        var layer = layers[layerIndex];
        var neuronIndex = random.Next(layer.Size);
        layers[layerIndex] = new GenomeLayer(layer.Neurons.Where((_, i) => i != neuronIndex));

        var nextLayer = layers[layerIndex + 1];
        layers[layerIndex + 1] = new GenomeLayer(nextLayer.Neurons.Select(neuron => RemoveWeight(neuron, neuronIndex)));
    }

    private void AddLayer(IList<GenomeLayer> layers)
    {
        var hiddenCount = layers.Count - 1;
        if (hiddenCount >= config.MaxHiddenLayersCount)
            return;

        var insertIndex = hiddenCount;
        var previousOutputSize = layers[insertIndex - 1].Size;
        var nextOutputSize = layers[insertIndex].Size;
        var newLayerSize = random.Next(config.MinHiddenLayerSize, config.MaxHiddenLayerSize + 1);
        layers.Insert(insertIndex, genomeFactory.CreateLayer(previousOutputSize, newLayerSize, false));
        layers[insertIndex + 1] = genomeFactory.CreateLayer(newLayerSize, nextOutputSize, true);
    }

    private void RemoveLayer(IList<GenomeLayer> layers)
    {
        var hiddenCount = layers.Count - 1;
        if (hiddenCount <= 1)
            return;

        var removeIndex = random.Next(hiddenCount);
        var previousOutputSize = removeIndex == 0 ? layers[0].InputSize : layers[removeIndex - 1].Size;
        var nextOutputSize = layers[removeIndex + 1].Size;
        layers.RemoveAt(removeIndex);
        layers[removeIndex] = genomeFactory.CreateLayer(
            previousOutputSize,
            nextOutputSize,
            full: removeIndex == 0 || removeIndex == layers.Count - 1);
    }

    private float MutateWeight(float weight, float scale)
    {
        if (weight == 0)
            return 0;
        var delta = genomeFactory.NextWeight() * scale;
        return Math.Clamp(weight + delta, -1f, 1f);
    }

    private GenomeNeuron AppendWeight(GenomeNeuron neuron)
    {
        var weights = new float[neuron.Weights.Length + 1];
        Array.Copy(neuron.Weights, weights, neuron.Weights.Length);
        weights[^1] = random.CheckChance(config.AddLinkChance) ? genomeFactory.NextWeight() : 0f;
        return new GenomeNeuron(weights, neuron.Bias);
    }

    private static GenomeNeuron RemoveWeight(GenomeNeuron neuron, int index)
        => new(RemoveAt(neuron.Weights, index), neuron.Bias);

    private static float[] RemoveAt(float[] values, int index) => values.Where((_, i) => i != index).ToArray();
}