namespace Noushima.Island.Genetics;

public sealed class GenomeMixer(GenomeMutator mutator, Random random)
{
    public Genome CreateChild(Genome firstParent, Genome secondParent)
    {
        var baseGenome = firstParent.Layers.Count == secondParent.Layers.Count &&
                         firstParent.Layers.Zip(secondParent.Layers, (left, right) =>
                                 left.OutputSize == right.OutputSize && left.InputSize == right.InputSize)
                             .All(match => match)
            ? MixCompatible(firstParent, secondParent)
            : (random.Next(2) == 0 ? firstParent : secondParent).Clone();

        return mutator.Mutate(baseGenome, stronger: false);
    }

    private Genome MixCompatible(Genome firstParent, Genome secondParent)
    {
        var layers = new List<GenomeLayer>();
        for (var layerIndex = 0; layerIndex < firstParent.Layers.Count; layerIndex++)
        {
            var firstLayer = firstParent.Layers[layerIndex];
            var secondLayer = secondParent.Layers[layerIndex];
            var weights = new float[firstLayer.OutputSize][];
            for (var outputIndex = 0; outputIndex < firstLayer.OutputSize; outputIndex++)
            {
                weights[outputIndex] = new float[firstLayer.InputSize];
                for (var inputIndex = 0; inputIndex < firstLayer.InputSize; inputIndex++)
                    weights[outputIndex][inputIndex] = random.Next(2) == 0
                        ? firstLayer.Weights[outputIndex][inputIndex]
                        : secondLayer.Weights[outputIndex][inputIndex];
            }

            var bias = new float[firstLayer.OutputSize];
            for (var outputIndex = 0; outputIndex < firstLayer.OutputSize; outputIndex++)
                bias[outputIndex] = random.Next(2) == 0
                    ? firstLayer.Bias[outputIndex]
                    : secondLayer.Bias[outputIndex];

            layers.Add(new GenomeLayer(weights, bias));
        }

        return new Genome(Guid.NewGuid(), firstParent.InputSize, layers);
    }
}
