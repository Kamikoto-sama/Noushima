using Noushima.Island.Simulation;

namespace Noushima.Island.Genetics;

public class GenomeComplexityCalculator(IslandConfig config)
{
    public int Calculate(IReadOnlyList<GenomeLayer> layers)
    {
        var layersFactor = (layers.Count - 1) * config.ComplexityLayersCountFactor;
        var neuronsFactor = layers
            .Take(layers.Count - 1)
            .Sum(l => l.Size) * config.ComplexityNeuronsCountFactor;
        var linksFactor = layers
            .Sum(l => l.Neurons.Sum(n => n.LinksCount)) * config.ComplexityLinksCountFactor;

        var complexity = (layersFactor + neuronsFactor + linksFactor) * config.ComplexityDrainFactor;
        return (int)MathF.Round(complexity);
    }
}