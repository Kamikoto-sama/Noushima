namespace Noushima.Island.Genetics;

public sealed class Genome
{
    public Guid Id { get; }
    public GenomeLayer[] Layers { get; }
    public int Complexity { get; }

    public Genome(Guid id, IReadOnlyList<GenomeLayer> layers, int complexity)
    {
        Id = id;
        Complexity = complexity;
        Layers = layers.Select(layer => layer.Clone()).ToArray();
    }

    public Genome Clone(Guid? id = null) => new(id ?? Id, Layers, Complexity);
}

public sealed class GenomeLayer
{
    public GenomeNeuron[] Neurons { get; }
    public int Size => Neurons.Length;
    public int InputSize { get; }

    public GenomeLayer(IEnumerable<GenomeNeuron> neurons)
    {
        Neurons = neurons.ToArray();
        InputSize = Neurons.First().Weights.Length;
    }

    public GenomeLayer Clone() => new(Neurons.Select(n => n.Clone()));
}

public class GenomeNeuron(IReadOnlyList<float> weights, float bias)
{
    public float[] Weights { get; } = weights.ToArray();
    public float Bias { get; set; } = bias;
    public int LinksCount { get; } = weights.Count(w => w != 0);

    public GenomeNeuron Clone() => new(Weights, Bias);
}