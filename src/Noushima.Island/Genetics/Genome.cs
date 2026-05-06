using Noushima.Farm;

namespace Noushima.Island.Genetics;

public sealed class Genome
{
    public Guid Id { get; }
    public int InputSize { get; }
    public IReadOnlyList<GenomeLayer> Layers { get; }
    public int OutputSize => Layers[^1].OutputSize;
    public int Complexity => Layers.Sum(layer => layer.OutputSize + layer.OutputSize * layer.InputSize);

    public Genome(Guid id, int inputSize, IReadOnlyList<GenomeLayer> layers)
    {
        Id = id;
        InputSize = inputSize;
        Layers = layers.Select(layer => layer.Clone()).ToArray();
    }

    public Genome Clone(Guid? id = null) => new(id ?? Guid.NewGuid(), InputSize, Layers);

    public MlpModel ToMlpModel() => new()
    {
        InputSize = InputSize,
        Layers = Layers.Select(layer => new MlpModel.Layer
        {
            Bias = layer.Bias.ToArray(),
            Weights = layer.Weights.Select(row => row.ToArray()).ToArray(),
        }).ToArray()
    };
}