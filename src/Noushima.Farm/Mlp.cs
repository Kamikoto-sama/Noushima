namespace Noushima.Farm;

public class Mlp
{
    public int InputSize { get; init; }
    public required MlpLayer[] Layers { get; init; }
}

public class MlpLayer
{
    public required float[] Bias { get; init; }
    public required float[][] Weights { get; init; }
}