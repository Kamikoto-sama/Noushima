namespace Noushima.Island.Genetics;

public sealed class GenomeLayer
{
    public float[][] Weights { get; set; }
    public float[] Bias { get; set; }
    public int InputSize => Weights[0].Length;
    public int OutputSize => Bias.Length;

    public GenomeLayer(float[][] weights, float[] bias)
    {
        Weights = weights.Select(row => row.ToArray()).ToArray();
        Bias = bias.ToArray();
    }

    public GenomeLayer Clone() => new(Weights, Bias);
}