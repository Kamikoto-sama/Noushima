namespace FarmApi;

public class MlpDto
{
    public int Input { get; init; }
    public required MlpLayerDto[] Layers { get; init; }
}

public class MlpLayerDto
{
    public required float[] Bias { get; init; }
    public required float[][] Weights { get; init; }
}