namespace Noushima.Farm;

public class BotMlpModel
{
    public int InputSize { get; init; }
    public required Layer[] Layers { get; init; }

    public class Layer
    {
        public required float[] Bias { get; init; }
        public required float[][] Weights { get; init; }
    }
}
