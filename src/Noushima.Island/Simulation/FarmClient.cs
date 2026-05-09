using Noushima.Farm;
using Noushima.Island.Genetics;

namespace Noushima.Island.Simulation;

public sealed class FarmClient(IInferenceService inferenceService) : IFarmClient
{
    private readonly HashSet<Guid> loadedModels = [];

    public float[] Infer(Genome genome, float[] input)
    {
        if (loadedModels.Add(genome.Id))
            inferenceService.LoadModel(genome.Id, ToMlpModel(genome));

        return inferenceService.Infer(genome.Id, input);
    }

    public void Forget(Genome genome)
    {
        if (!loadedModels.Remove(genome.Id))
            return;

        inferenceService.RemoveModel(genome.Id);
    }

    private static MlpModel ToMlpModel(Genome genome)
    {
        return new MlpModel
        {
            InputSize = genome.Layers[0].InputSize,
            Layers =
            [
                ..genome.Layers.Select(layer => new MlpModel.Layer
                {
                    Bias = [..layer.Neurons.Select(neuron => neuron.Bias)],
                    Weights = [..layer.Neurons.Select(neuron => neuron.Weights)],
                }),
            ],
        };
    }
}
