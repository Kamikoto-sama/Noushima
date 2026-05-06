using Noushima.Farm;
using Noushima.Island.Genetics;

namespace Noushima.Island.Simulation;

public sealed class FarmClient(IInferenceService inferenceService) : IFarmClient
{
    private readonly HashSet<Guid> loadedModels = [];

    public float[] Infer(Genome genome, float[] input)
    {
        if (loadedModels.Add(genome.Id))
            inferenceService.LoadModel(genome.Id, genome.ToMlpModel());

        return inferenceService.Infer(genome.Id, input);
    }

    public void Forget(Genome genome)
    {
        if (!loadedModels.Remove(genome.Id))
            return;

        inferenceService.RemoveModel(genome.Id);
    }
}
