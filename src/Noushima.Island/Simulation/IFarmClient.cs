using Noushima.Island.Genetics;

namespace Noushima.Island.Simulation;

public interface IFarmClient
{
    float[] Infer(Genome genome, float[] input);
    void Forget(Genome genome);
}
