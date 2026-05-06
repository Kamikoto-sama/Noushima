using Noushima.Island.Entities;
using Noushima.Island.Map;

namespace Noushima.Island.Environment;

public interface IEnvParam
{
    int Size { get; }
    void Fill(Bot bot, WorldMap map, Span<float> target);
}
