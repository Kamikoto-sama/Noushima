using Noushima.Island.Entities;
using Noushima.Island.Map;
using Noushima.Island.Simulation;

namespace Noushima.Island.Actions;

public interface IBotAction
{
    string Name { get; }
    int Order { get; }
    float Cost { get; }
    int Size { get; }
    void Execute(Bot bot, ReadOnlySpan<float> intentions, BotActionExecutionContext context);
}

public sealed class BotActionExecutionContext(WorldMap map, IslandConfig config)
{
    public WorldMap Map { get; } = map;
    public IslandConfig Config { get; } = config;

    public required Random Random { get; init; }

    public required Action<Bot> AddBot { get; init; }

}
