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
    void Execute(Bot bot, ReadOnlySpan<float> intentions, BotActionContext context);
}

public sealed class BotActionContext(WorldMap map, IslandConfig config)
{
    public WorldMap Map { get; } = map;
    public IslandConfig Config { get; } = config;
}
