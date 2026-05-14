using Noushima.Island.Map;

namespace Noushima.Island.Simulation;

public sealed class SimulationSnapshot
{
    public CellSnapshot[,] Map { get; init; } = new CellSnapshot[0,0];
    public int GenerationNumber { get; init; }
    public float LongestGeneration { get; init; }
    public int BotsAlive { get; init; }
    public float BestEnergy { get; init; }
}

public sealed class CellSnapshot
{
    public required WorldObjectType? Type { get; init; }
    public BotSnapshot? BotSnapshot { get; init; }
}

public sealed class BotSnapshot
{
    public required Direction Direction { get; init; }
    public required int Energy { get; init; }
}