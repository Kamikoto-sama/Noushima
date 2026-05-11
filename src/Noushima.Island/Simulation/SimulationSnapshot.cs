using Noushima.Island.Map;

namespace Noushima.Island.Simulation;

public sealed class SimulationSnapshot(CellSnapshot[,] map, int generationNumber, int botsAlive, float bestEnergy)
{
    public CellSnapshot[,] Map { get; } = map;
    public int GenerationNumber { get; } = generationNumber;
    public int BotsAlive { get; } = botsAlive;
    public float BestEnergy { get; } = bestEnergy;
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
