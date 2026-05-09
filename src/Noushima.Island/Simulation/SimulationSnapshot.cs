using Noushima.Island.Map;

namespace Noushima.Island.Simulation;

public class SimulationSnapshot(CellSnapshot[,] map, int generationNumber)
{
    public CellSnapshot[,] Map { get; } = map;
    public int GenerationNumber { get; } = generationNumber;
}

public class CellSnapshot
{
    public required WorldObjectType? Type { get; init; }
    public BotSnapshot? BotSnapshot { get; init; }
}

public class BotSnapshot
{
    public required Direction Direction { get; init; }
    public required int Energy { get; init; }
}
