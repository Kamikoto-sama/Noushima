using Noushima.Island.Api.Contracts;
using Noushima.Island.Simulation;

namespace Noushima.Island.Api.Services;

public sealed class SimulationStateProvider
{
    private readonly SimulationEngine engine;
    private readonly SimulationSpeedControl speedControl;

    public SimulationStateProvider(SimulationEngine engine, SimulationSpeedControl speedControl)
    {
        this.engine = engine;
        this.speedControl = speedControl;
    }

    public SimulationStateDto GetState()
    {
        engine.EnsureInitialized();
        var snapshot = engine.Snapshot ?? throw new InvalidOperationException("Simulation snapshot is unavailable.");
        var map = snapshot.Map;
        var width = map.GetLength(0);
        var height = map.GetLength(1);
        var cells = new List<CellDto>(width * height);

        for (var x = 0; x < width; x++)
        {
            for (var y = 0; y < height; y++)
            {
                var cell = map[x, y];
                if (cell.Type is null)
                    continue;

                cells.Add(new CellDto
                {
                    X = x,
                    Y = y,
                    Type = cell.Type.Value.ToString(),
                    Energy = cell.BotSnapshot?.Energy,
                    Direction = cell.BotSnapshot?.Direction.ToString(),
                });
            }
        }

        return new SimulationStateDto
        {
            Generation = snapshot.GenerationNumber,
            LongestGeneration = snapshot.LongestGeneration,
            BotsAlive = snapshot.BotsAlive,
            BestEnergy = snapshot.BestEnergy,
            Mode = speedControl.Mode,
            Width = width,
            Height = height,
            Cells = cells,
        };
    }
}
