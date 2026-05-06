using System;
using System.Linq;
using Noushima.Farm;
using Noushima.Island;
using Noushima.Island.Config;
using Noushima.Island.Map;
using Noushima.Island.Simulation;

namespace Sandbox;

internal class Program
{
    public static void Main()
    {
        using var service = new InferenceService();
        var engine = new SimulationEngine(new IslandConfig(), service, new FileMapProvider("Maps/default-map.txt"));
        engine.Initialize();

        for (var tick = 0; tick < 10; tick++)
        {
            engine.RunTick();
            Console.WriteLine($"Tick={engine.TickNumber}, Generation={engine.GenerationNumber}, Alive={engine.Bots.Count(bot => bot.Alive)}, Food={engine.Map.CountEntities(WorldObjectType.Food)}, Poison={engine.Map.CountEntities(WorldObjectType.Poison)}");
        }
    }
}
