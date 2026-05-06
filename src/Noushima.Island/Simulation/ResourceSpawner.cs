using System.Drawing;
using Noushima.Island.Config;
using Noushima.Island.Entities;
using Noushima.Island.Map;

namespace Noushima.Island.Simulation;

public sealed class ResourceSpawner(IslandConfig config, Random random)
{
    public void Initialize(WorldMap map)
    {
        FillUntil(map, WorldObjectType.Food, config.MaxFoodCount);
        FillUntil(map, WorldObjectType.Poison, config.MaxPoisonCount);
    }

    public void Update(WorldMap map)
    {
        if (map.CountEntities(WorldObjectType.Food) < config.MaxFoodCount && ShouldSpawn())
            SpawnOne(map, WorldObjectType.Food);

        if (map.CountEntities(WorldObjectType.Poison) < config.MaxPoisonCount && ShouldSpawn())
            SpawnOne(map, WorldObjectType.Poison);
    }

    private void FillUntil(WorldMap map, WorldObjectType type, int targetCount)
    {
        while (map.CountEntities(type) < targetCount)
        {
            if (!SpawnOne(map, type))
                break;
        }
    }

    private bool SpawnOne(WorldMap map, WorldObjectType type)
    {
        var cell = map.GetRandomEmptyCell();
        if (cell is null)
            return false;

        map.SetEntity(cell.Position.X, cell.Position.Y, Create(type, cell.Position));
        return true;
    }

    private bool ShouldSpawn() => random.NextDouble() <= config.SpawnChancePerTick;

    private static IEntity Create(WorldObjectType type, Point position) => type switch
    {
        WorldObjectType.Food => new Food(position),
        WorldObjectType.Poison => new Poison(position),
        _ => throw new InvalidOperationException($"Unsupported spawn type {type}.")
    };
}
