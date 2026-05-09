using Noushima.Island.Entities;
using Noushima.Island.Map;
using Noushima.Island.Simulation;

namespace Noushima.Island.Environment;

public sealed class EnvParamRegistry
{
    private readonly IEnvParam[] envParams;
    public int TotalSize => envParams.Sum(param => param.Size);

    public EnvParamRegistry(IslandConfig config, int outputsSize)
    {
        envParams =
        [
            new RotationEnvParam(),
            new NeighborTypesEnvParam(),
            new EnergyEnvParam(config),
            new EnergyDiffEnvParam(config),
            new LastOutputsEnvParam(outputsSize),
        ];
    }

    public float[] Build(Bot bot, WorldMap map)
    {
        var result = new float[TotalSize];
        var offset = 0;
        foreach (var envParam in envParams)
        {
            envParam.Fill(bot, map, result.AsSpan(offset, envParam.Size));
            offset += envParam.Size;
        }

        return result;
    }

    private sealed class RotationEnvParam : IEnvParam
    {
        public int Size => 1;
        public void Fill(Bot bot, WorldMap map, Span<float> target) => target[0] = bot.Rotation.Normalize();
    }

    private sealed class NeighborTypesEnvParam : IEnvParam
    {
        public int Size => 8;
        public void Fill(Bot bot, WorldMap map, Span<float> target)
        {
            foreach (var direction in Enum.GetValues<Direction>())
            {
                var position = map.GetAdjacentPosition(bot.Position, direction);
                var entity = map.GetCellOrDefault(position)?.Entity;
                target[(int)direction] = entity?.Type switch
                {
                    null => 0f,
                    WorldObjectType.Wall => -1f,
                    WorldObjectType.Poison => -0.5f,
                    WorldObjectType.Food => 0.5f,
                    WorldObjectType.Bot => 1f,
                    _ => 0f,
                };
            }
        }
    }

    private sealed class EnergyEnvParam(IslandConfig config) : IEnvParam
    {
        public int Size => 1;
        public void Fill(Bot bot, WorldMap map, Span<float> target) =>
            target[0] = Normalize(bot.Energy, config.MaxBotEnergy);
    }

    private sealed class EnergyDiffEnvParam(IslandConfig config) : IEnvParam
    {
        public int Size => 1;
        public void Fill(Bot bot, WorldMap map, Span<float> target) =>
            target[0] = Math.Clamp(bot.EnergyDiff / config.MaxBotEnergy, -1f, 1f);
    }

    private sealed class LastOutputsEnvParam(int outputsSize) : IEnvParam
    {
        public int Size => outputsSize;
        public void Fill(Bot bot, WorldMap map, Span<float> target)
        {
            for (var index = 0; index < outputsSize; index++)
                target[index] = Math.Clamp(bot.LastOutputs[index], -1f, 1f);
        }
    }

    private static float Normalize(float value, float max) => Math.Clamp((value / max) * 2f - 1f, -1f, 1f);
}
