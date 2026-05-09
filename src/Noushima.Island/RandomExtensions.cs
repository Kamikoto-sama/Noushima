namespace Noushima.Island;

public static class RandomExtensions
{
    public static bool CheckChance(this Random random, float chance) => random.NextDouble() < chance;
}