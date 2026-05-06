using System.Drawing;
using Noushima.Island.Entities;
using Noushima.Island.Map;

namespace Noushima.Island.Actions;

internal sealed class ReproduceBotAction : IBotAction
{
    public string Name => "reproduce";
    public int Order => 4;
    public float Cost => 0f;
    public int Size => 1;

    public void Execute(Bot bot, ReadOnlySpan<float> intentions, BotActionExecutionContext context)
    {
        if (intentions[0] <= 0f)
            return;

        bot.SetWantsToReproduce();
        if (bot.Energy < context.Config.ReproductionEnergyThreshold || !IsCardinal(bot.Rotation))
            return;

        var partnerPosition = context.Map.GetAdjacentPosition(bot.Position, bot.Rotation);
        var partner = context.Map.GetCellOrDefault(partnerPosition)?.Entity as Bot;
        if (partner is null ||
            !partner.Alive ||
            !partner.WantsToReproduce ||
            partner.Energy < context.Config.ReproductionEnergyThreshold ||
            partner.Rotation != bot.Rotation.Opposite() ||
            AreParentAndChild(bot, partner))
        {
            return;
        }

        var spawnPosition = FindChildSpawnPosition(context.Map, context.Random, bot.Position, partner.Position);
        if (spawnPosition is null)
            return;

        bot.SetEnergy(bot.Energy * (1f - context.Config.ReproductionEnergyCostFactor), context.Config.MaxBotEnergy);
        partner.SetEnergy(partner.Energy * (1f - context.Config.ReproductionEnergyCostFactor), context.Config.MaxBotEnergy);

        var child = context.CreateChild(bot, partner);
        context.Map.SetEntity(spawnPosition.Value.X, spawnPosition.Value.Y, child);
        context.AddBot(child);
    }

    private static bool IsCardinal(Direction direction) => 
        direction is Direction.Up or Direction.Right or Direction.Down or Direction.Left;

    private static bool AreParentAndChild(Bot bot, Bot partner) =>
        bot.HasParent(partner.Id) || partner.HasParent(bot.Id);

    private static Point? FindChildSpawnPosition(WorldMap map, Random random, Point firstParent, Point secondParent)
    {
        var candidates = new HashSet<Point>();
        foreach (var origin in new[] { firstParent, secondParent })
        {
            foreach (var direction in Enum.GetValues<Direction>())
            {
                var candidate = map.GetAdjacentPosition(origin, direction);
                if (map.GetCellOrDefault(candidate)?.IsEmpty == true)
                    candidates.Add(candidate);
            }
        }

        if (candidates.Count == 0)
            return null;

        return candidates.ElementAt(random.Next(candidates.Count));
    }
}
