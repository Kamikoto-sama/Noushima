using Noushima.Island.Entities;
using Noushima.Island.Map;

namespace Noushima.Island.Actions;

internal sealed class EatBotAction : IBotAction
{
    public string Name => "eat";

    public int Order => 2;

    public float Cost => 0.8f;

    public int Size => 1;

    public void Execute(Bot bot, ReadOnlySpan<float> intentions, BotActionContext context)
    {
        if (intentions[0] <= 0f)
            return;

        var targetPosition = context.Map.GetAdjacentPosition(bot.Position, bot.Rotation);
        var targetCell = context.Map.GetCellOrDefault(targetPosition);
        if (targetCell?.Entity is not Food and not Poison)
            return;

        var delta = targetCell.Entity.Type == WorldObjectType.Food
            ? context.Config.FoodEnergyGain
            : -context.Config.PoisonEnergyPenalty;

        context.Map.SetEntity(targetPosition.X, targetPosition.Y, null);
        bot.ChangeEnergy(delta, context.Config.MaxBotEnergy);
    }
}
