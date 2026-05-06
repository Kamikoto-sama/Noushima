using Noushima.Island.Entities;
using Noushima.Island.Map;

namespace Noushima.Island.Actions;

internal sealed class MoveBotAction() : DirectionalScalarAction("move")
{
    public override int Order => 3;

    public override float Cost => 1f;

    protected override void Execute(Bot bot, Direction relativeDirection, BotActionExecutionContext context)
    {
        var moveDirection = bot.Rotation.ApplyRelative(relativeDirection);
        var targetPosition = context.Map.GetAdjacentPosition(bot.Position, moveDirection);
        var targetCell = context.Map.GetCellOrDefault(targetPosition);
        if (targetCell?.Entity is Wall or Bot)
            return;

        var targetType = targetCell?.Entity?.Type;
        if (!context.Map.MoveEntity(bot, targetPosition.X, targetPosition.Y))
            return;

        bot.ChangeEnergy(targetType switch
        {
            WorldObjectType.Food => context.Config.FoodEnergyGain,
            WorldObjectType.Poison => -context.Config.PoisonEnergyPenalty,
            _ => 0f,
        }, context.Config.MaxBotEnergy);
    }
}
