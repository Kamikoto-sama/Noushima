using Noushima.Island.Entities;
using Noushima.Island.Map;

namespace Noushima.Island.Actions;

internal sealed class PeelBotAction : IBotAction
{
    public string Name => "peel";

    public int Order => 1;

    public float Cost => 0.8f;

    public int Size => 1;

    public void Execute(Bot bot, ReadOnlySpan<float> intentions, BotActionContext context)
    {
        if (intentions[0] <= 0f)
            return;

        var targetPosition = context.Map.GetAdjacentPosition(bot.Position, bot.Rotation);
        var targetCell = context.Map.GetCellOrDefault(targetPosition);
        if (targetCell?.Entity is not Poison)
            return;

        context.Map.SetEntity(targetPosition.X, targetPosition.Y, new Food(targetPosition));
    }
}
