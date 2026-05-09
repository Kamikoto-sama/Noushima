using Noushima.Island.Entities;
using Noushima.Island.Map;

namespace Noushima.Island.Actions;

internal sealed class RotateBotAction() : DirectionalScalarAction("rotate")
{
    public override int Order => 0;

    public override float Cost => 0.4f;

    protected override void Execute(Bot bot, Direction relativeDirection, BotActionContext context)
    {
        bot.SetRotation(bot.Rotation.ApplyRelative(relativeDirection));
    }
}
