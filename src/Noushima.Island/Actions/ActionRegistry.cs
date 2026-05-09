using Noushima.Island.Entities;

namespace Noushima.Island.Actions;

public sealed class ActionRegistry
{
    private readonly IBotAction[] actions;
    public int TotalSize => actions.Sum(action => action.Size);

    public ActionRegistry()
    {
        actions = new IBotAction[]
        {
            new RotateBotAction(),
            new PeelBotAction(),
            new EatBotAction(),
            new MoveBotAction(),
        }
        .OrderBy(action => action.Order)
        .ToArray();
    }

    public void Execute(Bot bot, float[] intetions, BotActionContext context)
    {
        var offset = 0;
        foreach (var action in actions)
        {
            var actionOutput = intetions.AsSpan(offset, action.Size);
            if (actionOutput[0] > 0f)
            {
                action.Execute(bot, actionOutput, context);
                bot.ChangeEnergy(-action.Cost, context.Config.MaxBotEnergy);
            }

            offset += action.Size;
        }
    }
}
