using Noushima.Island.Entities;
using Noushima.Island.Map;

namespace Noushima.Island.Actions;

internal abstract class DirectionalScalarAction(string name) : IBotAction
{
    private const int DirectionsCount = 8;

    public string Name { get; } = name;
    public abstract int Order { get; }
    public abstract float Cost { get; }

    public int Size => 1;

    public void Execute(Bot bot, ReadOnlySpan<float> intentions, BotActionContext context)
    {
        if (!TryGetRelativeDirection(intentions, out var relativeDirection))
            return;

        Execute(bot, relativeDirection, context);
    }

    protected abstract void Execute(Bot bot, Direction relativeDirection, BotActionContext context);

    private static bool TryGetRelativeDirection(ReadOnlySpan<float> output, out Direction direction)
    {
        direction = Direction.Up;
        if (output[0] <= 0f)
            return false;

        var clamped = Math.Clamp(output[0], 0f, 1f);
        var option = Math.Min(DirectionsCount - 1, (int)Math.Ceiling(clamped * DirectionsCount) - 1);
        direction = (Direction)option;
        return true;
    }
}
