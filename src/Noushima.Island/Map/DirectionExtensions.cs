using System.Drawing;

namespace Noushima.Island.Map;

public static class DirectionExtensions
{
    private static readonly Point[] Offsets =
    [
        new Point(0, -1),
        new Point(1, -1),
        new Point(1, 0),
        new Point(1, 1),
        new Point(0, 1),
        new Point(-1, 1),
        new Point(-1, 0),
        new Point(-1, -1),
    ];

    public static Point GetOffset(this Direction direction) => Offsets[(int)direction];

    public static Direction ApplyRelative(this Direction facing, Direction relative) =>
        (Direction)(((int)facing + (int)relative) % Offsets.Length);

    public static Direction Opposite(this Direction direction) => direction.ApplyRelative(Direction.Down);

    public static float Normalize(this Direction direction) => (2f * (int)direction / (Offsets.Length - 1)) - 1f;

    public static bool TryDecodeRelative(float value, out Direction direction)
    {
        direction = Direction.Up;
        if (value <= 0f)
            return false;

        var clamped = Math.Clamp(value, 0f, 1f);
        var bucket = Math.Min(Offsets.Length - 1, (int)Math.Ceiling(clamped * Offsets.Length) - 1);
        direction = (Direction)Math.Max(bucket, 0);
        return true;
    }
}
