using System.Drawing;
using Noushima.Island.MapObjects;

namespace Noushima.Island;

public interface IMapObject
{
    Point Position { get; }
    public MapObjectSize Size { get; }
    WorldObjectType Type { get; }
}

public class MapObjectSize
{
    public static readonly MapObjectSize One = new() { Width = 1, Height = 1 };

    public required int Width { get; init; }
    public required int Height { get; init; }
}