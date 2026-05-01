using System.Drawing;

namespace Noushima.Island.MapObjects;

public class Poison : IMapObject
{
    public Point Position { get; }
    public MapObjectSize Size => MapObjectSize.One;
    public WorldObjectType Type => WorldObjectType.Poison;

    public Poison(Point position) => Position = position;
}