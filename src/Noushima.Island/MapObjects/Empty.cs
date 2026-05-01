using System.Drawing;

namespace Noushima.Island.MapObjects;

public class Empty: IMapObject
{
    public Point Position { get; }
    public MapObjectSize Size => MapObjectSize.One;
    public WorldObjectType Type => WorldObjectType.Empty;

    public Empty(Point position) => Position = position;
}