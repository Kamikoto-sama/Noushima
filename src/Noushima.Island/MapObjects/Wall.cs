using System.Drawing;

namespace Noushima.Island.MapObjects;

public class Wall : IMapObject
{
    public Point Position { get; }
    public MapObjectSize Size { get; }
    public WorldObjectType Type => WorldObjectType.Wall;

    public Wall(Point position, MapObjectSize size)
    {
        Position = position;
        Size = size;
    }
}