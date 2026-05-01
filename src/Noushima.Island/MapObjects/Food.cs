using System.Drawing;

namespace Noushima.Island.MapObjects;

public class Food: IMapObject
{
    public Point Position { get; }
    public MapObjectSize Size => MapObjectSize.One;
    public WorldObjectType Type => WorldObjectType.Food;

    public Food(Point position) => Position = position;
}