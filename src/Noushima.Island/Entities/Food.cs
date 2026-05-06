using System.Drawing;
using Noushima.Island.Map;

namespace Noushima.Island.Entities;

public sealed class Food(Point position) : IEntity
{
    public Point Position { get; set; } = position;
    public WorldObjectType Type => WorldObjectType.Food;
}
