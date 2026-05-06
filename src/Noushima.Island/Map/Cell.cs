using System.Drawing;
using Noushima.Island.Entities;

namespace Noushima.Island.Map;

public sealed class Cell
{
    public Point Position { get; }
    public IEntity? Entity { get; internal set; }
    public bool IsEmpty => Entity is null;

    public Cell(int x, int y)
    {
        Position = new Point(x, y);
    }
}
