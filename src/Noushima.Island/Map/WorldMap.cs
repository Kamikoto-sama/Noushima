using System.Drawing;
using Noushima.Island.Entities;

namespace Noushima.Island.Map;

public sealed class WorldMap
{
    private readonly Cell[,] grid;
    private readonly Random random;
    public int Width { get; }
    public int Height { get; }

    public WorldMap(WorldObjectType?[,] layout, Random random)
    {
        grid = BuildGrid(layout);
        this.random = random;
        Width = grid.GetLength(0);
        Height = grid.GetLength(1);
    }

    private static Cell[,] BuildGrid(WorldObjectType?[,] layout)
    {
        var width = layout.GetLength(0);
        var height = layout.GetLength(1);
        if (width == 0 || height == 0)
            throw new InvalidDataException("Map layout cannot be empty.");

        var grid = new Cell[width, height];
        for (var x = 0; x < width; x++)
        {
            for (var y = 0; y < height; y++)
            {
                var cell = new Cell(x, y);
                cell.Entity = CreateEntity(layout[x, y], cell.Position);
                grid[x, y] = cell;
            }
        }

        return grid;
    }

    public bool InBounds(Point point) => point.X >= 0 && point.X < Width && point.Y >= 0 && point.Y < Height;

    public Cell GetCell(int x, int y)
    {
        if (!InBounds(new Point(x, y)))
            throw new ArgumentOutOfRangeException($"Cell ({x}, {y}) is outside of the map.");

        return grid[x, y];
    }

    public Cell? GetCellOrDefault(Point point) => InBounds(point) ? grid[point.X, point.Y] : null;

    public void SetEntity(int x, int y, IEntity? entity)
    {
        var cell = GetCell(x, y);
        cell.Entity = entity;
        if (entity is not null)
            entity.Position = cell.Position;
    }

    public bool MoveEntity(IEntity entity, int newX, int newY)
    {
        var targetPosition = new Point(newX, newY);
        if (!InBounds(targetPosition))
            return false;

        var target = GetCell(newX, newY);
        if (target.Entity is Wall or Bot)
            return false;

        var source = GetCell(entity.Position.X, entity.Position.Y);
        source.Entity = null;
        target.Entity = entity;
        entity.Position = target.Position;
        return true;
    }

    public IReadOnlyList<Cell?> GetNeighbors(int x, int y)
    {
        var origin = new Point(x, y);
        return Enum.GetValues<Direction>()
            .Select(direction => GetCellOrDefault(Offset(origin, direction.GetOffset())))
            .ToArray();
    }

    public Cell? GetRandomEmptyCell()
    {
        var emptyCells = EnumerateCells().Where(cell => cell.IsEmpty).ToArray();
        if (emptyCells.Length == 0)
            return null;

        return emptyCells[random.Next(emptyCells.Length)];
    }

    public int CountEntities(WorldObjectType type) => EnumerateCells().Count(cell => cell.Entity?.Type == type);

    public IEnumerable<Cell> EnumerateCells()
    {
        for (var y = 0; y < Height; y++)
        for (var x = 0; x < Width; x++)
            yield return grid[x, y];
    }

    public Point GetAdjacentPosition(Point origin, Direction direction) => Offset(origin, direction.GetOffset());

    private static Point Offset(Point origin, Point delta) => new(origin.X + delta.X, origin.Y + delta.Y);

    private static IEntity? CreateEntity(WorldObjectType? type, Point position) => type switch
    {
        null => null,
        WorldObjectType.Wall => new Wall(position),
        WorldObjectType.Food => new Food(position),
        WorldObjectType.Poison => new Poison(position),
        WorldObjectType.Bot => throw new InvalidDataException("Prepared map layout cannot contain bots."),
        _ => throw new InvalidDataException($"Unsupported map entity type '{type}'."),
    };
}