using System;
using System.IO;
using Noushima.Island;
using Noushima.Island.Map;

namespace Sandbox;

internal sealed class FileMapProvider(string relativePath) : IMapProvider
{
    public WorldObjectType?[,] GetMap()
    {
        var path = Path.IsPathRooted(relativePath)
            ? relativePath
            : Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, relativePath));

        var lines = File.ReadAllLines(path);
        if (lines.Length == 0)
            throw new InvalidDataException($"Map file '{path}' is empty.");

        var width = lines[0].Length;
        if (width == 0)
            throw new InvalidDataException($"Map file '{path}' contains an empty first row.");

        var layout = new WorldObjectType?[width, lines.Length];
        for (var y = 0; y < lines.Length; y++)
        {
            if (lines[y].Length != width)
                throw new InvalidDataException($"Map row {y} in '{path}' has inconsistent width.");

            for (var x = 0; x < width; x++)
            {
                layout[x, y] = lines[y][x] switch
                {
                    '#' => WorldObjectType.Wall,
                    ' ' => null,
                    _ => throw new InvalidDataException($"Unsupported map character '{lines[y][x]}' at ({x}, {y})."),
                };
            }
        }

        return layout;
    }
}
