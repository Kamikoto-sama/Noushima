using System.Text.Json.Serialization;

namespace Noushima.Island.Api.Contracts;

public sealed class SimulationStateDto
{
    public int Generation { get; init; }
    public int BotsAlive { get; init; }
    public bool SpeedUpEnabled { get; init; }
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public int? Width { get; init; }
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public int? Height { get; init; }
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public IReadOnlyList<CellDto>? Cells { get; init; }
}

public sealed class SimulationSpeedUpDto
{
    public bool Enabled { get; init; }
}

public sealed class CellDto
{
    public int X { get; init; }
    public int Y { get; init; }
    public required string Type { get; init; }
    public int? Energy { get; init; }
    public string? Direction { get; init; }
}
