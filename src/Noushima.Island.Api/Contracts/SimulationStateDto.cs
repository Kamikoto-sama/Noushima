using System.Text.Json.Serialization;
using Noushima.Island.Simulation;

namespace Noushima.Island.Api.Contracts;

public sealed class SimulationStateDto
{
    public int Generation { get; init; }
    public float LongestGeneration { get; init; }
    public int BotsAlive { get; init; }
    public float BestEnergy { get; init; }
    [JsonConverter(typeof(JsonStringEnumConverter<SimulationMode>))]
    public SimulationMode Mode { get; init; }
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public int? Width { get; init; }
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public int? Height { get; init; }
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public IReadOnlyList<CellDto>? Cells { get; init; }
}

public sealed class SimulationModeDto
{
    [JsonConverter(typeof(JsonStringEnumConverter<SimulationMode>))]
    public SimulationMode Mode { get; init; }
}

public sealed class CellDto
{
    public int X { get; init; }
    public int Y { get; init; }
    public required string Type { get; init; }
    public int? Energy { get; init; }
    public string? Direction { get; init; }
}
