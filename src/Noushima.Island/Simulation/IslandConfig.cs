namespace Noushima.Island.Simulation;

public sealed class IslandConfig
{
    public int? RandomSeed { get; init; } = null;
    public int GenerationSize { get; init; } = 64;
    public int SurvivorThreshold { get; init; } = 8;

    public float InitialBotEnergy { get; init; } = 60f;
    public float MaxBotEnergy { get; init; } = 1000f;

    public float ComplexityDrainFactor { get; init; } = 0.3f;
    public float ComplexityLayersCountFactor { get; init; } = 0.75f;
    public float ComplexityNeuronsCountFactor { get; init; } = 0.35f;
    public float ComplexityLinksCountFactor { get; init; } = 0.008f;

    public int MaxFoodCount { get; init; } = 50;
    public int MaxPoisonCount { get; init; } = 50;
    public float SpawnChancePerTick { get; init; } = 1f;
    public float FoodEnergyGain { get; init; } = 200f;
    public float PoisonEnergyPenalty { get; init; } = 600f;

    public int MinHiddenLayersCount { get; init; } = 1;
    public int MaxHiddenLayersCount { get; init; } = 8;
    public int MinHiddenLayerSize { get; init; } = 1;
    public int MaxHiddenLayerSize { get; init; } = 24;

    public float WeightMutationChance { get; init; } = 0.18f;
    public float RemoveLinkChance { get; init; } = 0f;
    public float BiasMutationChance { get; init; } = 0.18f;
    public float AddNeuronChance { get; init; } = 1f;
    public float AddLinkChance { get; init; } = 0f;
    public float RemoveNeuronChance { get; init; } = 0f;
    public float AddLayerChance { get; init; } = 0f;
    public float RemoveLayerChance { get; init; } = 0f;
    public float StrongMutationMultiplier { get; init; } = 1f;
}
