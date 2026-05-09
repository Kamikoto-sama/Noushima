namespace Noushima.Island.Simulation;

public sealed class IslandConfig
{
    public int? RandomSeed { get; init; } = null;
    public int GenerationSize { get; init; } = 64;
    public int SurvivorThreshold { get; init; } = 8;

    public float InitialBotEnergy { get; init; } = 80f;
    public float MaxBotEnergy { get; init; } = 120f;

    public float ComplexityDrainFactor { get; init; } = 0.5f;
    public float ComplexityLayersCountFactor { get; init; } = 1f;
    public float ComplexityNeuronsCountFactor { get; init; } = 0.5f;
    public float ComplexityLinksCountFactor { get; init; } = 0.01f;

    public int MaxFoodCount { get; init; } = 80;
    public int MaxPoisonCount { get; init; } = 40;
    public float SpawnChancePerTick { get; init; } = 1f;
    public float FoodEnergyGain { get; init; } = 10f;
    public float PoisonEnergyPenalty { get; init; } = 5f;

    public int MinHiddenLayersCount { get; init; } = 1;
    public int MaxHiddenLayersCount { get; init; } = 10;
    public int MinHiddenLayerSize { get; init; } = 1;
    public int MaxHiddenLayerSize { get; init; } = 32;

    public float WeightMutationChance { get; init; } = 0.18f;
    public float RemoveLinkChance { get; init; } = 0.02f;
    public float BiasMutationChance { get; init; } = 0.18f;
    public float AddNeuronChance { get; init; } = 0.15f;
    public float AddLinkChance { get; init; } = 0.8f;
    public float RemoveNeuronChance { get; init; } = 0.15f;
    public float AddLayerChance { get; init; } = 0.08f;
    public float RemoveLayerChance { get; init; } = 0.08f;
    public float StrongMutationMultiplier { get; init; } = 1.75f;
}