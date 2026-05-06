namespace Noushima.Island.Config;

public sealed class IslandConfig
{
    public int RandomSeed { get; init; } = 12345;
    public int GenerationSize { get; init; } = 64;
    public int SurvivorThreshold { get; init; } = 8;
    public float InitialBotEnergy { get; init; } = 80f;
    public float MaxBotEnergy { get; init; } = 120f;
    public float ReproductionEnergyThreshold { get; init; } = 80f;
    public float ReproductionEnergyCostFactor { get; init; } = 0.5f;
    public int MaxFoodCount { get; init; } = 80;
    public int MaxPoisonCount { get; init; } = 40;
    public float SpawnChancePerTick { get; init; } = 1f;
    public float FoodEnergyGain { get; init; } = 18f;
    public float PoisonEnergyPenalty { get; init; } = 18f;
    public float BasePassiveDrain { get; init; } = 0.6f;
    public float ComplexityDrainFactor { get; init; } = 0.015f;
    public int InitialHiddenLayerCount { get; init; } = 1;
    public int InitialHiddenLayerSize { get; init; } = 16;
    public int MinHiddenLayerSize { get; init; } = 4;
    public int MaxHiddenLayerSize { get; init; } = 32;
    public int MaxHiddenLayerCount { get; init; } = 4;
    public float WeightMutationScale { get; init; } = 0.18f;
    public float BiasMutationScale { get; init; } = 0.18f;
    public float AddNeuronChance { get; init; } = 0.15f;
    public float RemoveNeuronChance { get; init; } = 0.1f;
    public float AddLayerChance { get; init; } = 0.08f;
    public float RemoveLayerChance { get; init; } = 0.05f;
    public float StrongMutationMultiplier { get; init; } = 1.75f;
}
