using System.Drawing;
using Noushima.Island.Genetics;
using Noushima.Island.Map;

namespace Noushima.Island.Entities;

public sealed class Bot(Point position, Direction rotation, float energy, Genome brain, int outputsSize)
    : IEntity
{
    public Guid Id { get; } = Guid.NewGuid();
    public Point Position { get; set; } = position;
    public WorldObjectType Type => WorldObjectType.Bot;
    public float Energy { get; private set; } = energy;
    public float LastEnergy { get; private set; } = energy;
    public Direction Rotation { get; private set; } = rotation;
    public float[] LastOutputs { get; private set; } = new float[outputsSize];
    public Genome Brain { get; } = brain;
    public bool Alive => Energy > 0f;
    public float EnergyDiff => Energy - LastEnergy;
    public bool WantsToReproduce { get; private set; }
    public Guid? FirstParentId { get; init; }
    public Guid? SecondParentId { get; init; }

    public void BeginTurn()
    {
        LastEnergy = Energy;
        WantsToReproduce = false;
    }

    public void SetRotation(Direction rotation) => Rotation = rotation;

    public void SetOutputs(float[] outputs) => LastOutputs = outputs;

    public void SetEnergy(float energy, float maxEnergy) => Energy = Math.Clamp(energy, 0f, maxEnergy);

    public void ChangeEnergy(float delta, float maxEnergy) => SetEnergy(Energy + delta, maxEnergy);

    public void SetWantsToReproduce() => WantsToReproduce = true;

    public bool HasParent(Guid botId) => FirstParentId == botId || SecondParentId == botId;
}
