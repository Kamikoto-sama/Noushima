using System.Drawing;
using Noushima.Farm;
using Noushima.Island.Actions;
using Noushima.Island.Entities;
using Noushima.Island.Environment;
using Noushima.Island.Genetics;
using Noushima.Island.Map;

namespace Noushima.Island.Simulation;

public sealed class SimulationEngine
{
    private readonly Random random;
    private readonly ActionRegistry actionRegistry;
    private readonly EnvParamRegistry envParamRegistry;
    private readonly GenomeFactory genomeFactory;
    private readonly GenomeMutator genomeMutator;
    private readonly ResourceSpawner resourceSpawner;
    private readonly IFarmClient farmClient;
    private readonly IMapProvider mapProvider;
    private readonly List<Bot> bots = [];
    public IslandConfig Config { get; }
    public WorldMap Map { get; private set; } = null!;
    public IReadOnlyList<Bot> Bots => bots;
    public int GenerationNumber { get; private set; }
    public int GenTickNumber { get; private set; }
    public int InputSize => envParamRegistry.TotalSize;
    public int OutputSize => actionRegistry.TotalSize;

    public SimulationEngine(IslandConfig config, IInferenceService inferenceService, IMapProvider mapProvider)
    {
        Config = config;
        random = config.RandomSeed.HasValue ? new Random(config.RandomSeed.Value) : new Random();
        actionRegistry = new ActionRegistry();
        envParamRegistry = new EnvParamRegistry(config, actionRegistry.TotalSize);
        var genomeComplexityCalculator = new GenomeComplexityCalculator(config);
        genomeFactory = new GenomeFactory(config, random, genomeComplexityCalculator);
        genomeMutator = new GenomeMutator(config, random, genomeFactory, genomeComplexityCalculator);
        resourceSpawner = new ResourceSpawner(config, random);
        farmClient = new FarmClient(inferenceService);
        this.mapProvider = mapProvider;
    }

    public void Initialize()
    {
        ResetWorld();
        bots.Clear();
        for (var index = 0; index < Config.GenerationSize; index++)
            bots.Add(CreateBot(genomeFactory.CreateNewGenome(InputSize, OutputSize)));

        PlaceBots(bots);
        resourceSpawner.Initialize(Map);
        GenerationNumber = 1;
        GenTickNumber = 0;
    }

    public void RunTick()
    {
        EnsureInitialized();
        var activeBots = bots.Where(bot => bot.Alive).OrderBy(bot => bot.Id).ToArray();
        foreach (var bot in activeBots)
            bot.BeginTurn();

        var executionContext = new BotActionExecutionContext(Map, Config)
        {
            Random = random,
            AddBot = bot => bots.Add(bot),
        };

        foreach (var bot in activeBots)
        {
            var input = envParamRegistry.Build(bot, Map);
            var output = farmClient.Infer(bot.Brain, input);
            bot.SetOutputs(output);

            actionRegistry.Execute(bot, output, executionContext);

            bot.ChangeEnergy(-GetPassiveDrain(bot), Config.MaxBotEnergy);
            if (!bot.Alive)
            {
                Map.SetEntity(bot.Position.X, bot.Position.Y, null);
                continue;
            }

            if (!bot.Alive)
                Map.SetEntity(bot.Position.X, bot.Position.Y, null);
        }

        GenTickNumber++;
        resourceSpawner.Update(Map);
        if (bots.Count(bot => bot.Alive) <= Config.SurvivorThreshold)
            ResetGeneration();
    }

    private void ResetGeneration()
    {
        var survivors = bots.Where(bot => bot.Alive).ToArray();
        foreach (var bot in bots)
            farmClient.Forget(bot.Brain);

        bots.Clear();
        if (survivors.Length == 0)
        {
            for (var index = 0; index < Config.GenerationSize; index++)
                bots.Add(CreateBot(genomeFactory.CreateNewGenome(InputSize, OutputSize)));
        }
        else
        {
            foreach (var survivor in survivors)
            {
                bots.Add(CreateBot(survivor.Brain.Clone()));
                var template = survivor.Brain;
                bots.Add(CreateBot(genomeMutator.Mutate(template, true)));
            }
        }

        ClearBotsFromWorld();
        PlaceBots(bots);
        resourceSpawner.Initialize(Map);
        GenerationNumber++;
        GenTickNumber = 0;
    }

    private void ResetWorld()
    {
        Map = new WorldMap(mapProvider.GetMap(), random);
    }

    private void ClearBotsFromWorld()
    {
        foreach (var cell in Map.EnumerateCells())
        {
            if (cell.Entity is Bot)
                Map.SetEntity(cell.Position.X, cell.Position.Y, null);
        }
    }

    private void PlaceBots(IEnumerable<Bot> newBots)
    {
        foreach (var bot in newBots)
        {
            var cell = Map.GetRandomEmptyCell() ?? throw new InvalidOperationException("Map has no empty cells for bot placement.");
            Map.SetEntity(cell.Position.X, cell.Position.Y, bot);
        }
    }

    private Bot CreateBot(Genome genome) =>
        new(Point.Empty, (Direction)random.Next(8), Config.InitialBotEnergy, genome, OutputSize);

    private float GetPassiveDrain(Bot bot) => bot.Brain.Complexity * Config.ComplexityDrainFactor;

    private void EnsureInitialized()
    {
        if (Map is not null)
            return;  

        Initialize();
    }
}
