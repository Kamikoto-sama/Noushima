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
    private readonly SimulationSpeedControl speedControl;
    private readonly List<Bot> bots = [];
    private bool initialized;
    private float bestEnergy;
    private float longestGeneration;
    private SimulationSnapshot snapshot;
    public IslandConfig Config { get; }
    public WorldMap Map { get; private set; } = null!;
    public int GenerationNumber { get; private set; }
    public int GenTickNumber { get; private set; }
    public SimulationSnapshot? Snapshot => Volatile.Read(ref snapshot);
    public int InputSize => envParamRegistry.TotalSize;
    public int OutputSize => actionRegistry.TotalSize;

    public SimulationEngine(
        IslandConfig config,
        IInferenceService inferenceService,
        IMapProvider mapProvider,
        SimulationSpeedControl speedControl)
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
        this.speedControl = speedControl;
        snapshot = new SimulationSnapshot();
    }

    public void EnsureInitialized()
    {
        if (initialized)
            return;

        InitializeCore();
    }

    public void RunTick()
    {
        if (!initialized)
            InitializeCore();

        var activeBots = bots.Where(bot => bot.Alive).OrderBy(bot => bot.Id).ToArray();
        foreach (var bot in activeBots)
            bot.BeginTurn();

        var context = new BotActionContext(Map, Config);

        foreach (var bot in activeBots)
        {
            var input = envParamRegistry.Build(bot, Map);
            var intentions = farmClient.Infer(bot.Brain, input);
            bot.SetIntentions(intentions);

            actionRegistry.Execute(bot, intentions, context);

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

        PublishSnapshot();
    }

    private void InitializeCore()
    {
        ResetWorld();
        bots.Clear();
        for (var index = 0; index < Config.GenerationSize; index++)
            bots.Add(CreateBot(genomeFactory.CreateNewGenome(InputSize, OutputSize)));

        PlaceBots(bots);
        resourceSpawner.Initialize(Map);
        GenerationNumber = 1;
        GenTickNumber = 0;
        PublishSnapshot();
        initialized = true;
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
                for (var i = 0; i < Config.GenerationSize / survivors.Length - 1; i++)
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
            var cell = Map.GetRandomEmptyCell() ??
                       throw new InvalidOperationException("Map has no empty cells for bot placement.");
            Map.SetEntity(cell.Position.X, cell.Position.Y, bot);
        }
    }

    private Bot CreateBot(Genome genome) =>
        new(Point.Empty, (Direction)random.Next(8), Config.InitialBotEnergy, genome, OutputSize);

    private float GetPassiveDrain(Bot bot) => bot.Brain.Complexity * Config.ComplexityDrainFactor;

    private void PublishSnapshot()
    {
        var botsAlive = bots.Count(bot => bot.Alive);
        bestEnergy = MathF.Max(bestEnergy, bots.Where(b => b.Alive).Max(b => b.Energy));
        longestGeneration = MathF.Max(GenTickNumber, longestGeneration);
        var map = speedControl.Mode == SimulationMode.Fast ? Snapshot!.Map : GetMapSnapshot();

        var simulationSnapshot = new SimulationSnapshot
        {
            Map = map,
            GenerationNumber = GenerationNumber,
            LongestGeneration = longestGeneration,
            BestEnergy = bestEnergy,
            BotsAlive = botsAlive
        };
        Volatile.Write(ref snapshot, simulationSnapshot);
    }

    private CellSnapshot[,] GetMapSnapshot()
    {
        var width = Map.Width;
        var height = Map.Height;
        var map = new CellSnapshot[width, height];

        for (var x = 0; x < width; x++)
        {
            for (var y = 0; y < height; y++)
            {
                var entity = Map.GetCell(x, y).Entity;
                map[x, y] = new CellSnapshot
                {
                    Type = entity?.Type,
                    BotSnapshot = entity is Bot bot
                        ? new BotSnapshot
                        {
                            Direction = bot.Rotation,
                            Energy = (int)bot.Energy,
                        }
                        : null
                };
            }
        }

        return map;
    }
}