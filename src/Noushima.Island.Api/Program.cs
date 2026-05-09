using Noushima.Island.Api.Services;
using Noushima.Island.Api.Simulation;
using Noushima.Farm;
using Noushima.Island.Map;
using Noushima.Island.Simulation;

namespace Noushima.Island.Api;

public class Program
{
    public static void Main(string[] args)
    {
        var builder = WebApplication.CreateBuilder(args);

        builder.Services.AddControllers();
        builder.Services.AddSingleton<IslandConfig>();
        builder.Services.AddSingleton<IInferenceService, InferenceService>();
        builder.Services.AddSingleton<IMapProvider>(_ =>
            new FileMapProvider(Path.Combine("Maps", "map.txt")));
        builder.Services.AddSingleton<SimulationSpeedControl>();
        builder.Services.AddSingleton<SimulationEngine>();
        builder.Services.AddSingleton<SimulationStateProvider>();
        builder.Services.AddHostedService<SimulationRunnerService>();

        var app = builder.Build();

        app.UseDefaultFiles();
        app.UseStaticFiles();
        app.MapControllers();

        app.Run();
    }
}
