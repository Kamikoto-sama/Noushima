using Noushima.Island.Simulation;

namespace Noushima.Island.Api.Services;

public sealed class SimulationRunnerService(
    SimulationEngine engine,
    SimulationSpeedControl speedControl) : BackgroundService
{
    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        while (!stoppingToken.IsCancellationRequested)
        {
            engine.RunTick();
            if (!speedControl.IsEnabled)
                await Task.Delay(10, stoppingToken);
        }
    }
}
