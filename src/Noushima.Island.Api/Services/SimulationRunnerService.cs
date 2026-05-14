using Noushima.Island.Simulation;

namespace Noushima.Island.Api.Services;

public sealed class SimulationRunnerService(
    SimulationEngine engine,
    SimulationSpeedControl speedControl) : BackgroundService
{
    private const int NormalDelayMs = 10;
    private const int SlowDelayMs = 250;

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        while (!stoppingToken.IsCancellationRequested)
        {
            var mode = speedControl.Mode;
            if (mode == SimulationMode.Pause)
            {
                await Task.Delay(NormalDelayMs, stoppingToken);
                continue;
            }

            engine.RunTick();

            var delayMs = mode switch
            {
                SimulationMode.Slow => SlowDelayMs,
                SimulationMode.Normal => NormalDelayMs,
                _ => 0,
            };

            if (delayMs > 0)
                await Task.Delay(delayMs, stoppingToken);
        }
    }
}
