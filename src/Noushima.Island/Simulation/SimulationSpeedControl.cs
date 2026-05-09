namespace Noushima.Island.Simulation;

public sealed class SimulationSpeedControl
{
    private int enabled;
    public bool IsEnabled => Volatile.Read(ref enabled) == 1;

    public void SetEnabled(bool value)
    {
        Volatile.Write(ref enabled, value ? 1 : 0);
    }
}
