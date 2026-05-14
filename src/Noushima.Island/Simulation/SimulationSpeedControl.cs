namespace Noushima.Island.Simulation;

public enum SimulationMode
{
    Pause,
    Slow,
    Normal,
    Fast,
}

public sealed class SimulationSpeedControl
{
    private int mode = (int)SimulationMode.Normal;
    public SimulationMode Mode => (SimulationMode)Volatile.Read(ref mode);

    public void SetMode(SimulationMode value)
    {
        Volatile.Write(ref mode, (int)value);
    }
}
