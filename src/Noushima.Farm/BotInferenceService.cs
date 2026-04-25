using System.Collections.Concurrent;
using TorchSharp;

namespace Noushima.Farm;

public class BotInferenceService : IBotInferenceService, IDisposable
{
    private readonly ConcurrentDictionary<Guid, MlpModule> bots = new();

    public void LoadBot(Guid botId, BotMlpModel botMlpModel)
    {
        var module = new MlpModule(botMlpModel);
        bots[botId] = module;
    }

    public void RemoveBot(Guid botId)
    {
        if (bots.Remove(botId, out var module))
            module.Dispose();
    }

    public float[] Inference(Guid botId, float[] input)
    {
        if (!bots.TryGetValue(botId, out var module))
            throw new InvalidOperationException($"Bot {botId} does not loaded");
        var inputVector = torch.tensor(input, device: FarmConstants.Device);
        var outputVector = module.forward(inputVector);
        return outputVector.to(torch.ScalarType.Float32).data<float>().ToArray();
    }

    public void Dispose()
    {
        foreach (var module in bots.Values)
            module.Dispose();
    }
}