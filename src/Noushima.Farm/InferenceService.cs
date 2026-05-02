using System.Collections.Concurrent;
using TorchSharp;

namespace Noushima.Farm;

public class InferenceService : IInferenceService, IDisposable
{
    private readonly ConcurrentDictionary<Guid, MlpModule> models = new();

    public void LoadModel(Guid modelId, MlpModel mlpModel)
    {
        var module = new MlpModule(mlpModel);
        models[modelId] = module;
    }

    public void RemoveModel(Guid modelId)
    {
        if (models.Remove(modelId, out var module))
            module.Dispose();
    }

    public float[] Infer(Guid modelId, float[] input)
    {
        if (!models.TryGetValue(modelId, out var module))
            throw new InvalidOperationException($"Model {modelId} has not loaded");
        var inputVector = torch.tensor(input, device: FarmConstants.Device);
        var outputVector = module.Inference(inputVector);
        return outputVector.to(torch.ScalarType.Float32).data<float>().ToArray();
    }

    public void Dispose()
    {
        foreach (var module in models.Values)
            module.Dispose();
    }
}