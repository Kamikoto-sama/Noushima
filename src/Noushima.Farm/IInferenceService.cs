namespace Noushima.Farm;

public interface IInferenceService
{
    void LoadModel(Guid modelId, MlpModel mlpModel);
    void RemoveModel(Guid modelId);
    float[] Infer(Guid modelId, float[] input);
}