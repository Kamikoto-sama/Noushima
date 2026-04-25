namespace Noushima.Farm;

public interface IInferenceService
{
    void LoadModel(Guid modelId, MlpModel mlpModel);
    void RemoveModel(Guid modelId);
    float[] Inference(Guid modelId, float[] input);
}