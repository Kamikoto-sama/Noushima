namespace Noushima.Farm;

public interface IBotInferenceService
{
    void LoadBot(Guid botId, BotMlpModel botMlpModel);
    void RemoveBot(Guid botId);
    float[] Inference(Guid botId, float[] input);
}