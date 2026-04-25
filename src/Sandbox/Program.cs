using System;
using System.IO;
using System.Text.Json;
using Noushima.Farm;

namespace Sandbox;

internal class Program
{
    public static void Main()
    {
        var mlpModel = JsonSerializer.Deserialize<MlpModel>(File.ReadAllText("mlp.json"))!;
        var id = Guid.NewGuid();
        using var service = new InferenceService();
        service.LoadModel(id, mlpModel);
        var output = service.Inference(id, [1f, 2f]);
        Console.WriteLine(string.Join(", ", output));
    }
}