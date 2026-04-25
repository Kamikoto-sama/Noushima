using System;
using System.IO;
using System.Text.Json;
using Noushima.Farm;
using TorchSharp;

namespace Sandbox;

internal class Program
{
    public static void Main(string[] args)
    {
        var mlp = JsonSerializer.Deserialize<Mlp>(File.ReadAllText("mlp.json"))!;
        var module = new MlpModule(mlp);
        var input = torch.tensor([2f, 1f], device: torch.CPU);
        var output = module.forward(input);

        Console.WriteLine(output.str());
    }
}