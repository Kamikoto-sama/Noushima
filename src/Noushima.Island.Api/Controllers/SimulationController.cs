using Microsoft.AspNetCore.Mvc;
using Noushima.Island.Api.Contracts;
using Noushima.Island.Api.Services;
using Noushima.Island.Simulation;

namespace Noushima.Island.Api.Controllers;

[ApiController]
[Route("api/[controller]")]
public sealed class SimulationController(
    SimulationStateProvider stateProvider,
    SimulationSpeedControl speedControl) : ControllerBase
{
    [HttpGet("state")]
    public ActionResult<SimulationStateDto> GetState()
    {
        var state = stateProvider.GetState();
        return Ok(state);
    }

    [HttpGet("speed-up")]
    public ActionResult<SimulationSpeedUpDto> GetSpeedUp()
    {
        return Ok(new SimulationSpeedUpDto
        {
            Enabled = speedControl.IsEnabled,
        });
    }

    [HttpPost("speed-up")]
    public IActionResult SetSpeedUp([FromQuery] bool enabled)
    {
        speedControl.SetEnabled(enabled);
        return NoContent();
    }
}
