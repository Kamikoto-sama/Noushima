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

    [HttpGet("mode")]
    public ActionResult<SimulationModeDto> GetMode()
    {
        return Ok(new SimulationModeDto
        {
            Mode = speedControl.Mode,
        });
    }

    [HttpPost("mode")]
    public IActionResult SetMode([FromQuery] SimulationMode mode)
    {
        speedControl.SetMode(mode);
        return NoContent();
    }
}
