using System.Drawing;
using Noushima.Island.Map;

namespace Noushima.Island.Entities;

public interface IEntity
{
    Point Position { get; set; }
    WorldObjectType Type { get; }
}
