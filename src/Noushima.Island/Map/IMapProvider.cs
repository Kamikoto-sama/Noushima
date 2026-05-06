namespace Noushima.Island.Map;

public interface IMapProvider
{
    WorldObjectType?[,] GetMap();
}
