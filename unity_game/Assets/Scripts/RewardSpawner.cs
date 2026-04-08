using UnityEngine;

/// <summary>
/// Spawns a random reward at a random safe position every minute.
/// Called by GameManager.
/// </summary>
public class RewardSpawner : MonoBehaviour
{
    [Header("Reward Prefabs")]
    public GameObject shieldPrefab;
    public GameObject speedBoostPrefab;
    public GameObject slowBulletsPrefab;
    public GameObject scoreBonusPrefab;
    public GameObject clearScreenPrefab;

    private Camera _cam;

    private void Awake()
    {
        _cam = Camera.main;
    }

    public void SpawnRandomReward()
    {
        GameObject prefab = PickRandomPrefab();
        if (prefab == null) return;

        Vector2 pos = RandomSafePosition();
        Instantiate(prefab, pos, Quaternion.identity);
    }

    private GameObject PickRandomPrefab()
    {
        // Build a list of non-null prefabs
        var available = new System.Collections.Generic.List<GameObject>();
        TryAdd(available, shieldPrefab);
        TryAdd(available, speedBoostPrefab);
        TryAdd(available, slowBulletsPrefab);
        TryAdd(available, scoreBonusPrefab);
        TryAdd(available, clearScreenPrefab);

        return available.Count > 0 ? available[Random.Range(0, available.Count)] : null;
    }

    private static void TryAdd(System.Collections.Generic.List<GameObject> list, GameObject obj)
    {
        if (obj != null) list.Add(obj);
    }

    private Vector2 RandomSafePosition()
    {
        float halfH = _cam.orthographicSize * 0.7f;
        float halfW = halfH * _cam.aspect;
        return new Vector2(Random.Range(-halfW, halfW), Random.Range(-halfH, halfH));
    }
}
