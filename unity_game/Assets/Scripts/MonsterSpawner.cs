using System.Collections;
using UnityEngine;

/// <summary>
/// Randomly spawns monsters at the edges of the screen.
/// Monster variety and frequency increase with time.
/// </summary>
public class MonsterSpawner : MonoBehaviour
{
    [Header("Prefabs – assign different monster prefabs here")]
    public GameObject[] monsterPrefabs;

    [Header("Timing")]
    public float initialInterval = 3f;
    public float minInterval = 0.6f;
    public float intervalDecreaseRate = 0.01f;

    private Camera _cam;
    private Coroutine _spawnCoroutine;

    private void Awake()
    {
        _cam = Camera.main;
    }

    public void StartSpawning()
    {
        _spawnCoroutine = StartCoroutine(SpawnLoop());
    }

    public void StopSpawning()
    {
        if (_spawnCoroutine != null) StopCoroutine(_spawnCoroutine);
    }

    private IEnumerator SpawnLoop()
    {
        while (true)
        {
            float elapsed = GameManager.Instance.ElapsedTime;
            float interval = Mathf.Max(minInterval,
                                       initialInterval - intervalDecreaseRate * elapsed);
            yield return new WaitForSeconds(interval);
            SpawnMonster();
        }
    }

    private void SpawnMonster()
    {
        if (monsterPrefabs == null || monsterPrefabs.Length == 0) return;

        GameObject prefab = monsterPrefabs[Random.Range(0, monsterPrefabs.Length)];
        Vector2 pos = RandomEdgePosition();
        Instantiate(prefab, pos, Quaternion.identity);
    }

    private Vector2 RandomEdgePosition()
    {
        float halfH = _cam.orthographicSize + 0.5f;
        float halfW = halfH * _cam.aspect + 0.5f;

        // Pick a random edge: 0=top, 1=bottom, 2=left, 3=right
        int edge = Random.Range(0, 4);
        return edge switch
        {
            0 => new Vector2(Random.Range(-halfW, halfW),  halfH),
            1 => new Vector2(Random.Range(-halfW, halfW), -halfH),
            2 => new Vector2(-halfW, Random.Range(-halfH, halfH)),
            _ => new Vector2( halfW, Random.Range(-halfH, halfH)),
        };
    }
}
