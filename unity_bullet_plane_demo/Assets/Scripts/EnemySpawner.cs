using UnityEngine;

/// <summary>
/// Spawns KamikazeEnemy objects off-screen at increasing rate.
/// Also tightens BulletSpawner timing each wave to ramp difficulty.
///
/// Attach to the same GameObject as BulletSpawner, or any persistent object.
/// </summary>
public class EnemySpawner : MonoBehaviour
{
    [Header("Prefabs")]
    public GameObject enemyPrefab;

    [Header("Spawn Area (slightly outside camera bounds)")]
    public Vector2 minSpawn   = new Vector2(-11f, -6.5f);
    public Vector2 maxSpawn   = new Vector2( 11f,  6.5f);
    public float   edgePad    = 1.5f;

    [Header("Timing")]
    public float spawnIntervalStart = 4f;   // seconds between spawns at start
    public float spawnIntervalMin   = 0.8f; // fastest possible

    private float spawnInterval;
    private float nextSpawnTime;

    private void Start()
    {
        spawnInterval = spawnIntervalStart;
        ScheduleNext();
    }

    private void Update()
    {
        if (GameManager.Instance == null) return;
        if (GameManager.Instance.State != GameManager.GameState.Playing) return;

        if (Time.time >= nextSpawnTime)
        {
            SpawnEnemy();
            ScheduleNext();
        }
    }

    private void ScheduleNext()
    {
        nextSpawnTime = Time.time + spawnInterval;
    }

    private void SpawnEnemy()
    {
        if (enemyPrefab == null) return;

        Vector2    pos = GetOffScreenPosition();
        GameObject obj = Instantiate(enemyPrefab, pos, Quaternion.identity);

        // Apply current wave speed
        KamikazeEnemy enemy = obj.GetComponent<KamikazeEnemy>();
        if (enemy != null && WaveManager.Instance != null)
            enemy.SetSpeed(WaveManager.Instance.GetCurrentEnemySpeed());
    }

    private Vector2 GetOffScreenPosition()
    {
        int   edge = Random.Range(0, 4);
        float xMin = minSpawn.x - edgePad;
        float xMax = maxSpawn.x + edgePad;
        float yMin = minSpawn.y - edgePad;
        float yMax = maxSpawn.y + edgePad;

        return edge switch
        {
            0 => new Vector2(Random.Range(xMin, xMax), yMax),
            1 => new Vector2(Random.Range(xMin, xMax), yMin),
            2 => new Vector2(xMin, Random.Range(yMin, yMax)),
            _ => new Vector2(xMax, Random.Range(yMin, yMax)),
        };
    }

    // ── Called by WaveManager ──────────────────────────────────────────────

    /// <summary>Reduces spawn interval by multiplier (e.g. 0.75 = 25% faster).</summary>
    public void ScaleSpawnRate(float multiplier)
    {
        spawnInterval = Mathf.Max(spawnInterval * multiplier, spawnIntervalMin);
    }
}
