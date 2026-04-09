using UnityEngine;

public class BulletSpawner : MonoBehaviour
{
    [Header("References")]
    public GameObject bulletPrefab;

    [Header("Spawn Area")]
    public Vector2 minBoundary  = new Vector2(-9.5f, -5.5f);
    public Vector2 maxBoundary  = new Vector2( 9.5f,  5.5f);
    public float   spawnPadding = 1.2f;

    [Header("Spawn Timing")]
    public float spawnIntervalMin = 0.15f;
    public float spawnIntervalMax = 0.60f;

    [Header("Bullet Speed")]
    public float bulletSpeedMin = 3f;
    public float bulletSpeedMax = 10f;

    [Header("Aim")]
    public Transform player;
    [Range(0f, 1f)] public float aimAtPlayerChance    = 0.7f;
    public float                 randomDirectionJitter = 0.35f;

    [Header("Special Bullets")]
    [Range(0f, 1f)]
    [Tooltip("Fraction of bullets that are gold / absorbable")]
    public float specialBulletChance = 0.12f;

    private float nextSpawnTime;

    // ── Unity lifecycle ────────────────────────────────────────────────────

    private void Start() => ScheduleNextSpawn();

    private void Update()
    {
        if (GameManager.Instance != null && GameManager.Instance.isGameOver) return;

        if (Time.time >= nextSpawnTime)
        {
            SpawnOneBullet();
            ScheduleNextSpawn();
        }
    }

    // ── Helpers ────────────────────────────────────────────────────────────

    private void ScheduleNextSpawn()
    {
        nextSpawnTime = Time.time + Random.Range(spawnIntervalMin, spawnIntervalMax);
    }

    private void SpawnOneBullet()
    {
        if (bulletPrefab == null) return;

        Vector2    spawnPos  = GetSpawnPositionFromEdges();
        GameObject bulletObj = Instantiate(bulletPrefab, spawnPos, Quaternion.identity);

        Bullet bullet = bulletObj.GetComponent<Bullet>();
        if (bullet == null) bullet = bulletObj.AddComponent<Bullet>();

        bullet.isSpecial = Random.value <= specialBulletChance;

        float   speed     = Random.Range(bulletSpeedMin, bulletSpeedMax);
        Vector2 direction = GetBulletDirection(spawnPos);
        bullet.Launch(direction, speed);
    }

    private Vector2 GetSpawnPositionFromEdges()
    {
        int   edge = Random.Range(0, 4);
        float xMin = minBoundary.x - spawnPadding;
        float xMax = maxBoundary.x + spawnPadding;
        float yMin = minBoundary.y - spawnPadding;
        float yMax = maxBoundary.y + spawnPadding;

        return edge switch
        {
            0 => new Vector2(Random.Range(xMin, xMax), yMax),  // top
            1 => new Vector2(Random.Range(xMin, xMax), yMin),  // bottom
            2 => new Vector2(xMin, Random.Range(yMin, yMax)),  // left
            _ => new Vector2(xMax, Random.Range(yMin, yMax)),  // right
        };
    }

    private Vector2 GetBulletDirection(Vector2 spawnPos)
    {
        bool aimAtPlayer = player != null && Random.value <= aimAtPlayerChance;

        if (aimAtPlayer)
        {
            Vector2 baseDir = ((Vector2)player.position - spawnPos).normalized;
            Vector2 jitter  = Random.insideUnitCircle * randomDirectionJitter;
            Vector2 final   = (baseDir + jitter).normalized;
            return final.sqrMagnitude < 0.0001f ? baseDir : final;
        }

        Vector2 random = Random.insideUnitCircle.normalized;
        return random.sqrMagnitude < 0.0001f ? Vector2.down : random;
    }
}
