using UnityEngine;

public class BulletSpawner : MonoBehaviour
{
    [Header("Reference")]
    public GameObject bulletPrefab;

    [Header("Spawn Area")]
    public Vector2 minBoundary = new Vector2(-9.5f, -5.5f);
    public Vector2 maxBoundary = new Vector2(9.5f, 5.5f);
    public float spawnPadding = 1.2f;

    [Header("Spawn Timing")]
    public float spawnIntervalMin = 0.15f;
    public float spawnIntervalMax = 0.6f;

    [Header("Bullet Speed")]
    public float bulletSpeedMin = 3f;
    public float bulletSpeedMax = 10f;

    [Header("Aim")]
    public Transform player;
    [Range(0f, 1f)]
    public float aimAtPlayerChance = 0.7f;
    public float randomDirectionJitter = 0.35f;

    private float nextSpawnTime;

    private void Start()
    {
        ScheduleNextSpawn();
    }

    private void Update()
    {
        if (GameManager.Instance != null && GameManager.Instance.isGameOver)
        {
            return;
        }

        if (Time.time >= nextSpawnTime)
        {
            SpawnOneBullet();
            ScheduleNextSpawn();
        }
    }

    private void ScheduleNextSpawn()
    {
        nextSpawnTime = Time.time + Random.Range(spawnIntervalMin, spawnIntervalMax);
    }

    private void SpawnOneBullet()
    {
        if (bulletPrefab == null)
        {
            return;
        }

        Vector2 spawnPos = GetSpawnPositionFromEdges();
        GameObject bulletObj = Instantiate(bulletPrefab, spawnPos, Quaternion.identity);

        Bullet bullet = bulletObj.GetComponent<Bullet>();
        if (bullet == null)
        {
            bullet = bulletObj.AddComponent<Bullet>();
        }

        float speed = Random.Range(bulletSpeedMin, bulletSpeedMax);
        Vector2 direction = GetBulletDirection(spawnPos);
        bullet.Launch(direction, speed);
    }

    private Vector2 GetSpawnPositionFromEdges()
    {
        int edge = Random.Range(0, 4);

        float xMin = minBoundary.x - spawnPadding;
        float xMax = maxBoundary.x + spawnPadding;
        float yMin = minBoundary.y - spawnPadding;
        float yMax = maxBoundary.y + spawnPadding;

        switch (edge)
        {
            case 0: // top
                return new Vector2(Random.Range(xMin, xMax), yMax);
            case 1: // bottom
                return new Vector2(Random.Range(xMin, xMax), yMin);
            case 2: // left
                return new Vector2(xMin, Random.Range(yMin, yMax));
            default: // right
                return new Vector2(xMax, Random.Range(yMin, yMax));
        }
    }

    private Vector2 GetBulletDirection(Vector2 spawnPos)
    {
        bool aimAtPlayer = player != null && Random.value <= aimAtPlayerChance;

        if (aimAtPlayer)
        {
            Vector2 baseDir = (Vector2)(player.position - (Vector3)spawnPos).normalized;
            Vector2 jitter = Random.insideUnitCircle * randomDirectionJitter;
            Vector2 finalDir = (baseDir + jitter).normalized;
            return finalDir.sqrMagnitude < 0.0001f ? baseDir : finalDir;
        }

        Vector2 randomDir = Random.insideUnitCircle.normalized;
        return randomDir.sqrMagnitude < 0.0001f ? Vector2.down : randomDir;
    }
}
