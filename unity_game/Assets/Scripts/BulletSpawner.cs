using System.Collections;
using UnityEngine;

/// <summary>
/// Spawns bullets from all 8 compass directions toward the center.
/// Spawn rate and bullet speed both escalate as the game progresses.
/// </summary>
public class BulletSpawner : MonoBehaviour
{
    [Header("Prefab")]
    public GameObject bulletPrefab;

    [Header("Spawn Timing")]
    [Tooltip("Initial interval between bullet waves (seconds)")]
    public float initialInterval = 0.8f;
    [Tooltip("Minimum spawn interval (will not go below this)")]
    public float minInterval = 0.15f;
    [Tooltip("How quickly the interval decreases (seconds per second)")]
    public float intervalDecreaseRate = 0.004f;

    [Header("Spread")]
    [Tooltip("Half-angle spread around each direction (degrees)")]
    public float spreadAngle = 10f;

    // ── Direction table: 8 compass points ───────────────────────────────────
    // Each vector is an inward-facing unit direction (toward center).
    private static readonly Vector2[] Directions =
    {
        Vector2.up,
        Vector2.down,
        Vector2.left,
        Vector2.right,
        new Vector2( 1f,  1f).normalized,
        new Vector2(-1f,  1f).normalized,
        new Vector2( 1f, -1f).normalized,
        new Vector2(-1f, -1f).normalized,
    };

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

            SpawnWave();
            yield return new WaitForSeconds(interval);
        }
    }

    private void SpawnWave()
    {
        float halfH = _cam.orthographicSize + 0.5f;
        float halfW = halfH * _cam.aspect + 0.5f;

        foreach (Vector2 inward in Directions)
        {
            // Spawn position: outside the screen edge, opposite to inward direction
            Vector2 spawnPos = GetSpawnPosition(-inward, halfW, halfH);

            // Add spread
            float angle = Mathf.Atan2(inward.y, inward.x) * Mathf.Rad2Deg;
            angle += Random.Range(-spreadAngle, spreadAngle);
            Vector2 dir = new Vector2(Mathf.Cos(angle * Mathf.Deg2Rad),
                                      Mathf.Sin(angle * Mathf.Deg2Rad));

            GameObject bullet = Instantiate(bulletPrefab, spawnPos, Quaternion.identity);
            Bullet b = bullet.GetComponent<Bullet>();
            if (b != null)
                b.Initialize(dir, GameManager.Instance.CurrentBulletSpeed);
        }
    }

    /// <summary>
    /// Returns a spawn position outside the screen in the given outward direction.
    /// </summary>
    private Vector2 GetSpawnPosition(Vector2 outward, float halfW, float halfH)
    {
        // Place along edge based on dominant axis
        float rx = Random.Range(-halfW, halfW);
        float ry = Random.Range(-halfH, halfH);

        if (Mathf.Abs(outward.x) >= Mathf.Abs(outward.y))
            return new Vector2(Mathf.Sign(outward.x) * halfW, ry);
        else
            return new Vector2(rx, Mathf.Sign(outward.y) * halfH);
    }
}
