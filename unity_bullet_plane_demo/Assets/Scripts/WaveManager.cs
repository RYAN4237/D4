using UnityEngine;

/// <summary>
/// Escalates difficulty every <waveInterval> seconds by:
///   1. Increasing bullet spawn rate (BulletSpawner).
///   2. Increasing enemy spawn rate (EnemySpawner).
///   3. Increasing enemy move speed (applied to newly spawned enemies via prefab scale).
///
/// Attach to any persistent GameObject alongside BulletSpawner / EnemySpawner.
/// </summary>
public class WaveManager : MonoBehaviour
{
    // ── Singleton (lightweight – no DontDestroyOnLoad needed for single-scene MVP) ──
    public static WaveManager Instance { get; private set; }

    [Header("References")]
    public BulletSpawner bulletSpawner;
    public EnemySpawner  enemySpawner;

    [Header("Wave Settings")]
    public float waveInterval       = 30f;  // seconds per wave
    public int   maxWaves           = 10;

    [Header("Per-Wave Scaling")]
    [Tooltip("Multiply spawn interval by this each wave (< 1 = faster)")]
    public float bulletIntervalScale = 0.80f;
    public float enemyIntervalScale  = 0.75f;
    [Tooltip("Add this to enemy base speed each wave")]
    public float enemySpeedAdd       = 0.4f;

    // ── State ─────────────────────────────────────────────────────────────
    public int  CurrentWave    { get; private set; } = 1;
    private float nextWaveTime;
    private float baseEnemySpeed = 2.5f; // matches KamikazeEnemy default

    // ── Unity lifecycle ────────────────────────────────────────────────────

    private void Awake()
    {
        if (Instance != null && Instance != this) { Destroy(gameObject); return; }
        Instance = this;
    }

    private void Start()
    {
        nextWaveTime = waveInterval;
    }

    private void Update()
    {
        if (GameManager.Instance == null) return;
        if (GameManager.Instance.State != GameManager.GameState.Playing) return;
        if (CurrentWave >= maxWaves) return;

        if (GameManager.Instance.SurviveTime >= nextWaveTime)
        {
            AdvanceWave();
        }
    }

    // ── Helpers ────────────────────────────────────────────────────────────

    private void AdvanceWave()
    {
        CurrentWave++;
        nextWaveTime += waveInterval;

        // Tighten bullet spawner
        if (bulletSpawner != null)
        {
            bulletSpawner.spawnIntervalMin *= bulletIntervalScale;
            bulletSpawner.spawnIntervalMax *= bulletIntervalScale;
        }

        // Tighten enemy spawner
        if (enemySpawner != null)
        {
            enemySpawner.ScaleSpawnRate(enemyIntervalScale);
        }

        // Increase speed for future enemies (stored here; EnemySpawner reads it via prefab)
        baseEnemySpeed += enemySpeedAdd;

        Debug.Log($"[WaveManager] Wave {CurrentWave} started. Enemy speed: {baseEnemySpeed:F1}");
    }

    /// <summary>
    /// Called by EnemySpawner or KamikazeEnemy to get the current base speed.
    /// </summary>
    public float GetCurrentEnemySpeed() => baseEnemySpeed;
}
