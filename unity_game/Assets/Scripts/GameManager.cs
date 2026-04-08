using System.Collections;
using UnityEngine;
using UnityEngine.SceneManagement;

/// <summary>
/// Central game controller.
/// - Tracks elapsed time and score.
/// - Fires a "minute reward" event every 60 seconds.
/// - Ends the game when the player is destroyed.
/// </summary>
public class GameManager : MonoBehaviour
{
    public static GameManager Instance { get; private set; }

    [Header("References")]
    public BulletSpawner bulletSpawner;
    public MonsterSpawner monsterSpawner;
    public RewardSpawner rewardSpawner;
    public UIManager uiManager;

    // Expose UIManager as a property for RewardItem access
    public UIManager UIManager => uiManager;

    [Header("Bullet Speed Scaling")]
    [Tooltip("Base bullet speed at game start")]
    public float baseBulletSpeed = 4f;
    [Tooltip("How much speed is added per second")]
    public float speedIncreasePerSecond = 0.05f;
    [Tooltip("Maximum bullet speed cap")]
    public float maxBulletSpeed = 20f;

    // ── public read-only state ──────────────────────────────────────────────
    public float ElapsedTime { get; private set; }
    public int Score { get; private set; }
    public bool IsGameOver { get; private set; }

    // Current bullet speed (read by BulletSpawner) — computed in Reward helpers section below

    // ── private ─────────────────────────────────────────────────────────────
    private int _lastRewardMinute;

    // ── Unity lifecycle ──────────────────────────────────────────────────────
    private void Awake()
    {
        if (Instance != null && Instance != this)
        {
            Destroy(gameObject);
            return;
        }
        Instance = this;
    }

    private void Start()
    {
        IsGameOver = false;
        ElapsedTime = 0f;
        Score = 0;
        _lastRewardMinute = 0;

        bulletSpawner.StartSpawning();
        monsterSpawner.StartSpawning();
    }

    private void Update()
    {
        if (IsGameOver) return;

        ElapsedTime += Time.deltaTime;

        // Add 1 point per second survived
        Score = Mathf.FloorToInt(ElapsedTime);
        uiManager.UpdateHUD(Score, ElapsedTime, CurrentBulletSpeed);

        // Bullet slow countdown
        if (_bulletSlowActive)
        {
            _bulletSlowTimer -= Time.deltaTime;
            if (_bulletSlowTimer <= 0f) _bulletSlowActive = false;
        }

        // Trigger minute reward
        int currentMinute = Mathf.FloorToInt(ElapsedTime / 60f);
        if (currentMinute > _lastRewardMinute)
        {
            _lastRewardMinute = currentMinute;
            rewardSpawner.SpawnRandomReward();
        }
    }

    /// <summary>Called by PlayerController when the player dies.</summary>
    public void TriggerGameOver()
    {
        if (IsGameOver) return;
        IsGameOver = true;

        bulletSpawner.StopSpawning();
        monsterSpawner.StopSpawning();

        uiManager.ShowGameOver(Score);
    }

    public void RestartGame()
    {
        SceneManager.LoadScene(SceneManager.GetActiveScene().buildIndex);
    }

    // ── Reward helpers ───────────────────────────────────────────────────────

    private float _bulletSlowTimer;
    private bool _bulletSlowActive;

    public float CurrentBulletSpeed
    {
        get
        {
            float speed = Mathf.Min(baseBulletSpeed + speedIncreasePerSecond * ElapsedTime,
                                    maxBulletSpeed);
            if (_bulletSlowActive) speed *= 0.5f;
            return speed;
        }
    }

    public void ActivateBulletSlow(float duration)
    {
        _bulletSlowActive = true;
        _bulletSlowTimer = duration;
    }

    public void AddBonusScore(int bonus)
    {
        Score += bonus;
    }
}
