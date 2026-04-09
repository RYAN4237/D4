using System;
using UnityEngine;
using UnityEngine.SceneManagement;

public class GameManager : MonoBehaviour
{
    // ── Singleton ──────────────────────────────────────────────────────────
    public static GameManager Instance { get; private set; }

    // ── Game State ─────────────────────────────────────────────────────────
    public enum GameState { Playing, LevelUp, GameOver }
    public GameState State { get; private set; } = GameState.Playing;

    // Legacy helpers so existing scripts keep compiling unchanged
    public bool isGameOver  => State == GameState.GameOver;
    public float surviveTime => SurviveTime;

    // ── Events (subscribe in other managers) ───────────────────────────────
    public event Action<float, float> OnXPChanged;   // (currentXP, threshold)
    public event Action<int>          OnLevelChanged; // new level number
    public event Action               OnLevelUp;      // triggers LevelUp UI
    public event Action               OnGameOver;

    // ── Inspector ──────────────────────────────────────────────────────────
    [Header("UI")]
    public GameObject gameOverPanel;

    [Header("XP / Level")]
    public float xpPerLevel       = 10f;
    public float xpGrowthFactor   = 1.4f;

    // ── Public read-only state ─────────────────────────────────────────────
    public float SurviveTime  { get; private set; }
    public float CurrentXP    { get; private set; }
    public float XPThreshold  { get; private set; }
    public int   Level        { get; private set; } = 1;

    // ── Unity lifecycle ────────────────────────────────────────────────────
    private void Awake()
    {
        if (Instance != null && Instance != this) { Destroy(gameObject); return; }
        Instance = this;
    }

    private void Start()
    {
        State        = GameState.Playing;
        XPThreshold  = xpPerLevel;
        SurviveTime  = 0f;
        CurrentXP    = 0f;
        Time.timeScale = 1f;

        if (gameOverPanel != null) gameOverPanel.SetActive(false);
    }

    private void Update()
    {
        if (State == GameState.Playing)
            SurviveTime += Time.deltaTime;

        if (State == GameState.GameOver && Input.GetKeyDown(KeyCode.R))
            Restart();
    }

    // ── Public API ─────────────────────────────────────────────────────────

    /// <summary>Award XP; triggers level-up flow when threshold is reached.</summary>
    public void AddXP(float amount)
    {
        if (State != GameState.Playing) return;

        CurrentXP += amount;
        OnXPChanged?.Invoke(CurrentXP, XPThreshold);

        if (CurrentXP >= XPThreshold)
        {
            CurrentXP   -= XPThreshold;
            Level++;
            XPThreshold *= xpGrowthFactor;

            State           = GameState.LevelUp;
            Time.timeScale  = 0f;

            OnLevelChanged?.Invoke(Level);
            OnLevelUp?.Invoke();
        }
    }

    /// <summary>Called by LevelUpManager after the player picks an upgrade.</summary>
    public void ResumePlaying()
    {
        if (State != GameState.LevelUp) return;
        State          = GameState.Playing;
        Time.timeScale = 1f;
    }

    public void GameOver()
    {
        if (State == GameState.GameOver) return;

        State          = GameState.GameOver;
        Time.timeScale = 0f;

        if (gameOverPanel != null) gameOverPanel.SetActive(true);
        OnGameOver?.Invoke();
    }

    public void Restart()
    {
        Time.timeScale = 1f;
        SceneManager.LoadScene(SceneManager.GetActiveScene().buildIndex);
    }
}
