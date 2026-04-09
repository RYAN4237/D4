using UnityEngine;
using UnityEngine.UI;

/// <summary>
/// Drives all HUD elements: survive time, XP bar, level, wave, HP, game-over text.
/// Assign references in the Inspector.
/// </summary>
public class HudController : MonoBehaviour
{
    [Header("Text Labels")]
    public Text surviveTimeText;
    public Text gameOverText;
    public Text levelText;
    public Text waveText;
    public Text hpText;

    [Header("XP Bar (optional Image set to Filled)")]
    public Image xpBarFill;

    private void Start()
    {
        if (gameOverText != null) gameOverText.gameObject.SetActive(false);
        if (xpBarFill    != null) xpBarFill.fillAmount = 0f;
    }

    private void Update()
    {
        GameManager gm = GameManager.Instance;
        if (gm == null) return;

        // ── Survive time ──────────────────────────────────────────────────
        if (surviveTimeText != null)
            surviveTimeText.text = "时间: " + gm.SurviveTime.ToString("F1") + "s";

        // ── Level ─────────────────────────────────────────────────────────
        if (levelText != null)
            levelText.text = "LV " + gm.Level;

        // ── XP bar ────────────────────────────────────────────────────────
        if (xpBarFill != null && gm.XPThreshold > 0f)
            xpBarFill.fillAmount = gm.CurrentXP / gm.XPThreshold;

        // ── HP ────────────────────────────────────────────────────────────
        if (hpText != null)
        {
            PlayerController player = FindFirstObjectByType<PlayerController>();
            if (player != null)
                hpText.text = "HP: " + new string('♥', player.CurrentHP);
        }

        // ── Wave ──────────────────────────────────────────────────────────
        if (waveText != null)
        {
            WaveManager wm = WaveManager.Instance;
            if (wm != null) waveText.text = "Wave " + wm.CurrentWave;
        }

        // ── Game Over ─────────────────────────────────────────────────────
        if (gameOverText != null)
        {
            bool over = gm.isGameOver;
            gameOverText.gameObject.SetActive(over);
            if (over)
                gameOverText.text = $"游戏失败\n生存时间: {gm.SurviveTime:F1}s\n按 R 重新开始";
        }
    }
}
