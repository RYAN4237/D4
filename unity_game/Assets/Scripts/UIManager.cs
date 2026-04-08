using System.Collections;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

/// <summary>
/// Manages all on-screen UI elements:
///   - Score label
///   - Elapsed time label
///   - Current bullet speed label
///   - Reward notification banner
///   - Game-over panel with restart button
/// </summary>
public class UIManager : MonoBehaviour
{
    [Header("HUD")]
    public TMP_Text scoreText;
    public TMP_Text timeText;
    public TMP_Text speedText;

    [Header("Reward Notification")]
    public TMP_Text rewardText;
    public float notificationDuration = 2.5f;

    [Header("Game Over Panel")]
    public GameObject gameOverPanel;
    public TMP_Text finalScoreText;
    public Button restartButton;

    private Coroutine _notifCoroutine;

    private void Start()
    {
        if (gameOverPanel != null) gameOverPanel.SetActive(false);
        if (rewardText != null) rewardText.gameObject.SetActive(false);
        if (restartButton != null)
            restartButton.onClick.AddListener(() => GameManager.Instance.RestartGame());
    }

    /// <summary>Called every frame while game is running.</summary>
    public void UpdateHUD(int score, float elapsed, float bulletSpeed)
    {
        if (scoreText != null)
            scoreText.text = $"Score: {score}";

        if (timeText != null)
        {
            int m = Mathf.FloorToInt(elapsed / 60f);
            int s = Mathf.FloorToInt(elapsed % 60f);
            timeText.text = $"Time: {m:00}:{s:00}";
        }

        if (speedText != null)
            speedText.text = $"Bullet Speed: {bulletSpeed:F1}";
    }

    /// <summary>Show a temporary notification banner.</summary>
    public void ShowRewardNotification(string message)
    {
        if (rewardText == null) return;

        if (_notifCoroutine != null) StopCoroutine(_notifCoroutine);
        _notifCoroutine = StartCoroutine(NotificationCoroutine(message));
    }

    private IEnumerator NotificationCoroutine(string message)
    {
        rewardText.text = message;
        rewardText.gameObject.SetActive(true);
        yield return new WaitForSeconds(notificationDuration);
        rewardText.gameObject.SetActive(false);
    }

    /// <summary>Display the game-over screen.</summary>
    public void ShowGameOver(int finalScore)
    {
        if (gameOverPanel != null) gameOverPanel.SetActive(true);
        if (finalScoreText != null)
            finalScoreText.text = $"Final Score\n{finalScore}";
    }
}
