using UnityEngine;
using UnityEngine.UI;

public class HudController : MonoBehaviour
{
    public Text surviveTimeText;
    public Text gameOverText;

    private void Start()
    {
        if (gameOverText != null)
        {
            gameOverText.gameObject.SetActive(false);
        }
    }

    private void Update()
    {
        if (GameManager.Instance == null)
        {
            return;
        }

        if (surviveTimeText != null)
        {
            surviveTimeText.text = "生存时间: " + GameManager.Instance.surviveTime.ToString("F1") + "s";
        }

        if (gameOverText != null)
        {
            gameOverText.gameObject.SetActive(GameManager.Instance.isGameOver);
            if (GameManager.Instance.isGameOver)
            {
                gameOverText.text = "游戏失败\n按 R 重新开始";
            }
        }
    }
}
