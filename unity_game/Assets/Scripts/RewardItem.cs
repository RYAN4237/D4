using UnityEngine;

/// <summary>
/// Defines all possible reward types and their effect when picked up.
/// </summary>
public enum RewardType
{
    Shield,       // Grants a temporary shield
    SpeedBoost,   // Temporarily increases player speed
    SlowBullets,  // Temporarily halves bullet speed
    ScoreBonus,   // Instantly adds bonus score points
    ClearScreen,  // Destroys all bullets and monsters on screen
}

/// <summary>
/// A floating reward item.  When collected by the player it applies
/// its effect and destroys itself.
/// </summary>
public class RewardItem : MonoBehaviour
{
    [Header("Reward")]
    public RewardType rewardType;

    [Header("Float animation")]
    public float floatAmplitude = 0.15f;
    public float floatFrequency = 2f;

    [Header("Auto-despawn")]
    public float lifeTime = 20f;

    private Vector3 _startPos;
    private float _age;

    private void Awake()
    {
        gameObject.tag = "Reward";
    }

    private void Start()
    {
        _startPos = transform.position;
    }

    private void Update()
    {
        // Gentle floating motion
        float y = _startPos.y + Mathf.Sin(Time.time * floatFrequency) * floatAmplitude;
        transform.position = new Vector3(_startPos.x, y, _startPos.z);

        // Slow spin
        transform.Rotate(0f, 0f, 90f * Time.deltaTime);

        _age += Time.deltaTime;
        if (_age >= lifeTime) Destroy(gameObject);
    }

    /// <summary>Apply this reward to the player.</summary>
    public void Apply(PlayerController player)
    {
        switch (rewardType)
        {
            case RewardType.Shield:
                player.ActivateShield();
                GameManager.Instance.UIManager?.ShowRewardNotification("🛡 Shield Activated!");
                break;

            case RewardType.SpeedBoost:
                player.BoostSpeed(3f, 8f);
                GameManager.Instance.UIManager?.ShowRewardNotification("⚡ Speed Boost!");
                break;

            case RewardType.SlowBullets:
                GameManager.Instance.ActivateBulletSlow(8f);
                GameManager.Instance.UIManager?.ShowRewardNotification("🐢 Bullets Slowed!");
                break;

            case RewardType.ScoreBonus:
                GameManager.Instance.AddBonusScore(100);
                GameManager.Instance.UIManager?.ShowRewardNotification("+100 Bonus Score!");
                break;

            case RewardType.ClearScreen:
                ClearAllHazards();
                GameManager.Instance.UIManager?.ShowRewardNotification("💥 Screen Cleared!");
                break;
        }
    }

    private void ClearAllHazards()
    {
        foreach (GameObject obj in GameObject.FindGameObjectsWithTag("Bullet"))
            Destroy(obj);
        foreach (GameObject obj in GameObject.FindGameObjectsWithTag("Monster"))
            Destroy(obj);
    }
}
