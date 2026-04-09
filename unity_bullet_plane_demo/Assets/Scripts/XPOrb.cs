using UnityEngine;

/// <summary>
/// XP pickup dropped by KamikazeEnemy on death.
/// Auto-collected when the player enters the trigger.
/// </summary>
public class XPOrb : MonoBehaviour
{
    [Header("Value")]
    public float xpValue = 5f;

    [Header("Magnet")]
    [Tooltip("At what distance the orb starts flying toward the player")]
    public float magnetRadius = 3f;
    public float magnetSpeed  = 8f;

    private Transform playerTransform;

    private void Start()
    {
        PlayerController pc = FindFirstObjectByType<PlayerController>();
        if (pc != null) playerTransform = pc.transform;
    }

    private void Update()
    {
        if (GameManager.Instance == null) return;
        if (GameManager.Instance.State != GameManager.GameState.Playing) return;
        if (playerTransform == null) return;

        float dist = Vector2.Distance(transform.position, playerTransform.position);
        if (dist < magnetRadius)
        {
            Vector2 dir = ((Vector2)playerTransform.position - (Vector2)transform.position).normalized;
            transform.position += (Vector3)(dir * magnetSpeed * Time.deltaTime);
        }
    }

    private void OnTriggerEnter2D(Collider2D other)
    {
        if (!other.CompareTag("Player")) return;

        GameManager.Instance?.AddXP(xpValue);
        Destroy(gameObject);
    }
}
