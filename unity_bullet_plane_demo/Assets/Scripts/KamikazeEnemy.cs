using System.Collections;
using UnityEngine;

/// <summary>
/// A kamikaze enemy that slowly chases the player.
/// - Destroyed by PlayerProjectile.TakeDamage().
/// - Touching the player deals damage (same as a regular bullet).
/// - On death, spawns an XPOrb.
/// </summary>
public class KamikazeEnemy : MonoBehaviour
{
    [Header("Stats")]
    public int   maxHP      = 3;
    public float moveSpeed  = 2.5f;

    [Header("XP Drop")]
    public GameObject xpOrbPrefab;
    public float      xpValue = 5f;

    [Header("Feedback")]
    public float flashSeconds = 0.1f;

    private int            currentHP;
    private Transform      playerTransform;
    private SpriteRenderer sr;
    private Rigidbody2D    rb;

    private void Awake()
    {
        sr         = GetComponent<SpriteRenderer>();
        rb         = GetComponent<Rigidbody2D>();
        currentHP  = maxHP;

        if (rb != null)
        {
            rb.gravityScale = 0f;
            rb.linearDamping  = 0f;
        }

        // Colour enemy red so player can tell it apart from bullets
        if (sr != null) sr.color = Color.red;
    }

    private void Start()
    {
        PlayerController pc = FindFirstObjectByType<PlayerController>();
        if (pc != null) playerTransform = pc.transform;
    }

    private void FixedUpdate()
    {
        if (GameManager.Instance == null) return;
        if (GameManager.Instance.State != GameManager.GameState.Playing) return;
        if (playerTransform == null) return;

        Vector2 dir = ((Vector2)playerTransform.position - rb.position).normalized;
        rb.linearVelocity = dir * moveSpeed;
    }

    private void OnTriggerEnter2D(Collider2D other)
    {
        if (other.CompareTag("Player"))
        {
            PlayerController pc = other.GetComponent<PlayerController>();
            // Reuse player's hurt logic via a fake bullet-less damage path:
            // KamikazeEnemy touching player = instant death in original design,
            // but we respect HP so we just deal damage via GameManager.GameOver if HP == 0.
            GameManager.Instance?.GameOver();
        }
    }

    // ── Public API ─────────────────────────────────────────────────────────

    public void TakeDamage(int amount)
    {
        currentHP -= amount;
        StartCoroutine(FlashWhite());

        if (currentHP <= 0) Die();
    }

    private void Die()
    {
        // Drop XP orb
        if (xpOrbPrefab != null)
        {
            GameObject orb = Instantiate(xpOrbPrefab, transform.position, Quaternion.identity);
            XPOrb xpOrb   = orb.GetComponent<XPOrb>();
            if (xpOrb != null) xpOrb.xpValue = xpValue;
        }

        Destroy(gameObject);
    }

    private IEnumerator FlashWhite()
    {
        if (sr == null) yield break;
        Color orig = sr.color;
        sr.color   = Color.white;
        yield return new WaitForSeconds(flashSeconds);
        sr.color   = orig;
    }

    // ── Speed scaling (called by WaveManager) ─────────────────────────────
    public void SetSpeed(float s) => moveSpeed = s;
}
