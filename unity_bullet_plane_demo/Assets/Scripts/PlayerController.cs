using System.Collections;
using UnityEngine;

[RequireComponent(typeof(Rigidbody2D))]
public class PlayerController : MonoBehaviour
{
    // ── Inspector ──────────────────────────────────────────────────────────
    [Header("Movement")]
    public float speed = 8f;

    [Header("Play Area")]
    public Vector2 minBoundary = new Vector2(-8.5f, -4.5f);
    public Vector2 maxBoundary = new Vector2( 8.5f,  4.5f);

    [Header("HP")]
    public int maxHP = 3;

    [Header("Absorb")]
    [Tooltip("Radius within which a special bullet is auto-absorbed (0 = only on direct hit)")]
    public float absorbRadius    = 0f;
    public float xpPerAbsorb     = 3f;

    [Header("Feedback")]
    public float hitFlashSeconds = 0.15f;

    // ── Public state (read by other scripts) ──────────────────────────────
    public int  CurrentHP       { get; private set; }
    public bool IsInvincible    { get; private set; }

    // ── Private ────────────────────────────────────────────────────────────
    private Rigidbody2D    rb;
    private SpriteRenderer sr;
    private Vector2        inputDir;
    private float          invincibleUntil;

    // ── Unity lifecycle ────────────────────────────────────────────────────

    private void Awake()
    {
        rb              = GetComponent<Rigidbody2D>();
        sr              = GetComponent<SpriteRenderer>();
        rb.gravityScale = 0f;
        rb.linearDamping  = 8f;
        rb.angularDamping = 8f;
        CurrentHP       = maxHP;
    }

    private void Update()
    {
        inputDir.x = Input.GetAxisRaw("Horizontal");
        inputDir.y = Input.GetAxisRaw("Vertical");
        if (inputDir.sqrMagnitude > 1f) inputDir.Normalize();

        IsInvincible = Time.time < invincibleUntil;

        // ── Proximity absorb (if absorbRadius > 0) ────────────────────────
        if (absorbRadius > 0f && GameManager.Instance != null
            && GameManager.Instance.State == GameManager.GameState.Playing)
        {
            TryAbsorbNearbySpecialBullets();
        }
    }

    private void FixedUpdate()
    {
        if (GameManager.Instance != null && GameManager.Instance.isGameOver)
        {
            rb.linearVelocity = Vector2.zero;
            return;
        }

        rb.linearVelocity = inputDir * speed;

        Vector2 pos   = rb.position;
        pos.x         = Mathf.Clamp(pos.x, minBoundary.x, maxBoundary.x);
        pos.y         = Mathf.Clamp(pos.y, minBoundary.y, maxBoundary.y);
        rb.position   = pos;
    }

    // ── Collision ──────────────────────────────────────────────────────────

    private void OnCollisionEnter2D(Collision2D col)
    {
        HandleBulletHit(col.collider);
    }

    private void OnTriggerEnter2D(Collider2D other)
    {
        HandleBulletHit(other);
    }

    // ── Helpers ────────────────────────────────────────────────────────────

    private void HandleBulletHit(Collider2D col)
    {
        if (!col.CompareTag("Bullet")) return;

        Bullet b = col.GetComponent<Bullet>();
        if (b != null && b.isSpecial)
        {
            AbsorbBullet(b.gameObject);
        }
        else
        {
            TakeDamage(col.gameObject);
        }
    }

    private void AbsorbBullet(GameObject bulletObj)
    {
        Destroy(bulletObj);
        GameManager.Instance?.AddXP(xpPerAbsorb);
        StartCoroutine(FlashColor(new Color(1f, 0.85f, 0f))); // gold flash
    }

    private void TakeDamage(GameObject bulletObj)
    {
        if (IsInvincible) return;

        Destroy(bulletObj);
        CurrentHP--;

        if (CurrentHP <= 0)
        {
            GameManager.Instance?.GameOver();
        }
        else
        {
            invincibleUntil = Time.time + hitFlashSeconds * 6f;
            StartCoroutine(FlashColor(Color.red));
        }
    }

    private void TryAbsorbNearbySpecialBullets()
    {
        Collider2D[] hits = Physics2D.OverlapCircleAll(transform.position, absorbRadius);
        foreach (Collider2D hit in hits)
        {
            if (!hit.CompareTag("Bullet")) continue;
            Bullet b = hit.GetComponent<Bullet>();
            if (b != null && b.isSpecial)
                AbsorbBullet(hit.gameObject);
        }
    }

    private IEnumerator FlashColor(Color flashCol)
    {
        if (sr == null) yield break;
        Color orig = sr.color;
        sr.color   = flashCol;
        yield return new WaitForSeconds(hitFlashSeconds);
        sr.color   = orig;
    }

    // ── Public API (called by WaveManager / upgrades) ─────────────────────

    public void SetAbsorbRadius(float r) => absorbRadius = r;
    public void SetSpeed(float s)        => speed        = s;
    public void AddHP(int amount)
    {
        CurrentHP = Mathf.Min(CurrentHP + amount, maxHP + amount); // allow over-heal cap extension
        maxHP     = Mathf.Max(maxHP, CurrentHP);
    }
}
