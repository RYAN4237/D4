using UnityEngine;

/// <summary>
/// Moves the player airplane with WASD / Arrow Keys.
/// Keeps the plane within the screen bounds.
/// Ends the game on any collision (bullets, monsters, etc.).
/// </summary>
[RequireComponent(typeof(Rigidbody2D), typeof(Collider2D))]
public class PlayerController : MonoBehaviour
{
    [Header("Movement")]
    public float moveSpeed = 6f;

    [Header("Shield (from reward)")]
    public bool hasShield = false;
    public float shieldDuration = 5f;

    // Visual feedback for shield
    public GameObject shieldEffect;

    private Rigidbody2D _rb;
    private Vector2 _input;
    private Camera _cam;
    private float _shieldTimer;

    // Half-extents of the camera's orthographic view
    private float _halfW;
    private float _halfH;

    private void Awake()
    {
        _rb = GetComponent<Rigidbody2D>();
        _rb.gravityScale = 0f;
        _rb.constraints = RigidbodyConstraints2D.FreezeRotation;
        _cam = Camera.main;
    }

    private void Start()
    {
        _halfH = _cam.orthographicSize;
        _halfW = _halfH * _cam.aspect;
        if (shieldEffect != null) shieldEffect.SetActive(false);
    }

    private void Update()
    {
        if (GameManager.Instance.IsGameOver) return;

        _input.x = Input.GetAxisRaw("Horizontal");
        _input.y = Input.GetAxisRaw("Vertical");
        _input = _input.normalized;

        // Shield countdown
        if (hasShield)
        {
            _shieldTimer -= Time.deltaTime;
            if (_shieldTimer <= 0f)
            {
                hasShield = false;
                if (shieldEffect != null) shieldEffect.SetActive(false);
            }
        }
    }

    private void FixedUpdate()
    {
        if (GameManager.Instance.IsGameOver)
        {
            _rb.linearVelocity = Vector2.zero;
            return;
        }

        _rb.linearVelocity = _input * moveSpeed;

        // Clamp position inside camera bounds
        Vector2 pos = _rb.position;
        float margin = 0.3f;
        pos.x = Mathf.Clamp(pos.x, -_halfW + margin, _halfW - margin);
        pos.y = Mathf.Clamp(pos.y, -_halfH + margin, _halfH - margin);
        _rb.position = pos;
    }

    private void OnTriggerEnter2D(Collider2D other)
    {
        if (GameManager.Instance.IsGameOver) return;

        // Ignore reward layer
        if (other.CompareTag("Reward"))
        {
            other.GetComponent<RewardItem>()?.Apply(this);
            Destroy(other.gameObject);
            return;
        }

        // Shield absorbs one hit
        if (hasShield)
        {
            hasShield = false;
            if (shieldEffect != null) shieldEffect.SetActive(false);
            // Destroy the hazard that hit us
            Destroy(other.gameObject);
            return;
        }

        // Anything else → game over
        if (other.CompareTag("Bullet") || other.CompareTag("Monster"))
        {
            GameManager.Instance.TriggerGameOver();
        }
    }

    /// <summary>Grant a temporary shield (called by RewardItem).</summary>
    public void ActivateShield()
    {
        hasShield = true;
        _shieldTimer = shieldDuration;
        if (shieldEffect != null) shieldEffect.SetActive(true);
    }

    /// <summary>Increase move speed temporarily (called by RewardItem).</summary>
    public void BoostSpeed(float extra, float duration)
    {
        StartCoroutine(SpeedBoostCoroutine(extra, duration));
    }

    private System.Collections.IEnumerator SpeedBoostCoroutine(float extra, float duration)
    {
        moveSpeed += extra;
        yield return new WaitForSeconds(duration);
        moveSpeed -= extra;
    }
}
