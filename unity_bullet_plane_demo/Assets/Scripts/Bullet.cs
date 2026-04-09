using UnityEngine;

[RequireComponent(typeof(Rigidbody2D))]
public class Bullet : MonoBehaviour
{
    [Header("Lifetime")]
    public float lifeSeconds = 10f;

    // Set by BulletSpawner before Launch()
    [HideInInspector] public bool isSpecial;

    private static readonly Color NormalColor  = Color.white;
    private static readonly Color SpecialColor = new Color(1f, 0.85f, 0f); // gold

    private Rigidbody2D rb;
    private SpriteRenderer sr;

    private void Awake()
    {
        rb = GetComponent<Rigidbody2D>();
        rb.gravityScale            = 0f;
        rb.collisionDetectionMode  = CollisionDetectionMode2D.Continuous;

        sr = GetComponent<SpriteRenderer>();
    }

    private void OnEnable()
    {
        Destroy(gameObject, lifeSeconds);
    }

    /// <summary>
    /// Fires the bullet. Must be called after isSpecial is set so colour is correct.
    /// </summary>
    public void Launch(Vector2 direction, float speed)
    {
        if (sr != null)
            sr.color = isSpecial ? SpecialColor : NormalColor;

        rb.linearVelocity = direction.normalized * speed;
    }
}
