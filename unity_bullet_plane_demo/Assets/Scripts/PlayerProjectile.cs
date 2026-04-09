using UnityEngine;

/// <summary>
/// A projectile fired by the player's AutoWeapon.
/// Damages KamikazeEnemy on contact and then destroys itself.
/// </summary>
[RequireComponent(typeof(Rigidbody2D))]
public class PlayerProjectile : MonoBehaviour
{
    [Header("Stats")]
    public int   damage      = 1;
    public float speed       = 14f;
    public float lifeSeconds = 3f;

    private Rigidbody2D rb;

    private void Awake()
    {
        rb                        = GetComponent<Rigidbody2D>();
        rb.gravityScale           = 0f;
        rb.collisionDetectionMode = CollisionDetectionMode2D.Continuous;
    }

    private void OnEnable()
    {
        Destroy(gameObject, lifeSeconds);
    }

    public void Launch(Vector2 direction)
    {
        rb.linearVelocity = direction.normalized * speed;
    }

    private void OnTriggerEnter2D(Collider2D other)
    {
        KamikazeEnemy enemy = other.GetComponent<KamikazeEnemy>();
        if (enemy != null)
        {
            enemy.TakeDamage(damage);
            Destroy(gameObject);
        }
    }
}
