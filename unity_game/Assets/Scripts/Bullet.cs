using UnityEngine;

/// <summary>
/// A single bullet.
/// Moves in a straight line at the assigned speed and destroys itself
/// when it leaves the screen bounds (with a generous margin).
/// </summary>
[RequireComponent(typeof(Rigidbody2D), typeof(Collider2D))]
public class Bullet : MonoBehaviour
{
    private Vector2 _direction;
    private float _speed;
    private Rigidbody2D _rb;
    private Camera _cam;

    private void Awake()
    {
        _rb = GetComponent<Rigidbody2D>();
        _rb.gravityScale = 0f;
        _cam = Camera.main;

        // Ensure correct tag for collision detection
        gameObject.tag = "Bullet";
    }

    /// <summary>Set movement parameters right after instantiation.</summary>
    public void Initialize(Vector2 direction, float speed)
    {
        _direction = direction.normalized;
        _speed = speed;

        // Rotate sprite to face direction of travel
        float angle = Mathf.Atan2(_direction.y, _direction.x) * Mathf.Rad2Deg;
        transform.rotation = Quaternion.Euler(0f, 0f, angle - 90f);
    }

    private void FixedUpdate()
    {
        _rb.linearVelocity = _direction * _speed;
    }

    private void Update()
    {
        // Destroy when well outside the camera view
        Vector3 vp = _cam.WorldToViewportPoint(transform.position);
        if (vp.x < -0.2f || vp.x > 1.2f || vp.y < -0.2f || vp.y > 1.2f)
            Destroy(gameObject);
    }
}
