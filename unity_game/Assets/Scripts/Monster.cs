using UnityEngine;

/// <summary>
/// A monster that slowly homes in on the player.
/// Ends the game if it touches the player.
/// Destroys itself when far outside the screen.
/// </summary>
[RequireComponent(typeof(Rigidbody2D), typeof(Collider2D))]
public class Monster : MonoBehaviour
{
    [Header("Movement")]
    [Tooltip("How fast the monster moves toward the player")]
    public float speed = 2f;
    [Tooltip("How strongly it homes – 0 = straight line, 1 = instant turn")]
    [Range(0f, 1f)]
    public float homingStrength = 0.04f;

    private Rigidbody2D _rb;
    private Camera _cam;
    private Transform _player;
    private Vector2 _moveDir;

    private void Awake()
    {
        _rb = GetComponent<Rigidbody2D>();
        _rb.gravityScale = 0f;
        _rb.constraints = RigidbodyConstraints2D.FreezeRotation;
        _cam = Camera.main;
        gameObject.tag = "Monster";
    }

    private void Start()
    {
        // Resolve player reference
        GameObject playerObj = GameObject.FindGameObjectWithTag("Player");
        if (playerObj != null) _player = playerObj.transform;

        // Initial direction: toward player or toward center
        Vector2 target = _player != null ? (Vector2)_player.position : Vector2.zero;
        _moveDir = (target - (Vector2)transform.position).normalized;
    }

    private void FixedUpdate()
    {
        if (GameManager.Instance.IsGameOver) return;

        // Softly home toward player
        if (_player != null)
        {
            Vector2 toPlayer = ((Vector2)_player.position - _rb.position).normalized;
            _moveDir = Vector2.Lerp(_moveDir, toPlayer, homingStrength).normalized;
        }

        _rb.linearVelocity = _moveDir * speed;

        // Face direction of travel
        float angle = Mathf.Atan2(_moveDir.y, _moveDir.x) * Mathf.Rad2Deg;
        transform.rotation = Quaternion.Euler(0f, 0f, angle - 90f);
    }

    private void Update()
    {
        // Destroy when well outside the camera view
        Vector3 vp = _cam.WorldToViewportPoint(transform.position);
        if (vp.x < -0.5f || vp.x > 1.5f || vp.y < -0.5f || vp.y > 1.5f)
            Destroy(gameObject);
    }
}
