using System.Collections;
using UnityEngine;

/// <summary>
/// Auto-fires PlayerProjectile toward the nearest KamikazeEnemy.
/// If no enemies are present, fires straight up.
/// Attach to the Player object alongside PlayerController.
/// </summary>
public class AutoWeapon : MonoBehaviour
{
    [Header("Prefab")]
    public GameObject projectilePrefab;

    [Header("Stats")]
    public float fireRate   = 0.5f;  // shots per second
    public int   shotCount  = 1;     // multishot (upgraded by LevelUpManager)
    public float spreadDeg  = 15f;   // spread angle between multishot bullets

    private float nextFireTime;

    private void Update()
    {
        if (GameManager.Instance == null) return;
        if (GameManager.Instance.State != GameManager.GameState.Playing) return;

        if (Time.time >= nextFireTime)
        {
            Fire();
            nextFireTime = Time.time + 1f / Mathf.Max(fireRate, 0.01f);
        }
    }

    private void Fire()
    {
        if (projectilePrefab == null) return;

        Vector2 aimDir = GetAimDirection();

        if (shotCount <= 1)
        {
            SpawnProjectile(aimDir);
            return;
        }

        // Multishot: fan out around aimDir
        float halfSpread = spreadDeg * (shotCount - 1) / 2f;
        for (int i = 0; i < shotCount; i++)
        {
            float   angle  = -halfSpread + spreadDeg * i;
            Vector2 rotDir = Quaternion.Euler(0, 0, angle) * aimDir;
            SpawnProjectile(rotDir);
        }
    }

    private void SpawnProjectile(Vector2 direction)
    {
        GameObject proj = Instantiate(projectilePrefab, transform.position, Quaternion.identity);
        PlayerProjectile pp = proj.GetComponent<PlayerProjectile>();
        if (pp != null) pp.Launch(direction);
    }

    private Vector2 GetAimDirection()
    {
        KamikazeEnemy nearest = FindNearestEnemy();
        if (nearest != null)
            return ((Vector2)nearest.transform.position - (Vector2)transform.position).normalized;

        return Vector2.up; // default: shoot upward
    }

    private KamikazeEnemy FindNearestEnemy()
    {
        KamikazeEnemy[] enemies = FindObjectsByType<KamikazeEnemy>(FindObjectsSortMode.None);
        KamikazeEnemy nearest   = null;
        float          minDist  = float.MaxValue;

        foreach (KamikazeEnemy e in enemies)
        {
            float d = Vector2.SqrMagnitude((Vector2)e.transform.position - (Vector2)transform.position);
            if (d < minDist) { minDist = d; nearest = e; }
        }

        return nearest;
    }

    // ── Public API (called by LevelUpManager) ──────────────────────────────

    public void IncreaseFireRate(float multiplier) => fireRate  *= multiplier;
    public void AddShot()                           => shotCount++;
}
