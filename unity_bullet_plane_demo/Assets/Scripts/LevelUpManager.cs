using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

/// <summary>
/// Displays a 3-choice upgrade panel when the player levels up.
/// Uses Time.timeScale = 0 (set by GameManager) while the panel is visible.
///
/// Scene setup:
///   - LevelUpPanel: a Canvas panel with 3 child buttons (upgradeButtons[0..2]).
///   - Each button needs a Text child for the label.
///   - Assign upgradeButtons in the Inspector.
/// </summary>
public class LevelUpManager : MonoBehaviour
{
    // ── Upgrade catalogue ──────────────────────────────────────────────────
    public enum UpgradeType
    {
        FireRate,       // auto-weapon fires 30% faster
        MoveSpeed,      // player moves 15% faster
        MultiShot,      // +1 bullet per shot
        AbsorbRadius,   // auto-absorb special bullets in radius +0.8
        ExtraHP,        // +1 max HP (and restore 1 HP)
    }

    private static readonly string[] UpgradeNames =
    {
        "火力加速",
        "引擎强化",
        "多重射击",
        "吸收扩张",
        "护盾充能",
    };

    private static readonly string[] UpgradeDescriptions =
    {
        "武器射速 +30%",
        "移动速度 +15%",
        "每次多发射 1 颗子弹",
        "自动吸收范围 +0.8",
        "最大血量 +1，并恢复 1 HP",
    };

    // ── Inspector ──────────────────────────────────────────────────────────
    [Header("Panel")]
    public GameObject levelUpPanel;

    [Header("3 Upgrade Buttons (order: slot 0, 1, 2)")]
    public Button[] upgradeButtons = new Button[3];

    // ── Private state ──────────────────────────────────────────────────────
    private UpgradeType[]       offeredUpgrades = new UpgradeType[3];
    private PlayerController    playerCtrl;
    private AutoWeapon          autoWeapon;

    // ── Unity lifecycle ────────────────────────────────────────────────────

    private void Start()
    {
        playerCtrl = FindFirstObjectByType<PlayerController>();
        autoWeapon = FindFirstObjectByType<AutoWeapon>();

        if (levelUpPanel != null) levelUpPanel.SetActive(false);

        // Wire button callbacks
        for (int i = 0; i < upgradeButtons.Length; i++)
        {
            int idx = i; // capture for lambda
            if (upgradeButtons[i] != null)
                upgradeButtons[i].onClick.AddListener(() => SelectUpgrade(idx));
        }

        if (GameManager.Instance != null)
            GameManager.Instance.OnLevelUp += ShowLevelUpPanel;
    }

    private void OnDestroy()
    {
        if (GameManager.Instance != null)
            GameManager.Instance.OnLevelUp -= ShowLevelUpPanel;
    }

    // ── Panel logic ────────────────────────────────────────────────────────

    private void ShowLevelUpPanel()
    {
        if (levelUpPanel == null) return;

        PickRandomUpgrades();
        UpdateButtonLabels();
        levelUpPanel.SetActive(true);
    }

    private void SelectUpgrade(int slotIndex)
    {
        ApplyUpgrade(offeredUpgrades[slotIndex]);

        if (levelUpPanel != null) levelUpPanel.SetActive(false);
        GameManager.Instance?.ResumePlaying();
    }

    // ── Random selection (no duplicates) ──────────────────────────────────

    private void PickRandomUpgrades()
    {
        List<UpgradeType> pool = new List<UpgradeType>((UpgradeType[])System.Enum.GetValues(typeof(UpgradeType)));

        for (int i = 0; i < offeredUpgrades.Length; i++)
        {
            if (pool.Count == 0) { offeredUpgrades[i] = UpgradeType.FireRate; continue; }
            int pick = Random.Range(0, pool.Count);
            offeredUpgrades[i] = pool[pick];
            pool.RemoveAt(pick);
        }
    }

    private void UpdateButtonLabels()
    {
        for (int i = 0; i < upgradeButtons.Length; i++)
        {
            if (upgradeButtons[i] == null) continue;

            Text label = upgradeButtons[i].GetComponentInChildren<Text>();
            if (label == null) continue;

            UpgradeType t = offeredUpgrades[i];
            label.text = $"{UpgradeNames[(int)t]}\n<size=14>{UpgradeDescriptions[(int)t]}</size>";
        }
    }

    // ── Apply ──────────────────────────────────────────────────────────────

    private void ApplyUpgrade(UpgradeType type)
    {
        switch (type)
        {
            case UpgradeType.FireRate:
                autoWeapon?.IncreaseFireRate(1.3f);
                break;

            case UpgradeType.MoveSpeed:
                if (playerCtrl != null) playerCtrl.SetSpeed(playerCtrl.speed * 1.15f);
                break;

            case UpgradeType.MultiShot:
                autoWeapon?.AddShot();
                break;

            case UpgradeType.AbsorbRadius:
                if (playerCtrl != null) playerCtrl.SetAbsorbRadius(playerCtrl.absorbRadius + 0.8f);
                break;

            case UpgradeType.ExtraHP:
                playerCtrl?.AddHP(1);
                break;
        }

        Debug.Log($"[LevelUp] Applied upgrade: {type}");
    }
}
