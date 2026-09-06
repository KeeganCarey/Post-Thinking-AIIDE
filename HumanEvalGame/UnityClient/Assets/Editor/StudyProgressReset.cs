using UnityEditor;
using UnityEngine;

public static class StudyProgressReset
{
    private static readonly string[] Keys =
    {
        "postthink_pair_json",
        "postthink_pair_part",
        "postthink_pair_done",
        "postthink_pair_participant",
    };

    [MenuItem("PostThink-RP/Reset Study Progress (clear saved pair)")]
    public static void Reset()
    {
        foreach (var key in Keys)
        {
            PlayerPrefs.DeleteKey(key);
        }
        PlayerPrefs.Save();
        Debug.Log("[PostThink-RP] Cleared saved study progress. Next Play starts a fresh /pair enrollment.");
    }
}
