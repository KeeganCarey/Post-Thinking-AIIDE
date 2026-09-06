using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

namespace PostThinkRP
{
    public class QuestController : MonoBehaviour
    {
        [SerializeField] private SessionManager sessionManager;
        [SerializeField] private Button huntButton;
        [SerializeField] private TMP_Text questStatusText;
        [SerializeField] private string[] requiredNpcIds = { "maid", "hunter" };

        [Header("Quest status copy (override per scenario)")]
        [SerializeField] private string introStatus = "Ask around about the wolves.";
        [SerializeField] private string readyStatus = "You have enough advice to hunt the wolves.";
        [SerializeField] private string inProgressStatus = "The hunt is underway...";
        [SerializeField] private string doneStatus = "The wolves have been dealt with. Speak to the villagers again.";

        [Tooltip("Player line auto-sent the first time an NPC is re-opened after " +
                 "the quest is completed, so the NPC reacts to the deed.")]
        [SerializeField] private string questCompleteAnnouncement =
            "I've come back. The wolves have been dealt with.";

        private readonly HashSet<string> talkedTo = new HashSet<string>();
        private bool questReady;
        private bool questCompleted;
        private bool completionInFlight;

        private readonly HashSet<string> talkedAfterComplete = new HashSet<string>();
        private bool debriefDone;

        public bool ReadyToLeave => questReady && !questCompleted && !completionInFlight;

        public bool QuestCompleted => questCompleted;
        public string QuestCompleteAnnouncement => questCompleteAnnouncement;

        public bool DebriefDone => debriefDone;

        private void Awake()
        {
            if (sessionManager == null)
            {
                sessionManager = FindObjectOfType<SessionManager>();
            }

            if (huntButton != null)
            {
                huntButton.onClick.AddListener(CompleteQuest);
                huntButton.gameObject.SetActive(false);
            }
            UpdateStatus(introStatus);
        }

        public void ConfigureScenario(string[] required, string intro, string ready,
            string inProgress, string done, string announcement)
        {
            requiredNpcIds = required;
            introStatus = intro;
            readyStatus = ready;
            inProgressStatus = inProgress;
            doneStatus = done;
            questCompleteAnnouncement = announcement;

            talkedTo.Clear();
            questReady = false;
            questCompleted = false;
            completionInFlight = false;
            talkedAfterComplete.Clear();
            debriefDone = false;
            if (huntButton != null)
            {
                huntButton.interactable = true;
                huntButton.gameObject.SetActive(false);
            }
            UpdateStatus(introStatus);
        }

        public void NotifyNpcTalked(string npcId)
        {
            if (questCompleted)
            {
                if (!debriefDone)
                {
                    talkedAfterComplete.Add(npcId);
                    if (HasTalkedToRequiredNpcsAfterComplete())
                    {
                        debriefDone = true;
                    }
                }
                return;
            }

            talkedTo.Add(npcId);
            if (!questReady && HasTalkedToRequiredNpcs())
            {
                questReady = true;
                if (huntButton != null)
                {
                    huntButton.gameObject.SetActive(true);
                }
                UpdateStatus(readyStatus);
            }
        }

        public void RequestCompletion()
        {
            CompleteQuest();
        }

        private bool HasTalkedToRequiredNpcs()
        {
            foreach (var npcId in requiredNpcIds)
            {
                if (!talkedTo.Contains(npcId))
                {
                    return false;
                }
            }
            return true;
        }

        private bool HasTalkedToRequiredNpcsAfterComplete()
        {
            foreach (var npcId in requiredNpcIds)
            {
                if (!talkedAfterComplete.Contains(npcId))
                {
                    return false;
                }
            }
            return true;
        }

        private void CompleteQuest()
        {
            if (completionInFlight || questCompleted || !questReady || sessionManager == null)
            {
                return;
            }

            completionInFlight = true;
            if (huntButton != null)
            {
                huntButton.interactable = false;
            }
            UpdateStatus(inProgressStatus);

            StartCoroutine(sessionManager.CompleteQuest(
                _ =>
                {
                    questCompleted = true;
                    completionInFlight = false;
                    if (huntButton != null)
                    {
                        huntButton.gameObject.SetActive(false);
                    }
                    UpdateStatus(doneStatus);
                },
                error =>
                {
                    completionInFlight = false;
                    if (huntButton != null)
                    {
                        huntButton.interactable = true;
                    }
                    UpdateStatus($"Could not complete quest: {error}");
                }));
        }

        private void UpdateStatus(string text)
        {
            if (questStatusText != null)
            {
                questStatusText.text = text;
            }
        }
    }
}
