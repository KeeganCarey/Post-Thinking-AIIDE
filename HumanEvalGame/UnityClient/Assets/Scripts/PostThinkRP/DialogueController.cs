using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

namespace PostThinkRP
{
    public class DialogueController : MonoBehaviour
    {
        [Header("Dependencies")]
        [SerializeField] private SessionManager sessionManager;
        [SerializeField] private QuestController questController;

        [Header("UI")]
        [SerializeField] private GameObject dialoguePanel;
        [SerializeField] private TMP_Text npcNameText;
        [SerializeField] private TMP_Text transcriptText;
        [SerializeField] private TMP_Text thinkingText;
        [SerializeField] private TMP_InputField inputField;
        [SerializeField] private Button sendButton;
        [SerializeField] private Button closeButton;

        [Header("Typewriter")]
        [SerializeField] private float charactersPerSecond = 45f;
        [SerializeField] private int maxTranscriptLines = 200;

        private readonly List<string> transcriptLines = new List<string>();
        private NpcInteractable currentNpc;
        private Coroutine typewriterRoutine;
        private bool requestInFlight;
        private ScrollRect transcriptScroll;

        private void Awake()
        {
            if (sessionManager == null)
            {
                sessionManager = FindObjectOfType<SessionManager>();
            }

            if (questController == null)
            {
                questController = FindObjectOfType<QuestController>();
            }

            if (sendButton != null)
            {
                sendButton.onClick.AddListener(SendCurrentInput);
            }

            if (closeButton != null)
            {
                closeButton.onClick.AddListener(CloseDialogue);
            }

            SetupTranscriptScroll();

            if (dialoguePanel != null)
            {
                dialoguePanel.SetActive(false);
            }

            SetThinking(false);
        }

        private void SetupTranscriptScroll()
        {
            if (transcriptText == null || transcriptScroll != null)
            {
                return;
            }

            var content = transcriptText.rectTransform;
            if (!(content.parent is RectTransform parent))
            {
                return;
            }

            var viewportGO = new GameObject(
                "TranscriptViewport",
                typeof(RectTransform), typeof(RectMask2D), typeof(ScrollRect));
            var viewport = viewportGO.GetComponent<RectTransform>();
            viewport.SetParent(parent, false);
            viewport.SetSiblingIndex(content.GetSiblingIndex());
            viewport.anchorMin = content.anchorMin;
            viewport.anchorMax = content.anchorMax;
            viewport.pivot = content.pivot;
            viewport.anchoredPosition = content.anchoredPosition;
            viewport.sizeDelta = content.sizeDelta;

            content.SetParent(viewport, false);
            content.anchorMin = new Vector2(0f, 1f);
            content.anchorMax = new Vector2(1f, 1f);
            content.pivot = new Vector2(0.5f, 1f);
            content.anchoredPosition = Vector2.zero;
            content.sizeDelta = new Vector2(0f, content.sizeDelta.y);

            transcriptText.textWrappingMode = TextWrappingModes.Normal;
            transcriptText.overflowMode = TextOverflowModes.Overflow;
            transcriptText.verticalAlignment = VerticalAlignmentOptions.Top;

            var fitter = content.gameObject.AddComponent<ContentSizeFitter>();
            fitter.horizontalFit = ContentSizeFitter.FitMode.Unconstrained;
            fitter.verticalFit = ContentSizeFitter.FitMode.PreferredSize;

            var scroll = viewportGO.GetComponent<ScrollRect>();
            scroll.viewport = viewport;
            scroll.content = content;
            scroll.horizontal = false;
            scroll.vertical = true;
            scroll.movementType = ScrollRect.MovementType.Clamped;
            scroll.scrollSensitivity = 28f;
            transcriptScroll = scroll;
        }

        private void ScrollToBottom()
        {
            if (transcriptScroll == null || !isActiveAndEnabled)
            {
                return;
            }
            Canvas.ForceUpdateCanvases();
            transcriptScroll.verticalNormalizedPosition = 0f;
            Canvas.ForceUpdateCanvases();
        }

        public void OpenNpc(NpcInteractable npc)
        {
            currentNpc = npc;
            transcriptLines.Clear();
            if (transcriptText != null)
            {
                transcriptText.text = string.Empty;
            }
            if (npcNameText != null)
            {
                npcNameText.text = npc.DisplayName;
            }
            if (dialoguePanel != null)
            {
                dialoguePanel.SetActive(true);
            }
            UiState.DialogueOpen = true;
            Cursor.lockState = CursorLockMode.None;
            Cursor.visible = true;
            if (inputField != null)
            {
                inputField.text = string.Empty;
            }

            if (sessionManager != null)
            {
                SetThinking(true);
                requestInFlight = true;
                StartCoroutine(LoadNpcConversation(npc));
            }
            else
            {
                inputField?.ActivateInputField();
            }
        }

        private IEnumerator LoadNpcConversation(NpcInteractable npc)
        {
            yield return sessionManager.GetHistory(
                npc.NpcId,
                resp =>
                {
                    if (currentNpc != npc || !UiState.DialogueOpen)
                    {
                        return;
                    }
                    if (resp != null && resp.turns != null && resp.turns.Length > 0)
                    {
                        RestoreHistory(npc.DisplayName, resp.turns);
                        requestInFlight = false;
                        SetThinking(false);
                        if (resp.quest_completed && !HistoryHasAnnouncement(resp))
                        {
                            SendPlayerTurn(QuestAnnouncementText());
                        }
                        else
                        {
                            inputField?.ActivateInputField();
                        }
                    }
                    else
                    {
                        StartGreeting(npc);
                    }
                },
                error =>
                {
                    if (currentNpc != npc || !UiState.DialogueOpen)
                    {
                        return;
                    }
                    StartGreeting(npc);
                });
        }

        private void StartGreeting(NpcInteractable npc)
        {
            SetThinking(true);
            requestInFlight = true;
            StartCoroutine(sessionManager.RequestGreeting(
                npc.NpcId,
                response =>
                {
                    if (currentNpc != npc || !UiState.DialogueOpen)
                    {
                        return;
                    }
                    var clean = TagSanitizer.StripHiddenTags(response.dialogue);
                    StartTypewriter(npc.DisplayName, clean);
                    requestInFlight = false;
                    SetThinking(false);
                    inputField?.ActivateInputField();
                },
                error =>
                {
                    if (currentNpc != npc || !UiState.DialogueOpen)
                    {
                        return;
                    }
                    requestInFlight = false;
                    SetThinking(false);
                    inputField?.ActivateInputField();
                }));
        }

        private void RestoreHistory(string npcName, PostThinkApiClient.HistoryTurn[] turns)
        {
            transcriptLines.Clear();
            foreach (var turn in turns)
            {
                if (!string.IsNullOrEmpty(turn.player_message))
                {
                    AppendLine("You", turn.player_message);
                }
                var clean = TagSanitizer.StripHiddenTags(turn.dialogue);
                if (!string.IsNullOrEmpty(clean))
                {
                    AppendLine(npcName, clean);
                }
            }
        }

        private bool HistoryHasAnnouncement(PostThinkApiClient.HistoryResponse resp)
        {
            var announcement = QuestAnnouncementText();
            foreach (var turn in resp.turns)
            {
                if (turn.player_message == announcement)
                {
                    return true;
                }
            }
            return false;
        }

        private string QuestAnnouncementText()
        {
            if (questController != null && !string.IsNullOrEmpty(questController.QuestCompleteAnnouncement))
            {
                return questController.QuestCompleteAnnouncement;
            }
            return "I've returned from the hunt.";
        }

        public void CloseDialogue()
        {
            if (dialoguePanel != null)
            {
                dialoguePanel.SetActive(false);
            }
            UiState.DialogueOpen = false;
            Cursor.lockState = CursorLockMode.Locked;
            Cursor.visible = false;
            currentNpc = null;
        }

        private void Update()
        {
            if (UiState.DialogueOpen && Input.GetKeyDown(KeyCode.Escape))
            {
                CloseDialogue();
            }
        }

        public void SendCurrentInput()
        {
            if (requestInFlight || currentNpc == null || sessionManager == null)
            {
                return;
            }

            if (inputField == null)
            {
                AppendLine("System", "Dialogue input is not configured.");
                return;
            }

            var message = inputField.text.Trim();
            if (string.IsNullOrEmpty(message))
            {
                return;
            }

            inputField.text = string.Empty;
            SendPlayerTurn(message);
        }

        private void SendPlayerTurn(string message)
        {
            if (requestInFlight || currentNpc == null || sessionManager == null)
            {
                return;
            }

            var npc = currentNpc;
            AppendLine("You", message);
            SetThinking(true);
            requestInFlight = true;

            StartCoroutine(sessionManager.SendChat(
                npc.NpcId,
                message,
                response =>
                {
                    questController?.NotifyNpcTalked(npc.NpcId);
                    if (currentNpc != npc || !UiState.DialogueOpen)
                    {
                        return;
                    }
                    var clean = TagSanitizer.StripHiddenTags(response.dialogue);
                    StartTypewriter(npc.DisplayName, clean);
                    requestInFlight = false;
                    SetThinking(false);
                    inputField?.ActivateInputField();
                },
                error =>
                {
                    if (currentNpc != npc || !UiState.DialogueOpen)
                    {
                        requestInFlight = false;
                        SetThinking(false);
                        return;
                    }
                    AppendLine("System", $"Connection error: {error}");
                    requestInFlight = false;
                    SetThinking(false);
                    inputField?.ActivateInputField();
                }));
        }

        private void StartTypewriter(string speaker, string text)
        {
            if (typewriterRoutine != null)
            {
                StopCoroutine(typewriterRoutine);
            }

            typewriterRoutine = StartCoroutine(TypewriterLine(speaker, text));
        }

        private IEnumerator TypewriterLine(string speaker, string text)
        {
            var prefix = $"{speaker}: ";
            var visible = string.Empty;
            for (var i = 0; i < text.Length; i++)
            {
                visible += text[i];
                RenderWithDraft(prefix + visible);
                yield return new WaitForSeconds(1f / Mathf.Max(1f, charactersPerSecond));
            }

            AppendLine(speaker, text);
            typewriterRoutine = null;
        }

        private void AppendLine(string speaker, string text)
        {
            transcriptLines.Add($"{speaker}: {text}");
            while (transcriptLines.Count > maxTranscriptLines)
            {
                transcriptLines.RemoveAt(0);
            }
            if (transcriptText != null)
            {
                transcriptText.text = string.Join("\n", transcriptLines);
            }
            ScrollToBottom();
        }

        private void RenderWithDraft(string draft)
        {
            var lines = new List<string>(transcriptLines);
            lines.Add(draft);
            while (lines.Count > maxTranscriptLines)
            {
                lines.RemoveAt(0);
            }
            if (transcriptText != null)
            {
                transcriptText.text = string.Join("\n", lines);
            }
            ScrollToBottom();
        }

        private void SetThinking(bool active)
        {
            if (thinkingText != null)
            {
                thinkingText.gameObject.SetActive(active);
                thinkingText.text = "thinking...";
            }

            if (sendButton != null)
            {
                sendButton.interactable = !active;
            }
        }
    }
}
