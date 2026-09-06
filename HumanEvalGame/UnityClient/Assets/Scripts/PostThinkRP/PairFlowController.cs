using System.Collections;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

namespace PostThinkRP
{
    public class PairFlowController : MonoBehaviour
    {
        public static PairFlowController Instance { get; private set; }

        [SerializeField] private PostThinkApiClient apiClient;
        [SerializeField] private SessionManager sessionManager;
        [SerializeField] private ScenarioController scenarioController;

        [Header("Part transition card")]
        [Tooltip("Optional overlay shown briefly when each playthrough begins, so " +
                 "the participant knows a new scenario/playthrough has started.")]
        [SerializeField] private GameObject transitionPanel;
        [SerializeField] private TMP_Text transitionText;
        [SerializeField] private float transitionSeconds = 3.5f;

        [Tooltip("Shown (and left up) when the whole study ends. The survey is " +
                 "handled separately/out of game.")]
        [SerializeField] private string endCardMessage = "Thanks for playing!";

        [Tooltip("Optional 'Copy ID' button on the end card. Wired automatically; " +
                 "starts hidden and is revealed with the end card. Copies the " +
                 "participant id to the clipboard so it can be pasted into the survey.")]
        [SerializeField] private Button copyIdButton;

        [Tooltip("Optional read-only input field on the end card. Filled with the " +
                 "participant id; players click it, select the text, and Ctrl+C. " +
                 "This is the plugin-free fallback that works even when the itch.io " +
                 "iframe blocks the Copy ID button's clipboard write.")]
        [SerializeField] private TMP_InputField participantIdField;

        private const string PrefPairKey = "postthink_pair_json";
        private const string PrefPartKey = "postthink_pair_part";
        private const string PrefDoneKey = "postthink_pair_done";
        private const string PrefParticipantKey = "postthink_pair_participant";

        private PostThinkApiClient.PairResponse _pair;
        private int _currentPart;
        private bool _ending;
        private string _participantId;

        public bool StudyComplete { get; private set; }

        private void Awake()
        {
            if (Instance != null && Instance != this)
            {
                Destroy(this);
                return;
            }
            Instance = this;

            if (copyIdButton != null)
            {
                copyIdButton.onClick.AddListener(CopyParticipantId);
                copyIdButton.gameObject.SetActive(false);
            }
            if (participantIdField != null)
            {
                participantIdField.gameObject.SetActive(false);
            }
        }

        private IEnumerator Start()
        {
            if (apiClient == null) apiClient = FindObjectOfType<PostThinkApiClient>();
            if (sessionManager == null) sessionManager = FindObjectOfType<SessionManager>();
            if (scenarioController == null) scenarioController = FindObjectOfType<ScenarioController>();

            if (apiClient == null || sessionManager == null || scenarioController == null)
            {
                Debug.LogError("PairFlowController: missing apiClient/sessionManager/scenarioController.");
                yield break;
            }

            yield return apiClient.Initialize();
            yield return RunPairFlow();
        }

        private IEnumerator RunPairFlow()
        {
            if (PlayerPrefs.GetInt(PrefDoneKey, 0) == 1)
            {
                StudyComplete = true;
                ShowEndCard();
                yield break;
            }

            if (TryResumeSavedPair())
            {
                yield break;
            }

            yield return apiClient.CreatePair(
                res =>
                {
                    _pair = res;
                    ConfigurePart(1);
                },
                err =>
                {
                    Debug.LogError($"/pair failed: {err}");
                    ShowError(
                        $"Couldn't reach the game server.\n{err}",
                        () => StartCoroutine(RunPairFlow()));
                });
        }

        private bool TryResumeSavedPair()
        {
            var json = PlayerPrefs.GetString(PrefPairKey, "");
            if (string.IsNullOrEmpty(json))
            {
                return false;
            }

            PostThinkApiClient.PairResponse saved = null;
            try
            {
                saved = JsonUtility.FromJson<PostThinkApiClient.PairResponse>(json);
            }
            catch (System.Exception ex)
            {
                Debug.LogWarning($"PairFlowController: discarding unreadable saved pair. {ex.Message}");
            }

            if (saved == null || saved.parts == null || saved.parts.Length == 0)
            {
                return false;
            }

            _pair = saved;
            var part = Mathf.Clamp(PlayerPrefs.GetInt(PrefPartKey, 1), 1, _pair.parts.Length);
            Debug.Log($"PairFlowController: resuming participant {saved.participant_id} at part {part}.");
            ConfigurePart(part);
            return true;
        }

        private PostThinkApiClient.PairPart PartByIndex(int idx)
        {
            if (_pair == null || _pair.parts == null) return null;
            foreach (var part in _pair.parts)
            {
                if (part.part_index == idx) return part;
            }
            return null;
        }

        private void ConfigurePart(int idx)
        {
            var part = PartByIndex(idx);
            if (part == null)
            {
                Debug.LogError($"PairFlowController: no part with index {idx}.");
                return;
            }

            _currentPart = idx;
            PersistProgress();
            StatusBanner.Instance?.Hide();
            scenarioController.Configure(part.scenario_id);
            sessionManager.BeginWithSession(part.session_id);

            int total = _pair?.parts?.Length ?? idx;
            ShowTransition($"Playthrough {idx} of {total}");
        }

        private void PersistProgress()
        {
            if (_pair == null)
            {
                return;
            }
            PlayerPrefs.SetString(PrefPairKey, JsonUtility.ToJson(_pair));
            PlayerPrefs.SetInt(PrefPartKey, _currentPart);
            PlayerPrefs.SetInt(PrefDoneKey, 0);
            PlayerPrefs.Save();
        }

        private void MarkStudyComplete()
        {
            StudyComplete = true;
            PlayerPrefs.SetInt(PrefDoneKey, 1);
            if (_pair != null && !string.IsNullOrEmpty(_pair.participant_id))
            {
                PlayerPrefs.SetString(PrefParticipantKey, _pair.participant_id);
            }
            PlayerPrefs.DeleteKey(PrefPairKey);
            PlayerPrefs.Save();
        }

        private void ShowEndCard()
        {
            string pid = _pair != null && !string.IsNullOrEmpty(_pair.participant_id)
                ? _pair.participant_id
                : PlayerPrefs.GetString(PrefParticipantKey, "");
            _participantId = pid;

            if (transitionText != null)
            {
                transitionText.text = string.IsNullOrEmpty(pid)
                    ? endCardMessage
                    : $"{endCardMessage}\n\nYour participant ID:\n{pid}\n\nPlease enter this ID in the survey.";
            }
            if (transitionPanel != null)
            {
                transitionPanel.SetActive(true);
            }
            if (copyIdButton != null)
            {
                copyIdButton.gameObject.SetActive(!string.IsNullOrEmpty(pid));
            }
            if (participantIdField != null)
            {
                participantIdField.readOnly = true;
                participantIdField.text = pid;
                participantIdField.gameObject.SetActive(!string.IsNullOrEmpty(pid));
            }
            Cursor.lockState = CursorLockMode.None;
            Cursor.visible = true;
        }

#if UNITY_WEBGL && !UNITY_EDITOR
        [System.Runtime.InteropServices.DllImport("__Internal")]
        private static extern void CopyToClipboard(string str);
#endif

        private void CopyParticipantId()
        {
            if (string.IsNullOrEmpty(_participantId))
            {
                return;
            }
#if UNITY_WEBGL && !UNITY_EDITOR
            CopyToClipboard(_participantId);
#else
            GUIUtility.systemCopyBuffer = _participantId;
#endif
        }

        private void ShowError(string message, System.Action onRetry = null)
        {
            if (StatusBanner.Instance != null)
            {
                StatusBanner.Instance.Show(message, onRetry);
            }
        }

        private void ShowTransition(string message)
        {
            if (transitionPanel == null)
            {
                return;
            }
            if (transitionText != null)
            {
                transitionText.text = message;
            }
            if (copyIdButton != null)
            {
                copyIdButton.gameObject.SetActive(false);
            }
            if (participantIdField != null)
            {
                participantIdField.gameObject.SetActive(false);
            }
            transitionPanel.SetActive(true);
            StartCoroutine(HideTransitionAfter(transitionSeconds));
        }

        private IEnumerator HideTransitionAfter(float seconds)
        {
            yield return new WaitForSeconds(seconds);
            if (transitionPanel != null)
            {
                transitionPanel.SetActive(false);
            }
        }

        public bool HasNextPart =>
            _pair != null && _pair.parts != null && _currentPart < _pair.parts.Length;

        public void AdvanceToNextPart()
        {
            if (_ending || sessionManager == null || !HasNextPart)
            {
                return;
            }
            _ending = true;

            StartCoroutine(sessionManager.EndSession(
                _ =>
                {
                    _ending = false;
                    ConfigurePart(_currentPart + 1);
                },
                err =>
                {
                    _ending = false;
                    Debug.LogError($"/session/end failed: {err}");
                    ShowError(
                        $"Couldn't travel to the next area.\n{err}",
                        AdvanceToNextPart);
                }));
        }

        public void EndStudyNow()
        {
            if (_ending || sessionManager == null)
            {
                return;
            }
            _ending = true;

            StartCoroutine(sessionManager.EndSession(
                _ =>
                {
                    MarkStudyComplete();
                    ShowEndCard();
                },
                err =>
                {
                    _ending = false;
                    Debug.LogError($"/session/end failed: {err}");
                    ShowError(
                        $"Couldn't finish the study.\n{err}",
                        EndStudyNow);
                }));
        }
    }
}
