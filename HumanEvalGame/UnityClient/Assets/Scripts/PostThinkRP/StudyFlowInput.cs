using TMPro;
using UnityEngine;

namespace PostThinkRP
{
    public class StudyFlowInput : MonoBehaviour
    {
        [SerializeField] private QuestController questController;

        [SerializeField] private TMP_Text hintText;

        [SerializeField] private KeyCode advanceKey = KeyCode.V;
        [SerializeField] private KeyCode finishKey = KeyCode.F;

        [SerializeField] private string advanceHint =
            "Quest complete — press V to travel onward.";
        [SerializeField] private string finishHint =
            "Quest complete — press F to finish.";

        private void Awake()
        {
            if (questController == null)
            {
                questController = FindObjectOfType<QuestController>();
            }
        }

        private void Update()
        {
            bool canAdvance = CanAdvance();
            bool canFinish = CanFinish();

            if (hintText != null)
            {
                hintText.gameObject.SetActive(canAdvance || canFinish);
                if (canAdvance)
                {
                    hintText.text = advanceHint;
                }
                else if (canFinish)
                {
                    hintText.text = finishHint;
                }
            }

            if (canAdvance && Input.GetKeyDown(advanceKey))
            {
                PairFlowController.Instance.AdvanceToNextPart();
            }
            else if (canFinish && Input.GetKeyDown(finishKey))
            {
                PairFlowController.Instance.EndStudyNow();
            }
        }

        private bool Ready =>
            !UiState.DialogueOpen
            && PairFlowController.Instance != null
            && !PairFlowController.Instance.StudyComplete
            && questController != null
            && questController.DebriefDone;

        private bool CanAdvance() => Ready && PairFlowController.Instance.HasNextPart;

        private bool CanFinish() => Ready && !PairFlowController.Instance.HasNextPart;
    }
}
