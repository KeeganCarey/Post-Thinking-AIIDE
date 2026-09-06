using UnityEngine;

namespace PostThinkRP
{
    public class NpcInteractable : MonoBehaviour, IInteractable
    {
        [SerializeField] private string npcId = "maid";
        [SerializeField] private string displayName = "Mirela";
        [SerializeField] private DialogueController dialogueController;

        public string NpcId => npcId;
        public string DisplayName => displayName;

        public bool CanInteract => true;
        public string GetPrompt(KeyCode key) => $"Press {key} to talk to {displayName}";
        public void Interact() => OpenDialogue();

        private void Awake()
        {
            if (dialogueController == null)
            {
                dialogueController = FindObjectOfType<DialogueController>();
            }
        }

        public void OpenDialogue()
        {
            if (dialogueController != null)
            {
                dialogueController.OpenNpc(this);
            }
        }
    }
}
