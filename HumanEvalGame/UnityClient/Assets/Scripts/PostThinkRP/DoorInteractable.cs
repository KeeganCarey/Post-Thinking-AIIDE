using UnityEngine;

namespace PostThinkRP
{
    [RequireComponent(typeof(Collider))]
    public class DoorInteractable : MonoBehaviour, IInteractable
    {
        [SerializeField] private QuestController questController;

        [SerializeField] private string actionLabel = "leave the tavern";

        private void Awake()
        {
            if (questController == null)
            {
                questController = FindFirstObjectByType<QuestController>();
            }
        }

        public bool CanInteract => questController != null && questController.ReadyToLeave;

        public string GetPrompt(KeyCode key) => $"Press {key} to {actionLabel}";

        public void Interact()
        {
            if (questController != null)
            {
                questController.RequestCompletion();
            }
        }
    }
}
