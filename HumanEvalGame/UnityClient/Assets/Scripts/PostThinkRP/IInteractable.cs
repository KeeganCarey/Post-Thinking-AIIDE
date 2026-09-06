using UnityEngine;

namespace PostThinkRP
{
    public interface IInteractable
    {
        bool CanInteract { get; }

        string GetPrompt(KeyCode key);

        void Interact();
    }
}
