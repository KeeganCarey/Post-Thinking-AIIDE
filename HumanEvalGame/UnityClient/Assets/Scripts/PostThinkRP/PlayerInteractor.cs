using TMPro;
using UnityEngine;

namespace PostThinkRP
{
    public class PlayerInteractor : MonoBehaviour
    {
        [SerializeField] private Camera playerCamera;
        [SerializeField] private float maxDistance = 3f;
        [SerializeField] private LayerMask interactionMask = -1;
        [SerializeField] private TMP_Text promptText;
        [SerializeField] private KeyCode interactKey = KeyCode.E;

        private IInteractable focused;

        private void Awake()
        {
            if (playerCamera == null)
            {
                playerCamera = Camera.main;
            }
        }

        private void Update()
        {
            if (UiState.DialogueOpen)
            {
                if (promptText != null)
                {
                    promptText.gameObject.SetActive(false);
                }
                focused = null;
                return;
            }

            focused = FindFocused();
            bool show = focused != null && focused.CanInteract;
            if (promptText != null)
            {
                promptText.gameObject.SetActive(show);
                promptText.text = show ? focused.GetPrompt(interactKey) : string.Empty;
            }

            if (show && Input.GetKeyDown(interactKey))
            {
                focused.Interact();
            }
        }

        private IInteractable FindFocused()
        {
            if (playerCamera == null)
            {
                return null;
            }

            var ray = new Ray(playerCamera.transform.position, playerCamera.transform.forward);
            if (Physics.Raycast(ray, out var hit, maxDistance, interactionMask))
            {
                return hit.collider.GetComponentInParent<IInteractable>();
            }
            return null;
        }
    }
}

