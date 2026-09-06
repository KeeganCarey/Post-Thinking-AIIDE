using System;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

namespace PostThinkRP
{
    public class StatusBanner : MonoBehaviour
    {
        public static StatusBanner Instance { get; private set; }

        [SerializeField] private GameObject panel;
        [SerializeField] private TMP_Text messageText;
        [SerializeField] private Button retryButton;

        private Action _onRetry;

        private void Awake()
        {
            if (Instance != null && Instance != this)
            {
                Destroy(this);
                return;
            }
            Instance = this;

            if (panel != null)
            {
                panel.SetActive(false);
            }
            if (retryButton != null)
            {
                retryButton.onClick.AddListener(OnRetryClicked);
            }
        }

        private void OnDestroy()
        {
            if (Instance == this)
            {
                Instance = null;
            }
        }

        public void Show(string message, Action onRetry = null)
        {
            _onRetry = onRetry;
            if (messageText != null)
            {
                messageText.text = message;
            }
            if (retryButton != null)
            {
                retryButton.gameObject.SetActive(onRetry != null);
            }
            if (panel != null)
            {
                panel.SetActive(true);
            }
            Cursor.lockState = CursorLockMode.None;
            Cursor.visible = true;
        }

        public void Hide()
        {
            _onRetry = null;
            if (panel != null)
            {
                panel.SetActive(false);
            }
        }

        private void OnRetryClicked()
        {
            var retry = _onRetry;
            Hide();
            retry?.Invoke();
        }
    }
}
