using System;
using System.Collections;
using UnityEngine;

namespace PostThinkRP
{
    public class SessionManager : MonoBehaviour
    {
        [SerializeField] private PostThinkApiClient apiClient;
        [SerializeField] private string startingNpc = "maid";

        [Tooltip("Legacy single-session (between-subjects) flow creates its own " +
                 "session on Start(). Set FALSE for the Option-B paired flow, " +
                 "where PairFlowController injects the session id via " +
                 "BeginWithSession().")]
        [SerializeField] private bool autoCreateSession = true;

        public string SessionId { get; private set; }
        public bool IsReady { get; private set; }

        public event Action<string> SessionStarted;
        public event Action<string> SessionError;

        private IEnumerator Start()
        {
            if (apiClient == null)
            {
                apiClient = FindObjectOfType<PostThinkApiClient>();
            }

            if (apiClient == null)
            {
                SessionError?.Invoke("No PostThinkApiClient found in scene.");
                yield break;
            }

            if (!autoCreateSession)
            {
                yield break;
            }

            yield return apiClient.Initialize();
            yield return apiClient.CreateSession(
                startingNpc,
                response =>
                {
                    SessionId = response.session_id;
                    IsReady = true;
                    SessionStarted?.Invoke(SessionId);
                },
                error => SessionError?.Invoke(error));
        }

        public void BeginWithSession(string sessionId)
        {
            SessionId = sessionId;
            IsReady = true;
            SessionStarted?.Invoke(sessionId);
        }

        public IEnumerator RequestGreeting(string npcId, Action<PostThinkApiClient.ChatResponse> onSuccess, Action<string> onError)
        {
            if (!IsReady)
            {
                onError?.Invoke("Session is not ready yet.");
                yield break;
            }

            yield return apiClient.RequestGreeting(SessionId, npcId, onSuccess, onError);
        }

        public IEnumerator GetHistory(string npcId, Action<PostThinkApiClient.HistoryResponse> onSuccess, Action<string> onError)
        {
            if (!IsReady)
            {
                onError?.Invoke("Session is not ready yet.");
                yield break;
            }

            yield return apiClient.GetHistory(SessionId, npcId, onSuccess, onError);
        }

        public IEnumerator SendChat(string npcId, string message, Action<PostThinkApiClient.ChatResponse> onSuccess, Action<string> onError)
        {
            if (!IsReady)
            {
                onError?.Invoke("Session is not ready yet.");
                yield break;
            }

            yield return apiClient.SendChat(SessionId, npcId, message, onSuccess, onError);
        }

        public IEnumerator CompleteQuest(Action<PostThinkApiClient.QuestCompleteResponse> onSuccess, Action<string> onError)
        {
            if (!IsReady)
            {
                onError?.Invoke("Session is not ready yet.");
                yield break;
            }

            yield return apiClient.CompleteQuest(SessionId, onSuccess, onError);
        }

        public IEnumerator EndSession(Action<PostThinkApiClient.EndSessionResponse> onSuccess, Action<string> onError)
        {
            if (!IsReady)
            {
                onError?.Invoke("Session is not ready yet.");
                yield break;
            }

            yield return apiClient.EndSession(SessionId, onSuccess, onError);
        }
    }
}
