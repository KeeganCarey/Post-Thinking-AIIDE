using System;
using System.Collections;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;

namespace PostThinkRP
{
    public class PostThinkApiClient : MonoBehaviour
    {
        [Header("Endpoint")]
        [SerializeField] private string fallbackApiBaseUrl = "https://YOUR-NGROK-URL.ngrok-free.app";
        [SerializeField] private bool loadStreamingAssetsConfig = true;
        [SerializeField] private string configFileName = "postthink_config.json";

        public string ApiBaseUrl { get; private set; }
        public bool IsInitialized { get; private set; }

        [Serializable]
        private class ApiConfig
        {
            public string apiBaseUrl;
        }

        [Serializable]
        public class SessionRequest
        {
            public string starting_npc = "maid";
        }

        [Serializable]
        public class SessionResponse
        {
            public string session_id;
            public string starting_npc;
        }

        [Serializable]
        public class GreetingRequest
        {
            public string session_id;
            public string npc_id;
        }

        [Serializable]
        public class HistoryRequest
        {
            public string session_id;
            public string npc_id;
        }

        [Serializable]
        public class HistoryTurn
        {
            public int turn_index;
            public string player_message;
            public string dialogue;
            public bool is_greeting;
        }

        [Serializable]
        public class HistoryResponse
        {
            public string session_id;
            public string npc_id;
            public bool quest_completed;
            public HistoryTurn[] turns;
        }

        [Serializable]
        public class ChatRequest
        {
            public string session_id;
            public string npc_id;
            public string message;
        }

        [Serializable]
        public class ChatResponse
        {
            public string dialogue;
            public string npc_id;
            public int turn_index;
        }

        [Serializable]
        public class QuestCompleteRequest
        {
            public string session_id;
            public bool quest_completed = true;
        }

        [Serializable]
        public class QuestCompleteResponse
        {
            public string session_id;
            public bool quest_completed;
        }

        [Serializable]
        public class EndSessionRequest
        {
            public string session_id;
        }

        [Serializable]
        public class EndSessionResponse
        {
            public string session_id;
            public string survey_url;
        }

        [Serializable]
        public class PairPart
        {
            public int part_index;
            public string session_id;
            public string scenario_id;
            public string starting_npc;
        }

        [Serializable]
        public class PairResponse
        {
            public string participant_id;
            public PairPart[] parts;
        }

        [Serializable]
        private class EmptyBody
        {
        }

        public IEnumerator Initialize()
        {
            if (IsInitialized)
            {
                yield break;
            }

            ApiBaseUrl = fallbackApiBaseUrl;
            if (loadStreamingAssetsConfig)
            {
                yield return LoadConfigFromStreamingAssets();
            }

            ApiBaseUrl = ApiBaseUrl.TrimEnd('/');
            IsInitialized = true;
        }

        private IEnumerator LoadConfigFromStreamingAssets()
        {
            var path = $"{Application.streamingAssetsPath.TrimEnd('/')}/{configFileName}";
            string text;

            if (path.Contains("://"))
            {
                using var request = UnityWebRequest.Get(path);
                request.SetRequestHeader("ngrok-skip-browser-warning", "true");
                yield return request.SendWebRequest();

                if (RequestFailed(request))
                {
                    Debug.LogWarning($"Could not load {configFileName}; using inspector endpoint. {request.error}");
                    yield break;
                }

                text = request.downloadHandler.text;
            }
            else
            {
                if (!System.IO.File.Exists(path))
                {
                    Debug.LogWarning($"Could not find {configFileName} at {path}; using inspector endpoint.");
                    yield break;
                }

                text = System.IO.File.ReadAllText(path);
            }

            var config = JsonUtility.FromJson<ApiConfig>(text);
            if (config != null && !string.IsNullOrWhiteSpace(config.apiBaseUrl))
            {
                ApiBaseUrl = config.apiBaseUrl;
            }
        }

        public IEnumerator CreateSession(string startingNpc, Action<SessionResponse> onSuccess, Action<string> onError)
        {
            var req = new SessionRequest { starting_npc = startingNpc };
            yield return PostJson("/session", req, onSuccess, onError);
        }

        public IEnumerator CreatePair(Action<PairResponse> onSuccess, Action<string> onError)
        {
            yield return PostJson("/pair", new EmptyBody(), onSuccess, onError);
        }

        public IEnumerator RequestGreeting(string sessionId, string npcId, Action<ChatResponse> onSuccess, Action<string> onError)
        {
            var req = new GreetingRequest
            {
                session_id = sessionId,
                npc_id = npcId
            };
            yield return PostJson("/greeting", req, onSuccess, onError);
        }

        public IEnumerator GetHistory(string sessionId, string npcId, Action<HistoryResponse> onSuccess, Action<string> onError)
        {
            var req = new HistoryRequest
            {
                session_id = sessionId,
                npc_id = npcId
            };
            yield return PostJson("/history", req, onSuccess, onError);
        }

        public IEnumerator SendChat(string sessionId, string npcId, string message, Action<ChatResponse> onSuccess, Action<string> onError)
        {
            var req = new ChatRequest
            {
                session_id = sessionId,
                npc_id = npcId,
                message = message
            };
            yield return PostJson("/chat", req, onSuccess, onError);
        }

        public IEnumerator CompleteQuest(string sessionId, Action<QuestCompleteResponse> onSuccess, Action<string> onError)
        {
            var req = new QuestCompleteRequest
            {
                session_id = sessionId,
                quest_completed = true
            };
            yield return PostJson("/quest/complete", req, onSuccess, onError);
        }

        public IEnumerator EndSession(string sessionId, Action<EndSessionResponse> onSuccess, Action<string> onError)
        {
            var req = new EndSessionRequest { session_id = sessionId };
            yield return PostJson("/session/end", req, onSuccess, onError);
        }

        private IEnumerator PostJson<TRequest, TResponse>(
            string path,
            TRequest requestBody,
            Action<TResponse> onSuccess,
            Action<string> onError)
        {
            if (!IsInitialized)
            {
                yield return Initialize();
            }

            var url = ApiBaseUrl + path;
            var json = JsonUtility.ToJson(requestBody);
            var bytes = Encoding.UTF8.GetBytes(json);

            using var request = new UnityWebRequest(url, "POST");
            request.uploadHandler = new UploadHandlerRaw(bytes);
            request.downloadHandler = new DownloadHandlerBuffer();
            request.SetRequestHeader("Content-Type", "application/json");
            request.SetRequestHeader("ngrok-skip-browser-warning", "true");

            yield return request.SendWebRequest();

            if (RequestFailed(request))
            {
                onError?.Invoke($"{request.responseCode}: {request.error} {request.downloadHandler.text}");
                yield break;
            }

            try
            {
                var response = JsonUtility.FromJson<TResponse>(request.downloadHandler.text);
                onSuccess?.Invoke(response);
            }
            catch (Exception ex)
            {
                onError?.Invoke($"Failed to parse response: {ex.Message}");
            }
        }

        private static bool RequestFailed(UnityWebRequest request)
        {
#if UNITY_2020_2_OR_NEWER
            return request.result != UnityWebRequest.Result.Success;
#else
            return request.isNetworkError || request.isHttpError;
#endif
        }
    }
}
