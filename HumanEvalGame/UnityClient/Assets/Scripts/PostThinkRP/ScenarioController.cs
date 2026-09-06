using UnityEngine;

namespace PostThinkRP
{
    public class ScenarioController : MonoBehaviour
    {
        [Header("Player teleport")]
        [SerializeField] private Transform player;
        [SerializeField] private CharacterController playerController;
        [SerializeField] private Vector3 tavernPlayerPos = new Vector3(0f, 1.05f, -5f);
        [SerializeField] private Vector3 villagePlayerPos = new Vector3(0f, 21.05f, -5f);

        [Header("Lighting (tavern ambient defaults to OFF per the tuned scene)")]
        [SerializeField] private Light villageSun;
        [SerializeField] private Light tavernSun;
        [SerializeField] private Color tavernAmbient = Color.black;
        [SerializeField] private Color villageAmbient = new Color(0.30f, 0.32f, 0.38f);

        [Header("Shared systems / UI")]
        [SerializeField] private QuestController questController;
        [SerializeField] private DialogueController dialogueController;

        public const string Tavern = "tavern";
        public const string Village = "village";

        [Header("Dev")]
        [Tooltip("Editor-only hotkeys: F1 jumps to the tavern, F2 to the village. " +
                 "Lets you preview either part without running the full /pair flow.")]
        [SerializeField] private bool devHotkeys = true;

        private void Awake()
        {
            Apply(Tavern);
        }

#if UNITY_EDITOR || DEVELOPMENT_BUILD
        private void Update()
        {
            if (!devHotkeys) return;
            if (Input.GetKeyDown(KeyCode.F1)) Apply(Tavern);
            else if (Input.GetKeyDown(KeyCode.F2)) Apply(Village);
        }
#endif

        public void Configure(string scenarioId)
        {
            Apply(scenarioId);
        }

        private void Apply(string scenarioId)
        {
            bool village = scenarioId == Village;

            TeleportPlayer(village ? villagePlayerPos : tavernPlayerPos);

            if (villageSun != null) villageSun.enabled = village;
            if (tavernSun != null) tavernSun.enabled = !village;

            RenderSettings.ambientMode = UnityEngine.Rendering.AmbientMode.Flat;
            RenderSettings.ambientLight = village ? villageAmbient : tavernAmbient;

            if (dialogueController != null) dialogueController.CloseDialogue();

            if (questController != null)
            {
                if (village)
                {
                    questController.ConfigureScenario(
                        new[] { "keeper", "guard" },
                        "Ask around about the bandit raids.",
                        "You've gathered enough. Head out through the village gate.",
                        "Riding out against the raiders...",
                        "The raiders have been driven off. Go talk to the villagers again.",
                        "I've come back. The raiders have been driven off.");
                }
                else
                {
                    questController.ConfigureScenario(
                        new[] { "maid", "hunter" },
                        "Ask around about the wolves.",
                        "You've gathered enough. Leave the tavern through the door.",
                        "The hunt is underway...",
                        "The wolves have been dealt with. Go talk to the villagers again.",
                        "I've come back. The wolves have been dealt with.");
                }
            }
        }

        private void TeleportPlayer(Vector3 pos)
        {
            if (player == null) return;

            bool hadController = playerController != null && playerController.enabled;
            if (hadController) playerController.enabled = false;
            player.position = pos;
            if (hadController) playerController.enabled = true;
        }
    }
}
