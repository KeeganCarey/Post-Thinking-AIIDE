#if UNITY_EDITOR
using PostThinkRP;
using TMPro;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

public static class PostThinkSceneGenerator
{
    private const string VillageRootName = "VillageLevel";
    private const float YOffset = 20f;

    private struct NpcDef
    {
        public string objectName;
        public string npcId;
        public string displayName;
        public Vector3 position;
        public Color color;
    }

    private static readonly NpcDef[] VillageNpcs =
    {
        new NpcDef { objectName = "Keeper_Odila", npcId = "keeper", displayName = "Odila", position = new Vector3(-3.0f, 1f, 3.6f), color = new Color(0.85f, 0.78f, 0.55f) },
        new NpcDef { objectName = "Guard_Tomas", npcId = "guard", displayName = "Tomas", position = new Vector3(4.2f, 1f, -2.6f), color = new Color(0.46f, 0.47f, 0.54f) },
        new NpcDef { objectName = "Trader_Gunnar", npcId = "trader", displayName = "Gunnar", position = new Vector3(4.4f, 1f, 3.4f), color = new Color(0.80f, 0.42f, 0.50f) },
    };

    [MenuItem("PostThink-RP/Add Village Level (Scene 2) to Open Scene")]
    public static void AddVillageLevel()
    {
        var dialogue = Object.FindObjectOfType<DialogueController>();
        if (dialogue == null)
        {
            Debug.LogError("[VillageAdd] No DialogueController in the open scene. Open your tuned tavern scene first.");
            return;
        }

        var prev = GameObject.Find(VillageRootName);
        if (prev != null) Object.DestroyImmediate(prev);

        var villageRoot = new GameObject(VillageRootName).transform;
        var env = new GameObject("VillageEnvironment").transform;
        env.SetParent(villageRoot);
        BuildVillageEnvironment(env, out var villageSun);

        var npcRoot = new GameObject("VillageNPCs").transform;
        npcRoot.SetParent(villageRoot);
        foreach (var n in VillageNpcs)
        {
            CreateNpc(npcRoot, n.objectName, n.npcId, n.displayName, n.position,
                Material($"PT_{n.npcId}", n.color), dialogue);
        }
        villageRoot.position = new Vector3(0f, YOffset, 0f);

        var session = Object.FindObjectOfType<SessionManager>();
        var quest = Object.FindObjectOfType<QuestController>();
        var api = Object.FindObjectOfType<PostThinkApiClient>();
        var systems = GameObject.Find("Systems") ?? (session != null ? session.gameObject : null);
        if (systems == null || session == null || quest == null || api == null)
        {
            Debug.LogError("[VillageAdd] Could not find Systems/SessionManager/QuestController/PostThinkApiClient. " +
                           "Village geometry was built; wire the components manually.");
            EditorSceneManager.MarkSceneDirty(SceneManager.GetActiveScene());
            return;
        }

        var scenario = systems.GetComponent<ScenarioController>() ?? systems.AddComponent<ScenarioController>();
        var pairFlow = systems.GetComponent<PairFlowController>() ?? systems.AddComponent<PairFlowController>();

        var player = GameObject.Find("Player");
        Vector3 tavernPos = player != null ? player.transform.position : new Vector3(0, 1.05f, -5f);

        if (player != null)
        {
            WireSerializedObject(scenario, "player", player.transform);
            var cc = player.GetComponent<CharacterController>();
            if (cc != null) WireSerializedObject(scenario, "playerController", cc);
        }
        WireSerializedVector3(scenario, "tavernPlayerPos", tavernPos);
        WireSerializedVector3(scenario, "villagePlayerPos", tavernPos + new Vector3(0, YOffset, 0));
        WireSerializedObject(scenario, "villageSun", villageSun);
        WireSerializedObject(scenario, "questController", quest);
        WireSerializedObject(scenario, "dialogueController", dialogue);

        var tavernSunGo = GameObject.Find("Directional Light");
        var tavernSun = tavernSunGo != null ? tavernSunGo.GetComponent<Light>() : null;
        if (tavernSun != null) WireSerializedObject(scenario, "tavernSun", tavernSun);
        else Debug.LogWarning("[VillageAdd] TODO: no 'Directional Light' found to wire as tavernSun (optional).");

        WireSerializedObject(pairFlow, "apiClient", api);
        WireSerializedObject(pairFlow, "sessionManager", session);
        WireSerializedObject(pairFlow, "scenarioController", scenario);

        WireSerializedBool(session, "autoCreateSession", false);

        WireStudyFlowInput(systems, quest);

        Selection.activeGameObject = systems;
        EditorSceneManager.MarkSceneDirty(SceneManager.GetActiveScene());
        Debug.Log($"[VillageAdd] Village level built at +{YOffset}Y and switching components wired. " +
                  "Review the Console for any TODO warnings, verify the wired fields on Systems' " +
                  "ScenarioController/PairFlowController, then Ctrl+S.");
    }

    [MenuItem("PostThink-RP/Wire StudyFlow Hint + Copy ID button (additive)")]
    public static void WireStudyFlowMenu()
    {
        var session = Object.FindObjectOfType<SessionManager>();
        var quest = Object.FindObjectOfType<QuestController>();
        var systems = GameObject.Find("Systems") ?? (session != null ? session.gameObject : null);
        if (systems == null || quest == null)
        {
            Debug.LogError("[StudyFlow] Could not find Systems / QuestController in the open scene.");
            return;
        }

        WireStudyFlowInput(systems, quest);
        EditorSceneManager.MarkSceneDirty(SceneManager.GetActiveScene());
        Debug.Log("[StudyFlow] Done. Review the new 'StudyFlow Hint' / 'Copy ID Button' objects, then Ctrl+S.");
    }

    private static void WireStudyFlowInput(GameObject systems, QuestController quest)
    {
        var input = systems.GetComponent<StudyFlowInput>() ?? systems.AddComponent<StudyFlowInput>();
        if (GetSerializedRef(input, "questController") == null)
        {
            WireSerializedObject(input, "questController", quest);
        }

        var canvas = Object.FindObjectOfType<Canvas>();

        if (GetSerializedRef(input, "hintText") == null)
        {
            if (canvas != null)
            {
                var hint = CreateText(canvas.transform, "StudyFlow Hint", string.Empty, 26,
                    FontStyles.Bold, TextAlignmentOptions.Center, Color.white);
                Rect(hint.rectTransform, new Vector2(0.15f, 0.06f), new Vector2(0.85f, 0.16f),
                    Vector2.zero, Vector2.zero);
                WireSerializedObject(input, "hintText", hint);
            }
            else
            {
                Debug.LogWarning("[VillageAdd] TODO: no Canvas found; wire StudyFlowInput.hintText manually.");
            }
        }

        var pairFlow = systems.GetComponent<PairFlowController>();
        if (pairFlow != null && GetSerializedRef(pairFlow, "copyIdButton") == null)
        {
            var panel = GetSerializedRef(pairFlow, "transitionPanel") as GameObject;
            var parent = panel != null ? panel.transform : (canvas != null ? canvas.transform : null);
            if (parent != null)
            {
                var copyBtn = CreateButton(parent, "Copy ID Button", "Copy ID");
                Rect((RectTransform)copyBtn.transform, new Vector2(0.40f, 0.28f), new Vector2(0.60f, 0.37f),
                    Vector2.zero, Vector2.zero);
                WireSerializedObject(pairFlow, "copyIdButton", copyBtn);
            }
            else
            {
                Debug.LogWarning("[VillageAdd] TODO: no transitionPanel/Canvas found; wire PairFlowController.copyIdButton manually.");
            }
        }

        Debug.Log("[VillageAdd] StudyFlowInput hint + Copy ID button created/wired (only where missing). " +
                  "Nothing existing was modified. Reminder: set StudyFlowInput.finishKey=F and update " +
                  "finishHint on a pre-existing component (the generator does not touch those).");
    }

    private static Object GetSerializedRef(Object target, string propertyName)
    {
        var so = new SerializedObject(target);
        var prop = so.FindProperty(propertyName);
        return prop != null ? prop.objectReferenceValue : null;
    }

    private static Button FindButton(string name)
    {
        var go = GameObject.Find(name);
        return go != null ? go.GetComponent<Button>() : null;
    }

    private static void BuildVillageEnvironment(Transform environment, out Light villageSun)
    {
        var ground = Material("PT_Ground", new Color(0.34f, 0.40f, 0.22f));
        var cobble = Material("PT_Cobble", new Color(0.46f, 0.44f, 0.40f));
        var stone = Material("PT_Stone", new Color(0.55f, 0.55f, 0.58f));
        var wood = Material("PT_VWood", new Color(0.34f, 0.22f, 0.12f));
        var plaster = Material("PT_Plaster", new Color(0.78f, 0.72f, 0.60f));
        var water = Material("PT_Water", new Color(0.18f, 0.34f, 0.46f));
        var roof = Material("PT_Roof", new Color(0.45f, 0.20f, 0.16f));

        CreateCube(environment, "Ground", new Vector3(0, -0.05f, 0), new Vector3(16, 0.1f, 16), ground);
        CreateCube(environment, "Market Path", new Vector3(0, 0.01f, 0), new Vector3(5.0f, 0.02f, 12f), cobble);
        CreateCube(environment, "Market Path Cross", new Vector3(0, 0.01f, 0), new Vector3(12f, 0.02f, 5.0f), cobble);

        CreateCube(environment, "Well Base", new Vector3(0, 0.45f, 0), new Vector3(1.6f, 0.9f, 1.6f), stone);
        CreateCube(environment, "Well Water", new Vector3(0, 0.86f, 0), new Vector3(1.0f, 0.05f, 1.0f), water);
        CreateCube(environment, "Well Post L", new Vector3(-0.6f, 1.5f, 0), new Vector3(0.12f, 1.4f, 0.12f), wood);
        CreateCube(environment, "Well Post R", new Vector3(0.6f, 1.5f, 0), new Vector3(0.12f, 1.4f, 0.12f), wood);
        CreateCube(environment, "Well Roof", new Vector3(0, 2.25f, 0), new Vector3(1.6f, 0.16f, 1.6f), roof);

        CreateBuilding(environment, "House Back 1", new Vector3(-4.5f, 0, 6.6f), new Vector3(4.2f, 3.4f, 1.4f), plaster, roof);
        CreateBuilding(environment, "House Back 2", new Vector3(0.5f, 0, 6.9f), new Vector3(4.6f, 3.9f, 1.4f), plaster, roof);
        CreateBuilding(environment, "House Back 3", new Vector3(5.4f, 0, 6.6f), new Vector3(3.6f, 3.0f, 1.4f), plaster, roof);
        CreateBuilding(environment, "House Left", new Vector3(-7.0f, 0, 0.5f), new Vector3(1.4f, 3.6f, 5.0f), plaster, roof);
        CreateBuilding(environment, "House Right", new Vector3(7.0f, 0, -1.0f), new Vector3(1.4f, 3.3f, 5.0f), plaster, roof);

        CreateStall(environment, "Stall A", new Vector3(4.4f, 0, 2.0f), wood, roof);
        CreateStall(environment, "Stall B", new Vector3(-3.2f, 0, 1.8f), wood, roof);
        CreateCube(environment, "Crate 1", new Vector3(3.4f, 0.35f, 2.2f), new Vector3(0.7f, 0.7f, 0.7f), wood);
        CreateCube(environment, "Crate 2", new Vector3(3.7f, 0.35f, 1.4f), new Vector3(0.7f, 0.7f, 0.7f), wood);
        CreateCube(environment, "Crate 3", new Vector3(-2.4f, 0.35f, 2.4f), new Vector3(0.7f, 0.7f, 0.7f), wood);

        CreateCube(environment, "Fence Back", new Vector3(0, 0.5f, 7.7f), new Vector3(16f, 1.0f, 0.15f), wood);
        CreateCube(environment, "Fence Left", new Vector3(-7.7f, 0.5f, 0), new Vector3(0.15f, 1.0f, 16f), wood);
        CreateCube(environment, "Fence Right", new Vector3(7.7f, 0.5f, 0), new Vector3(0.15f, 1.0f, 16f), wood);
        CreateCube(environment, "Fence Front L", new Vector3(-5.0f, 0.5f, -7.7f), new Vector3(5.5f, 1.0f, 0.15f), wood);
        CreateCube(environment, "Fence Front R", new Vector3(5.0f, 0.5f, -7.7f), new Vector3(5.5f, 1.0f, 0.15f), wood);

        CreateGate(environment, "Village Gate", new Vector3(0f, 1.1f, -7.6f), new Vector3(1.8f, 2.2f, 0.2f),
            wood, "head out to the raiders");

        villageSun = CreateDirectionalLight(environment, new Color(1f, 0.97f, 0.88f), 0.55f, new Vector3(52, 28, 0));
    }

    private static void CreateBuilding(Transform parent, string name, Vector3 basePos, Vector3 size, Material wallMat, Material roofMat)
    {
        var root = new GameObject(name).transform;
        root.SetParent(parent);
        CreateCube(root, $"{name} Body", basePos + new Vector3(0, size.y / 2f, 0), size, wallMat);
        CreateCube(root, $"{name} Roof", basePos + new Vector3(0, size.y + 0.25f, 0), new Vector3(size.x + 0.4f, 0.5f, size.z + 0.4f), roofMat);
    }

    private static void CreateStall(Transform parent, string name, Vector3 basePos, Material woodMat, Material roofMat)
    {
        var root = new GameObject(name).transform;
        root.SetParent(parent);
        CreateCube(root, $"{name} Counter", basePos + new Vector3(0, 0.55f, 0), new Vector3(1.8f, 0.16f, 0.9f), woodMat);
        CreateCube(root, $"{name} Leg A", basePos + new Vector3(-0.8f, 0.27f, -0.35f), new Vector3(0.12f, 0.55f, 0.12f), woodMat);
        CreateCube(root, $"{name} Leg B", basePos + new Vector3(0.8f, 0.27f, -0.35f), new Vector3(0.12f, 0.55f, 0.12f), woodMat);
        CreateCube(root, $"{name} Post L", basePos + new Vector3(-0.85f, 1.0f, 0.35f), new Vector3(0.1f, 1.6f, 0.1f), woodMat);
        CreateCube(root, $"{name} Post R", basePos + new Vector3(0.85f, 1.0f, 0.35f), new Vector3(0.1f, 1.6f, 0.1f), woodMat);
        CreateCube(root, $"{name} Awning", basePos + new Vector3(0, 1.75f, 0.0f), new Vector3(2.1f, 0.1f, 1.2f), roofMat);
    }

    private static GameObject CreateCube(Transform parent, string name, Vector3 position, Vector3 scale, Material material)
    {
        var cube = GameObject.CreatePrimitive(PrimitiveType.Cube);
        cube.name = name;
        cube.transform.SetParent(parent);
        cube.transform.position = position;
        cube.transform.localScale = scale;
        cube.GetComponent<Renderer>().sharedMaterial = material;
        return cube;
    }

    private static void CreateGate(Transform parent, string name, Vector3 position, Vector3 scale,
        Material material, string actionLabel)
    {
        var gate = CreateCube(parent, name, position, scale, material);
        var door = gate.AddComponent<DoorInteractable>();
        WireSerializedString(door, "actionLabel", actionLabel);
    }

    private static void CreateNpc(Transform parent, string objectName, string npcId, string displayName,
        Vector3 position, Material material, DialogueController dialogue)
    {
        var npc = GameObject.CreatePrimitive(PrimitiveType.Capsule);
        npc.name = objectName;
        npc.transform.SetParent(parent);
        npc.transform.position = position;
        npc.transform.localScale = new Vector3(0.8f, 1f, 0.8f);
        npc.GetComponent<Renderer>().sharedMaterial = material;

        var interactable = npc.AddComponent<NpcInteractable>();
        WireSerializedString(interactable, "npcId", npcId);
        WireSerializedString(interactable, "displayName", displayName);
        WireSerializedObject(interactable, "dialogueController", dialogue);

        var labelGo = new GameObject("Name Label", typeof(TextMeshPro));
        labelGo.transform.SetParent(npc.transform, false);
        labelGo.transform.localPosition = new Vector3(0, 1.35f, 0);
        labelGo.transform.localRotation = Quaternion.Euler(0, 180, 0);
        var label = labelGo.GetComponent<TextMeshPro>();
        label.text = displayName;
        label.fontSize = 3f;
        label.alignment = TextAlignmentOptions.Center;
        label.color = Color.white;
    }

    private static Light CreateDirectionalLight(Transform parent, Color color, float intensity, Vector3 euler)
    {
        var go = new GameObject("Village Sun", typeof(Light));
        go.transform.SetParent(parent);
        go.transform.rotation = Quaternion.Euler(euler.x, euler.y, euler.z);
        var light = go.GetComponent<Light>();
        light.type = LightType.Directional;
        light.intensity = intensity;
        light.color = color;
        return light;
    }

    private static Transform CreatePanel(Transform parent, string name, Color color)
    {
        var panel = new GameObject(name, typeof(RectTransform), typeof(Image));
        panel.transform.SetParent(parent, false);
        panel.GetComponent<Image>().color = color;
        return panel.transform;
    }

    private static TMP_Text CreateText(Transform parent, string name, string content, int size,
        FontStyles style, TextAlignmentOptions alignment, Color color)
    {
        var go = new GameObject(name, typeof(RectTransform), typeof(TextMeshProUGUI));
        go.transform.SetParent(parent, false);
        var text = go.GetComponent<TextMeshProUGUI>();
        text.text = content;
        text.fontSize = size;
        text.fontStyle = style;
        text.alignment = alignment;
        text.color = color;
        text.enableWordWrapping = true;
        return text;
    }

    private static Button CreateButton(Transform parent, string name, string label)
    {
        var go = new GameObject(name, typeof(RectTransform), typeof(Image), typeof(Button));
        go.transform.SetParent(parent, false);
        go.GetComponent<Image>().color = new Color(0.22f, 0.16f, 0.10f, 0.95f);
        var button = go.GetComponent<Button>();
        var text = CreateText(go.transform, "Label", label, 22, FontStyles.Bold, TextAlignmentOptions.Center, Color.white);
        Stretch(text.transform, Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero);
        return button;
    }

    private static Material Material(string name, Color color)
    {
        var shader = Shader.Find("Standard");
        return new Material(shader) { name = name, color = color };
    }

    private static void Rect(RectTransform rt, Vector2 anchorMin, Vector2 anchorMax, Vector2 offsetMin, Vector2 offsetMax)
    {
        rt.anchorMin = anchorMin;
        rt.anchorMax = anchorMax;
        rt.offsetMin = offsetMin;
        rt.offsetMax = offsetMax;
    }

    private static void Stretch(Transform transform, Vector2 anchorMin, Vector2 anchorMax, Vector2 offsetMin, Vector2 offsetMax)
    {
        Rect((RectTransform)transform, anchorMin, anchorMax, offsetMin, offsetMax);
    }

    private static void WireSerializedObject(Object target, string propertyName, Object value)
    {
        var so = new SerializedObject(target);
        var prop = so.FindProperty(propertyName);
        if (prop != null) { prop.objectReferenceValue = value; so.ApplyModifiedProperties(); }
        else Debug.LogWarning($"[VillageAdd] Could not find serialized property {propertyName} on {target.name}");
    }

    private static void WireSerializedString(Object target, string propertyName, string value)
    {
        var so = new SerializedObject(target);
        var prop = so.FindProperty(propertyName);
        if (prop != null) { prop.stringValue = value; so.ApplyModifiedProperties(); }
    }

    private static void WireSerializedBool(Object target, string propertyName, bool value)
    {
        var so = new SerializedObject(target);
        var prop = so.FindProperty(propertyName);
        if (prop != null) { prop.boolValue = value; so.ApplyModifiedProperties(); }
        else Debug.LogWarning($"[VillageAdd] Could not find serialized property {propertyName} on {target.name}");
    }

    private static void WireSerializedVector3(Object target, string propertyName, Vector3 value)
    {
        var so = new SerializedObject(target);
        var prop = so.FindProperty(propertyName);
        if (prop != null) { prop.vector3Value = value; so.ApplyModifiedProperties(); }
        else Debug.LogWarning($"[VillageAdd] Could not find serialized property {propertyName} on {target.name}");
    }

    private const string FloorAssetPath = "Assets/Models/FBX/Floor_WoodDark.fbx";
    private const string WallAssetPath = "Assets/Models/FBX/Wall_Plaster_Straight.fbx";
    private const string WoodMaterialPath = "Assets/Models/Materials/Mat_Tavern_Wood.mat";
    private const string PlasterMaterialPath = "Assets/Models/Materials/Mat_Tavern_Plaster.mat";
    private const string VillageFloorRoot = "Village Floor";
    private const string VillageWallsRoot = "Village Walls";

    private const float SquareHalf = 7.7f;
    private const float GateGapHalf = 1.6f;
    private const float WallFaceYaw = 0f;

    private static readonly Quaternion FloorBaseRot = Quaternion.Euler(90f, 0f, 0f);
    private static readonly Quaternion WallBaseRot = new Quaternion(0.5f, 0.5f, 0.5f, -0.5f);

    [MenuItem("PostThink-RP/Apply Village Floor + Walls")]
    public static void ApplyVillageFloorAndWalls()
    {
        var envGo = GameObject.Find("VillageEnvironment");
        if (envGo == null)
        {
            Debug.LogError("[VillageDress] No 'VillageEnvironment' in the open scene. Run " +
                           "'Add Village Level' first, then this.");
            return;
        }
        var env = envGo.transform;

        RemoveChild(env, VillageFloorRoot);
        RemoveChild(env, VillageWallsRoot);
        RemoveChild(env, "Ground");
        RemoveChild(env, "Market Path");
        RemoveChild(env, "Market Path Cross");
        foreach (var fence in ChildrenStartingWith(env, "Fence"))
        {
            Object.DestroyImmediate(fence.gameObject);
        }

        var floorPrefab = AssetDatabase.LoadAssetAtPath<GameObject>(FloorAssetPath);
        var wallPrefab = AssetDatabase.LoadAssetAtPath<GameObject>(WallAssetPath);
        if (floorPrefab == null || wallPrefab == null)
        {
            Debug.LogError($"[VillageDress] Could not load assets at '{FloorAssetPath}' / '{WallAssetPath}'.");
            return;
        }

        var woodMat = AssetDatabase.LoadAssetAtPath<Material>(WoodMaterialPath);
        var plasterMat = AssetDatabase.LoadAssetAtPath<Material>(PlasterMaterialPath);
        if (woodMat == null || plasterMat == null)
        {
            Debug.LogError($"[VillageDress] Could not load materials at '{WoodMaterialPath}' / '{PlasterMaterialPath}'.");
            return;
        }

        LayFloor(env, floorPrefab, woodMat);
        BuildPerimeterWalls(env, wallPrefab, plasterMat);

        EditorSceneManager.MarkSceneDirty(SceneManager.GetActiveScene());
        Selection.activeGameObject = envGo;
        Debug.Log("[VillageDress] Village floor + walls applied. ACCEPT TEST: floor flush with the " +
                  "NPCs' feet, walls upright and facing into the square, no gaps or z-fighting. If the " +
                  "walls face outward, set WallFaceYaw = 180f and re-run.");
    }

    private static void LayFloor(Transform env, GameObject prefab, Material woodMat)
    {
        var root = new GameObject(VillageFloorRoot).transform;
        root.SetParent(env, false);

        var probe = ProbeBounds(prefab, FloorBaseRot, out var size, out var center, out var minY);
        Object.DestroyImmediate(probe);
        if (size.x < 1e-3f || size.z < 1e-3f)
        {
            Debug.LogError("[VillageDress] Floor asset reported no renderer bounds.");
            return;
        }

        int nx = Mathf.CeilToInt((SquareHalf * 2f) / size.x);
        int nz = Mathf.CeilToInt((SquareHalf * 2f) / size.z);
        float startX = -((nx - 1) * size.x) / 2f;
        float startZ = -((nz - 1) * size.z) / 2f;
        float topY = minY + size.y;

        for (int ix = 0; ix < nx; ix++)
        {
            for (int iz = 0; iz < nz; iz++)
            {
                float cx = startX + ix * size.x;
                float cz = startZ + iz * size.z;
                var tile = (GameObject)PrefabUtility.InstantiatePrefab(prefab, root);
                tile.name = $"Floor {ix}_{iz}";
                tile.transform.localRotation = FloorBaseRot;
                tile.transform.localPosition = new Vector3(cx - center.x, -topY, cz - center.z);
                ApplyMaterial(tile, woodMat);
            }
        }

        var floorCollider = root.gameObject.AddComponent<BoxCollider>();
        floorCollider.center = new Vector3(0f, -0.1f, 0f);
        floorCollider.size = new Vector3(SquareHalf * 2f, 0.2f, SquareHalf * 2f);
    }

    private static void BuildPerimeterWalls(Transform env, GameObject prefab, Material plasterMat)
    {
        var root = new GameObject(VillageWallsRoot).transform;
        root.SetParent(env, false);

        var probe = ProbeBounds(prefab, WallBaseRot, out var size, out var center, out var minY);
        Object.DestroyImmediate(probe);
        float length = Mathf.Max(size.x, size.z);
        if (length < 1e-3f)
        {
            Debug.LogError("[VillageDress] Wall asset reported no renderer bounds.");
            return;
        }

        bool baseLongIsX = size.x >= size.z;
        float xEdgeYaw = baseLongIsX ? 0f : 90f;
        float zEdgeYaw = baseLongIsX ? 90f : 0f;

        WallEdge(root, prefab, plasterMat, center, minY, length, alongX: true, fixedCoord: +SquareHalf, yaw: xEdgeYaw, gateGap: 0f);
        WallEdge(root, prefab, plasterMat, center, minY, length, alongX: false, fixedCoord: -SquareHalf, yaw: zEdgeYaw, gateGap: 0f);
        WallEdge(root, prefab, plasterMat, center, minY, length, alongX: false, fixedCoord: +SquareHalf, yaw: zEdgeYaw + 180f, gateGap: 0f);
        WallEdge(root, prefab, plasterMat, center, minY, length, alongX: true, fixedCoord: -SquareHalf, yaw: xEdgeYaw + 180f, gateGap: GateGapHalf);
    }

    private static void WallEdge(Transform root, GameObject prefab, Material plasterMat, Vector3 center, float minY,
        float length, bool alongX, float fixedCoord, float yaw, float gateGap)
    {
        int n = Mathf.CeilToInt((SquareHalf * 2f) / length);
        float start = -((n - 1) * length) / 2f;
        var yawRot = Quaternion.AngleAxis(yaw + WallFaceYaw, Vector3.up);
        var rot = yawRot * WallBaseRot;
        Vector3 centerRot = yawRot * center;

        for (int i = 0; i < n; i++)
        {
            float a = start + i * length;
            if (gateGap > 0f && Mathf.Abs(a) < gateGap + length * 0.5f)
            {
                continue;
            }
            Vector3 cell = alongX ? new Vector3(a, 0f, fixedCoord) : new Vector3(fixedCoord, 0f, a);
            var wall = (GameObject)PrefabUtility.InstantiatePrefab(prefab, root);
            wall.name = $"Wall {(alongX ? "X" : "Z")} {i}";
            wall.transform.localRotation = rot;
            wall.transform.localPosition = new Vector3(cell.x - centerRot.x, -minY, cell.z - centerRot.z);
            ApplyMaterial(wall, plasterMat);
            AddMeshColliders(wall);
        }
    }

    private static void ApplyMaterial(GameObject go, Material material)
    {
        foreach (var renderer in go.GetComponentsInChildren<Renderer>())
        {
            var mats = renderer.sharedMaterials;
            for (int i = 0; i < mats.Length; i++) mats[i] = material;
            renderer.sharedMaterials = mats;
        }
    }

    private static void AddMeshColliders(GameObject go)
    {
        foreach (var filter in go.GetComponentsInChildren<MeshFilter>())
        {
            if (filter.sharedMesh == null) continue;
            if (filter.GetComponent<MeshCollider>() != null) continue;
            filter.gameObject.AddComponent<MeshCollider>();
        }
    }

    private static GameObject ProbeBounds(GameObject prefab, Quaternion baseRot,
        out Vector3 size, out Vector3 center, out float minY)
    {
        var probe = (GameObject)PrefabUtility.InstantiatePrefab(prefab);
        probe.transform.position = Vector3.zero;
        probe.transform.rotation = baseRot;
        var renderers = probe.GetComponentsInChildren<Renderer>();
        if (renderers.Length == 0)
        {
            size = Vector3.zero; center = Vector3.zero; minY = 0f;
            return probe;
        }
        var b = renderers[0].bounds;
        for (int i = 1; i < renderers.Length; i++)
        {
            b.Encapsulate(renderers[i].bounds);
        }
        size = b.size; center = b.center; minY = b.min.y;
        return probe;
    }

    private static void RemoveChild(Transform parent, string name)
    {
        var t = parent.Find(name);
        if (t != null) Object.DestroyImmediate(t.gameObject);
    }

    private static System.Collections.Generic.List<Transform> ChildrenStartingWith(Transform parent, string prefix)
    {
        var list = new System.Collections.Generic.List<Transform>();
        foreach (Transform child in parent)
        {
            if (child.name.StartsWith(prefix)) list.Add(child);
        }
        return list;
    }
}
#endif
