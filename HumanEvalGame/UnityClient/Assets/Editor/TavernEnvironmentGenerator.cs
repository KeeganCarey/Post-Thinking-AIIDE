#if UNITY_EDITOR
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

public static class TavernEnvironmentGenerator
{
    private const string FloorFbx     = "Assets/Models/FBX/Floor_WoodDark.fbx";
    private const string WallFbx      = "Assets/Models/FBX/Wall_Plaster_Straight.fbx";
    private const string WallDoorFbx  = "Assets/Models/FBX/Wall_Plaster_Door_Flat.fbx";
    private const string DoorFrameFbx = "Assets/Models/FBX/DoorFrame_Flat_WoodDark.fbx";
    private const string DoorFbx      = "Assets/Models/FBX/Door_1_Flat.fbx";

    private const string WoodAlbedo   = "Assets/Decor/glTF/T_WoodTrim_BaseColor.png";
    private const string WoodNormal   = "Assets/Decor/glTF/T_WoodTrim_Normal.png";
    private const string PlasterAlbedo = "Assets/Decor/glTF/T_Plaster_BaseColor.png";
    private const string PlasterNormal = "Assets/Decor/glTF/T_Plaster_Normal.png";

    private const string MaterialDir  = "Assets/Models/Materials";

    private const float HalfExtent   = 7f;
    private const float CeilingHeight = 3.15f;

    private const string RootName = "TavernEnvironment";

    [MenuItem("PostThink-RP/Generate Tavern Environment")]
    public static void Generate()
    {
        var floorAsset = Load(FloorFbx);
        var wallAsset = Load(WallFbx);
        var wallDoorAsset = Load(WallDoorFbx);
        var frameAsset = Load(DoorFrameFbx);
        var doorAsset = Load(DoorFbx);
        if (floorAsset == null || wallAsset == null || wallDoorAsset == null ||
            frameAsset == null || doorAsset == null)
        {
            Debug.LogError("[TavernGen] One or more kit FBX assets are missing. Aborting.");
            return;
        }

        var woodMat = BuildMaterial("Mat_Tavern_Wood", WoodAlbedo, WoodNormal);
        var plasterMat = BuildMaterial("Mat_Tavern_Plaster", PlasterAlbedo, PlasterNormal);

        var existing = GameObject.Find(RootName);
        if (existing != null)
            Object.DestroyImmediate(existing);

        var root = new GameObject(RootName).transform;

        float floorModule = MeasureXZ(floorAsset);
        if (floorModule <= 0.001f)
        {
            Debug.LogError("[TavernGen] Could not measure floor tile size. Aborting.");
            Object.DestroyImmediate(root.gameObject);
            return;
        }

        Vector3 wallSize = MeasureSize(wallAsset);
        float wallWidth = Mathf.Max(0.001f, wallSize.x);
        float wallThick = Mathf.Max(0.05f, wallSize.z);

        int floorTiles = Mathf.Max(1, Mathf.RoundToInt((2f * HalfExtent) / floorModule));
        float floorStart = -(floorTiles * floorModule) / 2f;
        var floorParent = new GameObject("Floor").transform;
        floorParent.SetParent(root);
        for (int i = 0; i < floorTiles; i++)
        {
            for (int j = 0; j < floorTiles; j++)
            {
                float x = floorStart + floorModule * (i + 0.5f);
                float z = floorStart + floorModule * (j + 0.5f);
                Place(floorAsset, new Vector2(x, z), 0f, floorParent, woodMat, xRot: 90f, forceY: 0f);
            }
        }

        int wallSegs = Mathf.Max(1, Mathf.RoundToInt((2f * HalfExtent) / wallWidth));
        float wallStart = -(wallSegs * wallWidth) / 2f;
        float outer = HalfExtent + wallThick / 2f;
        int doorSeg = wallSegs / 2;

        var wallParent = new GameObject("Walls").transform;
        wallParent.SetParent(root);

        for (int k = 0; k < wallSegs; k++)
        {
            float along = wallStart + wallWidth * (k + 0.5f);

            Place(wallAsset, new Vector2(along, outer - 1.75f), 180f, wallParent, plasterMat, true, xRot: -90f, forceY: 0f);
            var frontPrefab = (k == doorSeg) ? wallDoorAsset : wallAsset;
            Place(frontPrefab, new Vector2(along, -outer + 1.75f), 0f, wallParent, plasterMat, true, xRot: -90f, forceY: 0f);
            if (k == doorSeg)
            {
                Place(frameAsset, new Vector2(along, -outer + 1.75f), 0f, wallParent, woodMat, true, xRot: -90f, forceY: 0f);
                Place(doorAsset, new Vector2(along, -outer + 1.75f + 0.06f), 30f, wallParent, woodMat, true, xRot: -90f, forceY: 0f);
            }
            Place(wallAsset, new Vector2(-outer + 1.75f, along), 90f, wallParent, plasterMat, true, xRot: -90f, forceY: 0f);
            Place(wallAsset, new Vector2(outer - 1.75f, along), 270f, wallParent, plasterMat, true, xRot: -90f, forceY: 0f);
        }

        var ceilingParent = new GameObject("Ceiling").transform;
        ceilingParent.SetParent(root);
        for (int i = 0; i < floorTiles; i++)
        {
            for (int j = 0; j < floorTiles; j++)
            {
                float x = floorStart + floorModule * (i + 0.5f);
                float z = floorStart + floorModule * (j + 0.5f);
                Place(floorAsset, new Vector2(x, z), 0f, ceilingParent, woodMat, xRot: -90f, forceY: CeilingHeight);
            }
        }

        var floorCol = new GameObject("Floor_Collider");
        floorCol.transform.SetParent(root);
        var box = floorCol.AddComponent<BoxCollider>();
        box.center = new Vector3(0f, -0.1f, 0f);
        box.size = new Vector3(2f * HalfExtent, 0.2f, 2f * HalfExtent);

        RemovePrimitive("Environment", new[] { "Floor", "Back Wall", "Left Wall", "Right Wall" });

        var player = GameObject.Find("Player");
        if (player != null)
            player.transform.position = new Vector3(0f, 1.05f, -(HalfExtent - 2f));

        EditorSceneManager.MarkSceneDirty(SceneManager.GetActiveScene());
        Debug.Log($"[TavernGen] Done. Floor {floorTiles}x{floorTiles} (module {floorModule:F2}), " +
                  $"{wallSegs} wall segments/side, door at segment {doorSeg}. Review and Ctrl+S.");
    }

    private static GameObject Place(GameObject prefab, Vector2 targetXZ, float yRot,
        Transform parent, Material mat, bool addCollider = false, float xRot = 0f, float? forceY = null)
    {
        var go = (GameObject)PrefabUtility.InstantiatePrefab(prefab);
        go.transform.SetParent(parent);
        go.transform.rotation = Quaternion.Euler(xRot, yRot, 0f);
        go.transform.position = Vector3.zero;

        var b = WorldBounds(go);
        Vector3 p = go.transform.position;
        float yPos = forceY.HasValue ? forceY.Value : p.y + (0f - b.min.y);
        go.transform.position = new Vector3(
            p.x + (targetXZ.x - b.center.x),
            yPos,
            p.z + (targetXZ.y - b.center.z));

        ApplyMaterial(go, mat);
        if (addCollider)
            foreach (var mf in go.GetComponentsInChildren<MeshFilter>())
                if (mf.GetComponent<Collider>() == null)
                    mf.gameObject.AddComponent<MeshCollider>();
        return go;
    }

    private static void ApplyMaterial(GameObject go, Material mat)
    {
        foreach (var r in go.GetComponentsInChildren<Renderer>())
        {
            var mats = new Material[r.sharedMaterials.Length];
            for (int i = 0; i < mats.Length; i++) mats[i] = mat;
            r.sharedMaterials = mats.Length > 0 ? mats : new[] { mat };
        }
    }

    private static float MeasureXZ(GameObject prefab)
    {
        var size = MeasureSize(prefab);
        return Mathf.Max(size.x, size.z);
    }

    private static Vector3 MeasureSize(GameObject prefab)
    {
        var temp = (GameObject)PrefabUtility.InstantiatePrefab(prefab);
        temp.transform.position = Vector3.zero;
        temp.transform.rotation = Quaternion.identity;
        var b = WorldBounds(temp);
        Object.DestroyImmediate(temp);
        return b.size;
    }

    private static Bounds WorldBounds(GameObject go)
    {
        var rs = go.GetComponentsInChildren<Renderer>();
        if (rs.Length == 0) return new Bounds(go.transform.position, Vector3.zero);
        var b = rs[0].bounds;
        foreach (var r in rs) b.Encapsulate(r.bounds);
        return b;
    }

    private static Material BuildMaterial(string name, string albedoPath, string normalPath)
    {
        if (!Directory.Exists(MaterialDir))
            Directory.CreateDirectory(MaterialDir);
        string path = $"{MaterialDir}/{name}.mat";

        var mat = AssetDatabase.LoadAssetAtPath<Material>(path);
        if (mat == null)
        {
            mat = new Material(Shader.Find("Standard"));
            AssetDatabase.CreateAsset(mat, path);
        }

        var albedo = AssetDatabase.LoadAssetAtPath<Texture2D>(albedoPath);
        if (albedo != null) mat.SetTexture("_MainTex", albedo);

        var normal = AssetDatabase.LoadAssetAtPath<Texture2D>(normalPath);
        if (normal != null)
        {
            EnsureNormalImport(normalPath);
            mat.SetTexture("_BumpMap", normal);
            mat.EnableKeyword("_NORMALMAP");
        }

        EditorUtility.SetDirty(mat);
        AssetDatabase.SaveAssets();
        return mat;
    }

    private static void EnsureNormalImport(string path)
    {
        var importer = AssetImporter.GetAtPath(path) as TextureImporter;
        if (importer != null && importer.textureType != TextureImporterType.NormalMap)
        {
            importer.textureType = TextureImporterType.NormalMap;
            importer.SaveAndReimport();
        }
    }

    private static void RemovePrimitive(string parentName, IEnumerable<string> childNames)
    {
        var parent = GameObject.Find(parentName);
        if (parent == null) return;
        foreach (var n in childNames)
        {
            var t = parent.transform.Find(n);
            if (t != null) Object.DestroyImmediate(t.gameObject);
        }
    }

    private static GameObject Load(string path)
    {
        var go = AssetDatabase.LoadAssetAtPath<GameObject>(path);
        if (go == null) Debug.LogError($"[TavernGen] Missing asset: {path}");
        return go;
    }
}
#endif
