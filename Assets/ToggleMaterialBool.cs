using UnityEngine;

public class ToggleMaterialBool : MonoBehaviour
{
    [SerializeField] private Material material;
    [SerializeField] private CustomRenderTexture customRenderTexture;
    [SerializeField] private int initializeFrames;
    void Update()
    {
        if (initializeFrames > 0)
        {
            material.SetFloat("_reset", 1.0f);
            initializeFrames--;
            if (initializeFrames > 20) customRenderTexture.Initialize();
        }
        else
        {
            material.SetFloat("_reset", 0.0f);
        }
    }
}
