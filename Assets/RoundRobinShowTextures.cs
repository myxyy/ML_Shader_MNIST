using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;

public class RoundRobinShowTextures : MonoBehaviour
{
    [SerializeField] private Texture2D[] textures;
    [SerializeField] private Material material;
    private float timeAcc = 0;
    private int index = 0;
    void Update()
    {
        material.SetTexture("_MainTex", textures[index]);
        timeAcc += Time.deltaTime;
        if (timeAcc > 1.0f)
        {
            index = (index + 1) % textures.Length;
            timeAcc = 0;
        }
    }
}
