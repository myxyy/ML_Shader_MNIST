Shader "ML_MNIST/ShowResult"
{
    Properties
    {
        _Input ("Input", 2D) = "white" {}
        _Fetch ("Fetch Pos", Vector) = (0,0,1,1)
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" "Queue"="Geometry"}
        LOD 100

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "UnityCG.cginc"
            sampler2D _Input;
            float4 _Input_TexelSize;
            float4 _Fetch;

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                return o;
            }

            float2 uvg2lsize(float4 xywh, float2 size, float2 guv)
            {
                float2 xyuv = xywh.xy / size.xy;
                float2 xypwhuv = (xywh.xy + xywh.zw) / size.xy;
                return (guv - xyuv)/(xypwhuv - xyuv);
            }

            float2 uvl2gsize(float4 xywh, float2 size, float2 luv)
            {
                return (xywh.xy + xywh.zw * luv) / size.xy;
            }

            float tex2Drange(sampler2D tex, float2 size, float4 rect, float2 uv)
            {
                return tex2D(tex, uvl2gsize(rect, size, uv)).r;
            }


            float4 frag (v2f i) : SV_Target
            {
                float4 rect = _Fetch;
                float a = tex2Drange(_Input, _Input_TexelSize.zw, rect, float2(i.uv.x,.5));

                return i.uv.y < a ? 1 : 0;
            }
            ENDCG
        }
    }
}
