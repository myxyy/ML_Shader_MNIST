Shader "ML_MNIST/ViewSignedFloat"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        [MaterialToggle] _nan ("Emphasis NaN", Float) = 0
        [MaterialToggle] _inf ("Emphasis Inf", Float) = 0
        [MaterialToggle] _eth ("Emphasis by threshold", Float) = 0
        [MaterialToggle] _test ("test", Float) = 0
        _Th ("Threashold", Float) = 100000
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "UnityCG.cginc"

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

            sampler2D _MainTex;
            float _nan;
            float _inf;
            float _eth;
            float _test;
            float _Th;

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                return o;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                float v = tex2D(_MainTex, i.uv).r;
                if (isnan(v) && _nan) return fixed4(0,1,0,1);
                if (isinf(v) && _inf) return fixed4(0,.5,0,1);
                if (abs(v)>_Th && _eth) return fixed4(0,1,0,1);
                if (!(v > 0 || v < 0 || v == 0) && _test) return fixed4(0,1,0,1);
                return v > 0 ? fixed4(v,0,0,1) : fixed4(0,0,-v,1);
            }
            ENDCG
        }
    }
}
