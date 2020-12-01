Shader "ML_MNIST/Calc"
{
    Properties
    {
        _MainTex ("MNIST", 2D) = "white" {}
        _Size ("CRTSize", Vector) = (1,1,1,1)
        _N ("N", Int) = 70000
        _lr ("Learning rate", Float) = 1
        _beta ("RMSProp rate", Float) = 1
        _beta2 ("Momentum rate", Float) = 1
        _amp ("Initial amplitude", Float) = 1
        [MaterialToggle] _reset ("Reset", Float) = 0
        _Input ("Input", 2D) = "white" {}
        _Test ("Test", Vector) = (0,0,0,0)
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" "Queue"="Geometry"}
        LOD 100

        Pass
        {
            Cull Off
            ZWrite Off
            ZTest Always
            CGPROGRAM
            #include "UnityCustomRenderTexture.cginc"
            #pragma vertex CustomRenderTextureVertexShader 
            #pragma fragment frag

            #include "UnityCG.cginc"
            float4 _Size;
            uint _N;
            float _lr;
            sampler2D _MainTex;
            float4 _MainTex_TexelSize;
            sampler2D _Input;
            float4 _Input_TexelSize;
            float _reset;
            float4 _Test;
            float _amp;
            float _beta;
            float _beta2;

            #define WHOLE (float4(0,0,_Size.xy))

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

            bool isvaliduv(float2 uv)
            {
                return 0 <= uv.x && uv.x <= 1 && 0 <= uv.y && uv.y <= 1;
            }

            float tex2Drange(sampler2D tex, float2 size, float4 rect, float2 uv)
            {
                return tex2D(tex, uvl2gsize(rect, size, uv)).r;
            }

            float2 uvg2l(float4 xywh, float2 guv)
            {
                float2 xyuv = xywh.xy / _Size.xy;
                float2 xypwhuv = (xywh.xy + xywh.zw) / _Size.xy;
                return (guv - xyuv)/(xypwhuv - xyuv);
            }

            float2 uvl2g(float4 xywh, float2 luv)
            {
                return (xywh.xy + xywh.zw * luv) / _Size.xy;
            }

            bool isrange(float4 xywh, float2 guv)
            {
                float2 luv = uvg2l(xywh, guv);
                return isvaliduv(luv);
            }

            float getvaluefromtexture(uint index)
            {
                int x = (index % (int)(_MainTex_TexelSize.z * 4)) / 4;
                int y = index / (int)(_MainTex_TexelSize.z * 4);
                float2 uv = (float2(x,y)+.5)/_MainTex_TexelSize.zw;
                uv.y = 1-uv.y;
                switch (index % 4)
                {
                    case 0:
                        return tex2D(_MainTex, uv).r;
                    case 1:
                        return tex2D(_MainTex, uv).g;
                    case 2:
                        return tex2D(_MainTex, uv).b;
                    case 3:
                        return tex2D(_MainTex, uv).a;
                    default:
                        return 1;
                }
            }

            float tex2Dself(float4 rect, float2 uv)
            {
                return tex2D(_SelfTexture2D, uvl2g(rect,uv)).r;
            }

            void delay(float4 inrect, float4 workrect, float2 uv, inout float value)
            {
                if (isrange(workrect,uv))
                {
                    float2 wuv = uvg2l(workrect, uv);
                    float row = floor(wuv.y*workrect.w);
                    value = tex2Dself(float4(row>0?workrect.x:inrect.x, row>0?workrect.y+row-1:inrect.y, workrect.z, 1),float2(wuv.x, .5));
                }
            }

            float4 delay_out(float4 delayworkrect)
            {
                return float4(delayworkrect.x,delayworkrect.y+delayworkrect.w-1,delayworkrect.z,1);
            }

            void mul(float4 arect, float4 brect, float4 orect, float2 uv, inout float value)
            {
                if (isrange(orect,uv))
                {
                    float2 luv = uvg2l(orect,uv);
                    value = tex2Dself(arect,luv)*tex2Dself(brect,luv);
                }
            }

            float h12(float2 p)
            {
                return frac(sin(dot(float2(23.25425,44.3254),p))*14321.5415);
            }

            void mul_transposed(float4 arect, float4 brect, float4 orect, float2 uv, inout float value)
            {
                if (isrange(orect,uv))
                {
                    float2 luv = uvg2l(orect,uv);
                    value = tex2Dself(arect,luv)*tex2Dself(brect,luv.yx);
                }
            }

            void lcomb(float4 arect, float ac, float4 brect, float bc, float4 orect, float2 uv, inout float value)
            {
                if (isrange(orect,uv))
                {
                    float2 luv = uvg2l(orect,uv);
                    float a = tex2Dself(arect,luv);
                    float b = tex2Dself(brect,luv);
                    value = a*ac+b*bc;
                }
            }

            void randomize(float4 orect, float amp, float2 uv, inout float value)
            {
                if (isrange(orect,uv))
                {
                    float2 iuv = floor(uvg2l(orect,uv)*orect.zw);
                    //value = (2*step(.5,h12(iuv))-1)*amp;
                    value = (2*h12(iuv)-1)*amp;
                }
            }

            float add_naiive(float4 arect, float2 uv)
            {
                float a = 0;
                for (int i=0;i+.5<arect.z;i++)
                {
                    float2 tuv = float2((i+.5)/arect.z,.5);
                    a += tex2Dself(arect,tuv);
                }
                return a;
            }

            float relu(float x)
            {
                return max(0,x);
            }

            void relu(float4 irect, float4 orect, float2 uv, inout float value)
            {
                if (isrange(orect,uv))
                {
                    float2 luv = uvg2l(orect,uv);
                    value = relu(tex2Dself(irect,luv));
                }
            }

            void step(float4 irect, float4 orect, float2 uv, inout float value)
            {
                if (isrange(orect,uv))
                {
                    float2 luv = uvg2l(orect,uv);
                    value = step(0,tex2Dself(irect,luv));
                }
            }

            float tanh4large(float x)
            {
                float e = x > 0 ? exp(-2*x) : exp(2*x);
                return x > 0 ? (1-e)/(1+e) : (e-1)/(e+1);
            }

            void tanh(float4 irect, float4 orect, float2 uv, inout float value)
            {
                if (isrange(orect,uv))
                {
                    float2 luv = uvg2l(orect,uv);
                    value = tanh4large(tex2Dself(irect,luv));
                }
            }

            void dtanh(float4 irect, float4 orect, float2 uv, inout float value)
            {
                if (isrange(orect,uv))
                {
                    float2 luv = uvg2l(orect,uv);
                    float t = cosh(tex2Dself(irect,luv));
                    value = rcp(t*t);
                }
            }

            void id(float4 irect, float4 orect, float2 uv, inout float value)
            {
                if (isrange(orect,uv))
                {
                    float2 luv = uvg2l(orect,uv);
                    value = tex2Dself(irect,luv);
                }
            }

            void softmax_naiive(float4 inrect, float4 outrect, float2 uv, inout float value)
            {
                if (isrange(outrect,uv))
                {
                    float a = 0;
                    float m = -1e10;
                    float2 luv = uvg2l(outrect,uv);
                    int j;
                    for (j=0;j+.5<inrect.z;j++)
                    {
                        float2 tuv = float2((j+.5)/inrect.z,.5);
                        m = max(tex2Dself(inrect,tuv),m);
                    }
                    for (j=0;j+.5<inrect.z;j++)
                    {
                        float2 tuv = float2((j+.5)/inrect.z,.5);
                        a += exp(tex2Dself(inrect,tuv)-m);
                    }
                    value = exp(tex2Dself(inrect,luv)-m)/a;
                }
            }

            void action_linear_1024d(float4 lrect, float4 vrect, float4 wrect, float4 orect, float2 uv, inout float value)
            {
                if (isrange(wrect,uv))
                {
                    float2 luv = uvg2l(wrect,uv);
                    if (luv.x > .5)
                    {
                        float ind = floor((2*luv.x-1)*512);
                        float2 uv0 = float2((ind*2+0.5)/(vrect.z),.5);
                        float2 uv1 = float2((ind*2+1.5)/(vrect.z),.5);
                        value =
                            (isvaliduv(uv0) ? (tex2Dself(lrect,uv0.yx) * tex2Dself(vrect,uv0)) : 0) +
                            (isvaliduv(uv1) ? (tex2Dself(lrect,uv1.yx) * tex2Dself(vrect,uv1)) : 0);
                    }
                    else
                    {
                        float ind = floor(luv.x*1024);
                        value = tex2Dself(wrect,float2((ind*2+0.5)/1024,.5))+tex2Dself(wrect,float2((ind*2+1.5)/1024,.5));
                    }
                }
                if (isrange(orect,uv))
                {
                    value = tex2Dself(wrect,float2(1.5/1024,.5));
                }
            }

            void action_1024d(float4 mrect, float4 vrect, float4 wrect, float4 orect, float2 uv, inout float value)
            {
                float row = 0;
                if (isrange(wrect,uv))
                {
                    row = floor(uvg2l(wrect,uv).y*mrect.z);
                }
                if (isrange(orect,uv))
                {
                    row = floor(uvg2l(orect,uv).x*mrect.z);
                }
                float4 mrect_i = float4(mrect.x+row,mrect.y,1,mrect.w);
                float4 wrect_i = float4(wrect.x,wrect.y+row,wrect.z,1);
                float4 orect_i = float4(orect.x+row,orect.y,1,1);
                if (isrange(wrect,uv) || isrange(orect,uv))
                {
                    action_linear_1024d(mrect_i,vrect,wrect_i,orect_i,uv,value);
                }
            }

            void transpose_action_linear_1024d(float4 lrect, float4 vrect, float4 wrect, float4 orect, float2 uv, inout float value)
            {
                if (isrange(wrect,uv))
                {
                    float2 luv = uvg2l(wrect,uv);
                    if (luv.x > .5)
                    {
                        float ind = floor((2*luv.x-1)*512);
                        float2 uv0 = float2((ind*2+0.5)/(vrect.z),.5);
                        float2 uv1 = float2((ind*2+1.5)/(vrect.z),.5);
                        value =
                            (isvaliduv(uv0) ? (tex2Dself(lrect,uv0.xy) * tex2Dself(vrect,uv0)) : 0) +
                            (isvaliduv(uv1) ? (tex2Dself(lrect,uv1.xy) * tex2Dself(vrect,uv1)) : 0);
                    }
                    else
                    {
                        float ind = floor(luv.x*1024);
                        value = tex2Dself(wrect,float2((ind*2+0.5)/1024,.5))+tex2Dself(wrect,float2((ind*2+1.5)/1024,.5));
                    }
                }
                if (isrange(orect,uv))
                {
                    value = tex2Dself(wrect,float2(1.5/1024,.5));
                }
            }

            void transpose_action_1024d(float4 mrect, float4 vrect, float4 wrect, float4 orect, float2 uv, inout float value)
            {
                float row = 0;
                if (isrange(wrect,uv))
                {
                    row = floor(uvg2l(wrect,uv).y*mrect.w);
                }
                if (isrange(orect,uv))
                {
                    row = floor(uvg2l(orect,uv).x*mrect.w);
                }
                float4 mrect_i = float4(mrect.x,mrect.y+row,mrect.z,1);
                float4 wrect_i = float4(wrect.x,wrect.y+row,wrect.z,1);
                float4 orect_i = float4(orect.x+row,orect.y,1,1);
                if (isrange(wrect,uv) || isrange(orect,uv))
                {
                    transpose_action_linear_1024d(mrect_i,vrect,wrect_i,orect_i,uv,value);
                }
            }

            void rmsprop_moving_average(float4 gradrect, float4 avgrect, float beta, float2 uv, inout float value)
            {
                if (isrange(avgrect,uv))
                {
                    float2 luv = uvg2l(avgrect,uv);
                    float avg = tex2Dself(avgrect,luv);
                    float grad = tex2Dself(gradrect,luv);
                    value = beta*avg+(1-beta)*grad*grad;
                }
            }

            void rmsprop_normalize(float4 gradrect, float4 avgrect, float4 normrect, float2 uv, inout float value)
            {
                if (isrange(normrect,uv))
                {
                    float2 luv = uvg2l(normrect,uv);
                    float avg = tex2Dself(avgrect,luv);
                    float grad = tex2Dself(gradrect,luv);
                    value = grad/sqrt(avg+1e-12);
                }
            }

            void set(float4 outrect, float v, float2 uv, inout float value)
            {
                if (isrange(outrect,uv))
                {
                    value = v;
                }
            }

            float frag (v2f_customrendertexture i) : SV_Target
            {
                float2 uv = i.globalTexcoord;
                float value = 0;
                float main = 1024;
                float rmsprop = 0;

                //Data fetch
                float4 counter = float4(1023,0,1,1);

                uint index = (uint)floor(tex2Dself(counter, float2(.5,.5)));
                if (_reset) index = 0;
                if (isrange(counter,uv))
                {
                    value = floor(fmod(index+1.5, _N));
                }
               
                float4 mnist_X = float4(main,0,28*28,1);
                if (isrange(mnist_X,uv))
                {
                    float2 luv = uvg2l(mnist_X,uv);
                    value = getvaluefromtexture((uint)floor(luv.x*28*28) + index*(28*28+1));
                }

                //Learning
                float4 mnist_X1 = float4(mnist_X.x+mnist_X.z,mnist_X.y,1,1);
                set(mnist_X1,1,uv,value);

                float4 xb1 = float4(main,mnist_X.y,mnist_X.z+1,mnist_X.w); //0

                float4 Wb1 = float4(main,2,800,xb1.z); //-
                if (_reset) randomize(Wb1,_amp,uv,value);

                float4 v1work = float4(main,Wb1.y+Wb1.w+2,1024,Wb1.z);

                float4 v1 = float4(main,v1work.y+v1work.w+1,Wb1.z,1); //11

                action_1024d(Wb1,xb1,v1work,v1,uv,value);

                float4 x2 = float4(main,v1.y+2,v1.zw); //12
                relu(v1,x2,uv,value);
                //tanh(v1,x2,uv,value);

                float4 xb2_b = float4(x2.x+x2.z,x2.y,1,1);
                set(xb2_b,1,uv,value);

                float4 xb2 = float4(main,x2.y,x2.z+1,x2.w); //12

                float4 Wb2 = float4(main,xb2.y+2,600,xb2.z); //-
                if (_reset) randomize(Wb2,_amp,uv,value);

                float4 v2work = float4(main,Wb2.y+Wb2.w+1,1024,Wb2.z);

                float4 v2 = float4(main,v2work.y+v2work.w+1,Wb2.z,1); //23
                action_1024d(Wb2,xb2,v2work,v2,uv,value);

                float4 x3 = float4(main,v2.y+2,v2.zw); //24
                relu(v2,x3,uv,value);
                //tanh(v2,x3,uv,value);

                float4 xb3_b = float4(x3.x+x3.z,x3.y,1,1);
                set(xb3_b,1,uv,value);

                float4 xb3 = float4(main,x3.y,x3.z+1,x3.w); //24

                float4 Wb3 = float4(main,xb3.y+2,10,xb3.z); //-
                if (_reset) randomize(Wb3,_amp,uv,value);

                float4 v3work = float4(main,Wb3.y+Wb3.w+1,1024,Wb3.z);
                float4 v3 = float4(main,v3work.y+v3work.w+1,Wb3.z,1); //35
                action_1024d(Wb3,xb3,v3work,v3,uv,value);

                float4 y3 =  float4(main,v3.y+2,v3.zw); //36

                softmax_naiive(v3,y3,uv,value);

                float4 mnist_y = float4(main,y3.y+2,10,1); //0
                if (isrange(mnist_y,uv))
                {
                    float2 luv = uvg2l(mnist_y,uv);
                    value = (floor(luv.x*10) == floor(getvaluefromtexture(index*(28*28+1)+28*28)*255+.5)?1:0);
                }

                float4 t_delay36_work = float4(main,mnist_y.y+1,10,36);
                delay(mnist_y,t_delay36_work,uv,value);
                float4 t_delay36 = delay_out(t_delay36_work); //36

                float4 dE_dv3 = float4(main,t_delay36.y+2,10,1); //37
                lcomb(y3,1,t_delay36,-1,dE_dv3,uv,value);

                float4 xb3_delay13_work = float4(main,dE_dv3.y+2,601,13);
                delay(xb3,xb3_delay13_work,uv,value);
                float4 xb3_delay13 = delay_out(xb3_delay13_work); //37

                float4 dE_dWb3 = float4(main,xb3_delay13.y+2,10,601); //38
                mul_transposed(dE_dv3,xb3_delay13,dE_dWb3,uv,value);

                float4 W3T_dE_dv3_work = float4(main,dE_dWb3.y+dE_dWb3.w+1,1024,600);
                float4 W3T_dE_dv3 = float4(main,W3T_dE_dv3_work.y+W3T_dE_dv3_work.w+1,600,1); //48
                float4 W3 = float4(Wb3.xyz,Wb3.w-1);
                transpose_action_1024d(W3,dE_dv3,W3T_dE_dv3_work,W3T_dE_dv3,uv,value);

                float4 v2_delay_24_work = float4(main,W3T_dE_dv3.y+2,600,24);
                delay(v2,v2_delay_24_work,uv,value);

                float4 v2_delay_24 = delay_out(v2_delay_24_work); //47
                float4 dy2_dv2 = float4(main,v2_delay_24.y+2,v2_delay_24.zw); //48
                step(v2_delay_24,dy2_dv2,uv,value);
                //dtanh(v2_delay_24,dy2_dv2,uv,value);

                float4 dE_dv2 = float4(main,dy2_dv2.y+2,dy2_dv2.zw); //49
                mul(W3T_dE_dv3,dy2_dv2,dE_dv2,uv,value);

                float4 xb2_delay_37_work = float4(main,dE_dv2.y+2,xb2.z,37);
                delay(xb2,xb2_delay_37_work,uv,value);
                float4 xb2_delay_37 = delay_out(xb2_delay_37_work); //49

                float4 dE_dWb2 = float4(main,xb2_delay_37.y+2,600,801); //50
                mul_transposed(dE_dv2,xb2_delay_37,dE_dWb2,uv,value);

                float4 W2T_dE_dv2_work = float4(main,dE_dWb2.y+dE_dWb2.w+1,1024,800);
                float4 W2T_dE_dv2 = float4(main,W2T_dE_dv2_work.y+W2T_dE_dv2_work.w+1,800,1); //60
                float4 W2 = float4(Wb2.xyz,Wb2.w-1);
                transpose_action_1024d(W2,dE_dv2,W2T_dE_dv2_work,W2T_dE_dv2,uv,value);

                float4 v1_delay_48_work = float4(main,W2T_dE_dv2.y+2,800,48);
                delay(v1,v1_delay_48_work,uv,value);

                float4 v1_delay_48 = delay_out(v1_delay_48_work); //59
                float4 dy1_dv1 = float4(main,v1_delay_48.y+2,800,1); //60
                step(v1_delay_48,dy1_dv1,uv,value);
                //dtanh(v1_delay_48,dy1_dv1,uv,value);

                float4 dE_dv1 = float4(main,dy1_dv1.y+2,800,1); //61
                mul(W2T_dE_dv2,dy1_dv1,dE_dv1,uv,value);

                float4 xb1_delay_61_work = float4(main,dE_dv1.y+2,xb1.z,61);
                delay(xb1,xb1_delay_61_work,uv,value);
                float4 xb1_delay_61 = delay_out(xb1_delay_61_work); //61

                float4 dE_dWb1 = float4(main,xb1_delay_61.y+2,800,28*28+1);
                mul_transposed(dE_dv1,xb1_delay_61,dE_dWb1,uv,value);

                float4 dE_dWb1_square_moving_average = float4(rmsprop,0,dE_dWb1.zw);
                float4 dE_dWb2_square_moving_average = float4(rmsprop,dE_dWb1_square_moving_average.y+dE_dWb1_square_moving_average.w+1,dE_dWb2.zw);
                float4 dE_dWb3_square_moving_average = float4(rmsprop,dE_dWb2_square_moving_average.y+dE_dWb2_square_moving_average.w+1,dE_dWb3.zw);
                if (_reset)
                {
                    set(dE_dWb1_square_moving_average,1000000,uv,value);
                    set(dE_dWb2_square_moving_average,1000000,uv,value);
                    set(dE_dWb3_square_moving_average,1000000,uv,value);
                }
                else
                {
                    rmsprop_moving_average(dE_dWb1,dE_dWb1_square_moving_average,_beta,uv,value);
                    rmsprop_moving_average(dE_dWb2,dE_dWb2_square_moving_average,_beta,uv,value);
                    rmsprop_moving_average(dE_dWb3,dE_dWb3_square_moving_average,_beta,uv,value);
                }
                float4 dE_dWb1_delay = float4(rmsprop,dE_dWb3_square_moving_average.y+dE_dWb3_square_moving_average.w+1,dE_dWb1.zw);
                float4 dE_dWb2_delay = float4(rmsprop,dE_dWb1_delay.y+dE_dWb1_delay.w+1,dE_dWb2.zw);
                float4 dE_dWb3_delay = float4(rmsprop,dE_dWb2_delay.y+dE_dWb2_delay.w+1,dE_dWb3.zw);
                lcomb(dE_dWb1_delay,_beta2,dE_dWb1,(1-_beta2),dE_dWb1_delay,uv,value);
                lcomb(dE_dWb2_delay,_beta2,dE_dWb2,(1-_beta2),dE_dWb2_delay,uv,value);
                lcomb(dE_dWb3_delay,_beta2,dE_dWb3,(1-_beta2),dE_dWb3_delay,uv,value);

                float4 dE_dWb1_norm = float4(rmsprop,dE_dWb3_delay.y+dE_dWb3_delay.w+1,dE_dWb1.zw);
                float4 dE_dWb2_norm = float4(rmsprop,dE_dWb1_norm.y+dE_dWb1_norm.w+1,dE_dWb2.zw);
                float4 dE_dWb3_norm = float4(rmsprop,dE_dWb2_norm.y+dE_dWb2_norm.w+1,dE_dWb3.zw);
                rmsprop_normalize(dE_dWb1_delay,dE_dWb1_square_moving_average,dE_dWb1_norm,uv,value);
                rmsprop_normalize(dE_dWb2_delay,dE_dWb2_square_moving_average,dE_dWb2_norm,uv,value);
                rmsprop_normalize(dE_dWb3_delay,dE_dWb3_square_moving_average,dE_dWb3_norm,uv,value);

                if (!_reset) lcomb(Wb1,1,dE_dWb1_norm,-_lr,Wb1,uv,value);
                if (!_reset) lcomb(Wb2,1,dE_dWb2_norm,-_lr,Wb2,uv,value);
                if (!_reset) lcomb(Wb3,1,dE_dWb3_norm,-_lr,Wb3,uv,value);

                //Prediction
                float4 input_x = float4(rmsprop, dE_dWb3_norm.y+dE_dWb3_norm.w+1,28*28,1);
                if (isrange(input_x,uv))
                {
                    float2 luv = uvg2l(input_x,uv);
                    float ind = floor(luv.x*28*28);
                    value = tex2Drange(_Input, _Input_TexelSize.zw, float4(0,0,28,28), float2((fmod(ind,28)+.5)/28,1-(floor(ind/28)+.5)/28));
                }
                float4 input_xb1_b = float4(input_x.z,input_x.y,1,1);
                set(input_xb1_b,1,uv,value);
                float4 ixb1 = float4(rmsprop,input_x.y,input_x.z+1,1);
                float4 iv1_work = float4(rmsprop,ixb1.y+2,1024,800);
                float4 iv1 = float4(rmsprop,iv1_work.y+iv1_work.w+1,800,1);
                action_1024d(Wb1,ixb1,iv1_work,iv1,uv,value);
                float4 ix2 = float4(rmsprop,iv1.y+2,800,1);
                relu(iv1,ix2,uv,value);
                float4 ixb2_b = float4(ix2.z,ix2.y,1,1);
                set(ixb2_b,1,uv,value);
                float4 ixb2 = float4(rmsprop,ix2.y,ix2.z+1,1);
                float4 iv2_work = float4(rmsprop,ixb2.y+2,1024,600);
                float4 iv2 = float4(rmsprop,iv2_work.y+iv2_work.w+1,600,1);
                action_1024d(Wb2,ixb2,iv2_work,iv2,uv,value);
                float4 ix3 = float4(rmsprop,iv2.y+2,600,1);
                relu(iv2,ix3,uv,value);
                float4 ixb3_b = float4(ix3.z,ix3.y,1,1);
                set(ixb3_b,1,uv,value);
                float4 ixb3 = float4(rmsprop,ix3.y,ix3.z+1,1);
                float4 iv3_work = float4(rmsprop,ixb3.y+2,1024,10);
                float4 iv3 = float4(rmsprop,iv3_work.y+iv3_work.w+1,10,1);
                action_1024d(Wb3,ixb3,iv3_work,iv3,uv,value);
                float4 predict = float4(rmsprop,iv3.y+2,10,1);
                softmax_naiive(iv3,predict,uv,value);

                //Visualize error rate
                float4 iscorrect = float4(main+1024-16,1024*7,16,1);
                if (isrange(iscorrect,uv))
                {
                    float m = 0;
                    float c = 0;
                    float a;
                    for (int j=0;j<10;j++)
                    {
                        a = tex2Dself(y3,float2((j+.5)/10.,.5));
                        m = max(m,a);
                        if (tex2Dself(t_delay36,float2((j+.5)/10.,.5))>.5) c = a;
                    }
                    value = m == c ? 1 : 0;
                }
                float4 iscorrectbuff = float4(iscorrect.x,iscorrect.y+1,16,1023);
                delay(iscorrect,iscorrectbuff,uv,value);

                //set(_Test,1,uv,value);

                return value;
            }
            ENDCG
        }
    }
}
