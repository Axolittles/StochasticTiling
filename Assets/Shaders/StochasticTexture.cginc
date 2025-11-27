#ifndef STOCHASTICTILING_INCLUDED
#define STOCHASTICTILING_INCLUDED

float2 hash2(int2 p)
{
    float2 h = float2(
        dot(float2(p), float2(127.1, 311.7)),
        dot(float2(p), float2(269.5, 183.3))
    );
    return frac(sin(h) * 43758.5453);
}

// simple scalar hash in [0,1)
float hash1(int2 p)
{
    float h = dot(float2(p), float2(12.9898, 78.233));
    return frac(sin(h) * 43758.5453123);
}

// smoothstep polynomial for value noise
float sCurve(float t)
{
    return t * t * (3.0 - 2.0 * t);
}

// 2D value noise in [0,1], continuous, deterministic on the integer lattice
float valueNoise(float2 x)
{
    int2 i = int2(floor(x));
    float2 f = frac(x);

    float v00 = hash1(i + int2(0, 0));
    float v10 = hash1(i + int2(1, 0));
    float v01 = hash1(i + int2(0, 1));
    float v11 = hash1(i + int2(1, 1));

    float2 u = float2(sCurve(f.x), sCurve(f.y));
    float v0 = lerp(v00, v10, u.x);
    float v1 = lerp(v01, v11, u.x);
    return lerp(v0, v1, u.y);
}

// Pack two independent value-noise channels into a vector in [-0.5,0.5]^2
float2 valueNoise2D(float2 x)
{
    // decorrelate channels by shifting coordinate
    float nx = valueNoise(x);
    float ny = valueNoise(x + float2(37.2, 19.7));
    return float2(nx, ny) - 0.5;
}

float4 Sample2DGrad(UnityTexture2D t, float2 uv, float2 dUVdx, float2 dUVdy)
{
#if defined(SHADER_API_GLES) || defined(SHADER_API_GLES3)
    return SAMPLE_TEXTURE2D(t.tex, t.samplerstate, uv);
#else
    return SAMPLE_TEXTURE2D_GRAD(t.tex, t.samplerstate, uv, dUVdx, dUVdy);
#endif
}

// Build a 0/90/180/270° rotation matrix from k = {0..3}
float2x2 rot90(int k)
{
    k &= 3;
    if (k == 0)
        return float2x2(1, 0, 0, 1);
    if (k == 1)
        return float2x2(0, -1, 1, 0);
    if (k == 2)
        return float2x2(-1, 0, 0, -1);
    return float2x2(0, 1, -1, 0);
}

// Triangle-lattice tiling with per-corner random transforms + curvy edges
void StochasticTiling_float(
    UnityTexture2D MainTexture,
    float2 UV,
    float Scale,
    float BlendSharpness,
    float EdgeNoiseAmp,
    float EdgeNoiseFreq,
    out float4 Color)
{
    // ---- Base UV & gradients (for texture sampling) ----
    float2 uvBase = UV * Scale;
    float2 dUVdx = ddx(uvBase);
    float2 dUVdy = ddy(uvBase);

    // ---- Skew into simplex space ----
    const float2x2 gridToSkewedGrid = float2x2(1.0, 0.0, -0.57735027, 1.15470054);
    float2 skewed = mul(gridToSkewedGrid, uvBase * 3.464); // ~2/sqrt(3)

    // ---- Curvy-edge warp: deterministic, continuous vector noise ----
    // Small, smooth displacement bends the w=0 lines (triangle edges).
    skewed += valueNoise2D(skewed * EdgeNoiseFreq) * EdgeNoiseAmp;
    skewed += valueNoise2D(skewed * (EdgeNoiseFreq / 10)) * (EdgeNoiseAmp * 10);

    // ---- Re-evaluate triangle & barycentrics in the WARPED space ----
    int2 baseId = int2(floor(skewed));
    float2 f = frac(skewed);
    float3 w = float3(f.x, f.y, 0.0);
    w.z = 1.0 - w.x - w.y;

    int2 v1, v2, v3;
    float W1, W2, W3;
    if (w.z > 0.0)
    {
        v1 = baseId;
        v2 = baseId + int2(0, 1);
        v3 = baseId + int2(1, 0);
        W1 = w.z;
        W2 = w.y;
        W3 = w.x;
    }
    else
    {
        v1 = baseId + int2(1, 1);
        v2 = baseId + int2(1, 0);
        v3 = baseId + int2(0, 1);
        W1 = -w.z;
        W2 = 1.0 - w.y;
        W3 = 1.0 - w.x;
    }

    // ---- Per-corner random transforms (rotation + optional mirror) ----
    float2 local = frac(uvBase);

    float2 r1 = hash2(v1);
    float2 r2 = hash2(v2);
    float2 r3 = hash2(v3);

    int k1 = (int) floor(r1.x * 4.0);
    int k2 = (int) floor(r2.x * 4.0);
    int k3 = (int) floor(r3.x * 4.0);

    float2x2 R1 = rot90(k1);
    float2x2 R2 = rot90(k2);
    float2x2 R3 = rot90(k3);

    bool m1 = (r1.y > 0.5);
    bool m2 = (r2.y > 0.5);
    bool m3 = (r3.y > 0.5);

    float2x2 M1 = float2x2(m1 ? -1 : 1, 0, 0, 1);
    float2x2 M2 = float2x2(m2 ? -1 : 1, 0, 0, 1);
    float2x2 M3 = float2x2(m3 ? -1 : 1, 0, 0, 1);

    float2x2 A1 = mul(R1, M1);
    float2x2 A2 = mul(R2, M2);
    float2x2 A3 = mul(R3, M3);

    float2 l1 = local;
    if (m1)
        l1.x = 1.0 - l1.x;
    float2 l2 = local;
    if (m2)
        l2.x = 1.0 - l2.x;
    float2 l3 = local;
    if (m3)
        l3.x = 1.0 - l3.x;

    // Independent random offsets in [0,1)
    float2 t1 = frac(hash2(v1 + 13));
    float2 t2 = frac(hash2(v2 + 17));
    float2 t3 = frac(hash2(v3 + 29));

    // Transformed UVs
    float2 uv1 = frac(mul(A1, l1) + t1);
    float2 uv2 = frac(mul(A2, l2) + t2);
    float2 uv3 = frac(mul(A3, l3) + t3);

    // Gradients (linear part only)
    float2 g1x = mul(A1, dUVdx), g1y = mul(A1, dUVdy);
    float2 g2x = mul(A2, dUVdx), g2y = mul(A2, dUVdy);
    float2 g3x = mul(A3, dUVdx), g3y = mul(A3, dUVdy);

    // Sample the texture 3 times at different points
    float4 c1 = Sample2DGrad(MainTexture, uv1, g1x, g1y);
    float4 c2 = Sample2DGrad(MainTexture, uv2, g2x, g2y);
    float4 c3 = Sample2DGrad(MainTexture, uv3, g3x, g3y);
    
    if (BlendSharpness != 1.0)
    {
        W1 = pow(W1, BlendSharpness);
        W2 = pow(W2, BlendSharpness);
        W3 = pow(W3, BlendSharpness);
        float inv = 1.0 / max(1e-6, (W1 + W2 + W3));
        W1 *= inv;
        W2 *= inv;
        W3 *= inv;
    }

    // Blend our samples by weights
    Color = c1 * W1 + c2 * W2 + c3 * W3;
}

#endif
