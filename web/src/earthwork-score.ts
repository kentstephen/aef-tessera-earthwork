import type { Texture } from "@luma.gl/core";
import type { ShaderModule } from "@luma.gl/shadertools";

/**
 * Props for the {@link EarthworkScore} shader module.
 */
export type EarthworkScoreProps = {
  /** r8sint Texture2DArray, 64 layers: the window's first year. */
  firstTex: Texture;
  /** r8sint Texture2DArray, 64 layers: the window's last year. */
  lastTex: Texture;
  /** r32float 256 x 1: the logistic regression's weights. */
  weightTex: Texture;
  /** The logistic regression's intercept. */
  bias: number;
  /** Change (1 - cos) at which the gate opens. */
  gateLo: number;
  /** Change (1 - cos) at which the gate is fully open. */
  gateHi: number;
};

const MODULE_NAME = "earthworkScore";

/**
 * The earthwork score per pixel, as earthwork.py computes it per row.
 *
 * Both years are dequantized ((v / 127.5)^2 * sign(v)) and scaled to unit
 * length (b = first year, a = last year). The model is a logistic regression
 * on [b, a, a * b, (a - b)^2], so weight i applies to b[i], 64 + i to a[i],
 * 128 + i to a[i] * b[i] and 192 + i to (a[i] - b[i])^2. The chance is
 * multiplied by a gate on how far AlphaEarth moved, clip((1 - cos - lo) /
 * (hi - lo), 0, 1), so ground that barely changed (open desert) stays dark.
 *
 * Writes the score (0..1) to color.r for a following Colormap module.
 * Pixels missing in either year (-128) are discarded.
 */
export const EarthworkScore = {
  name: MODULE_NAME,
  fs: `\
uniform ${MODULE_NAME}Uniforms {
  float bias;
  float gateLo;
  float gateHi;
} ${MODULE_NAME};
`,
  inject: {
    "fs:#decl": `
precision highp isampler2DArray;
precision highp sampler2D;
uniform highp isampler2DArray firstTex;
uniform highp isampler2DArray lastTex;
uniform highp sampler2D weightTex;

// the score, kept for the alpha step after the colormap (EarthworkAlpha)
float earthworkScore_v;

float earthworkScore_dequant(int v) {
  float f = float(v) / 127.5;
  return f * abs(f);
}

float earthworkScore_w(int i) {
  return texelFetch(weightTex, ivec2(i, 0), 0).r;
}
`,
    "fs:DECKGL_FILTER_COLOR": /* glsl */ `
      ivec2 size = textureSize(firstTex, 0).xy;
      ivec2 px = clamp(ivec2(geometry.uv * vec2(size)), ivec2(0), size - 1);
      float b[64];
      float a[64];
      float nb = 0.0;
      float na = 0.0;
      for (int i = 0; i < 64; i++) {
        int vb = texelFetch(firstTex, ivec3(px, i), 0).r;
        int va = texelFetch(lastTex, ivec3(px, i), 0).r;
        if (vb == -128 || va == -128) discard;
        b[i] = earthworkScore_dequant(vb);
        a[i] = earthworkScore_dequant(va);
        nb += b[i] * b[i];
        na += a[i] * a[i];
      }
      float ib = 1.0 / max(sqrt(nb), 1e-9);
      float ia = 1.0 / max(sqrt(na), 1e-9);
      float z = ${MODULE_NAME}.bias;
      float cosine = 0.0;
      for (int i = 0; i < 64; i++) {
        float bi = b[i] * ib;
        float ai = a[i] * ia;
        float d = ai - bi;
        cosine += ai * bi;
        z += earthworkScore_w(i) * bi
           + earthworkScore_w(64 + i) * ai
           + earthworkScore_w(128 + i) * ai * bi
           + earthworkScore_w(192 + i) * d * d;
      }
      float gate = clamp(((1.0 - cosine) - ${MODULE_NAME}.gateLo)
                         / (${MODULE_NAME}.gateHi - ${MODULE_NAME}.gateLo), 0.0, 1.0);
      float score = gate / (1.0 + exp(-z));
      earthworkScore_v = score;
      color = vec4(score, score, score, 1.0);
    `,
  },
  uniformTypes: {
    bias: "f32",
    gateLo: "f32",
    gateHi: "f32",
  },
  getUniforms: (props: Partial<EarthworkScoreProps>) => ({
    firstTex: props.firstTex,
    lastTex: props.lastTex,
    weightTex: props.weightTex,
    bias: props.bias ?? 0,
    gateLo: props.gateLo ?? 0.05,
    gateHi: props.gateHi ?? 0.15,
  }),
} as const satisfies ShaderModule<EarthworkScoreProps>;

/**
 * After the colormap: quiet ground faint, likely earthwork nearly opaque, as
 * earthwork.py's hexagons were (alpha 70 to 235 of 255 along the score).
 */
export const EarthworkAlpha = {
  name: "earthworkAlpha",
  inject: {
    "fs:DECKGL_FILTER_COLOR": /* glsl */ `
      color.a = mix(70.0 / 255.0, 235.0 / 255.0, clamp(earthworkScore_v, 0.0, 1.0));
    `,
  },
} as const satisfies ShaderModule;

/**
 * One embedding's pair in a single pass, for 128 bands without holding both
 * vectors: with B = |b| and A = |a|, z = (b.wb) / B + (a.wa) / A + (a*b).wab /
 * (A B) + sum wd (a / A - b / B)^2, the last term expanded into a.a, b.b and
 * a.b sums. AlphaEarth is dequantized first and missing at -128; TESSERA is
 * taken as is.
 */
const HALF_GLSL = /* glsl */ `
precision highp isampler2DArray;
precision highp usampler2D;
precision highp sampler2D;
uniform highp sampler2D weightTex;

// the score, kept for the steps after it (EarthworkAlpha, EarthworkCut)
float earthworkScore_v;

float earthworkHalf_w(int i) {
  return texelFetch(weightTex, ivec2(i, 0), 0).r;
}

float earthworkHalf_v(int v, bool dq) {
  float f = float(v);
  if (!dq) return f;
  f /= 127.5;
  return f * abs(f);
}

// z from weights off .. off + 4d, and the pair's cosine; false where a year is missing
bool earthworkHalf(highp isampler2DArray fT, highp isampler2DArray lT, ivec2 px, int d, int off, bool dq,
                   out float z, out float cosine) {
  float sb = 0.0, sa = 0.0, sab = 0.0, daa = 0.0, dbb = 0.0, dab = 0.0, nb = 0.0, na = 0.0, ab = 0.0;
  for (int i = 0; i < d; i++) {
    int vb = texelFetch(fT, ivec3(px, i), 0).r;
    int va = texelFetch(lT, ivec3(px, i), 0).r;
    if (dq && (vb == -128 || va == -128)) return false;
    float b = earthworkHalf_v(vb, dq);
    float a = earthworkHalf_v(va, dq);
    float wd = earthworkHalf_w(off + 3 * d + i);
    sb += earthworkHalf_w(off + i) * b;
    sa += earthworkHalf_w(off + d + i) * a;
    sab += earthworkHalf_w(off + 2 * d + i) * a * b;
    daa += wd * a * a;
    dbb += wd * b * b;
    dab += wd * a * b;
    nb += b * b;
    na += a * a;
    ab += a * b;
  }
  float ib = 1.0 / max(sqrt(nb), 1e-9);
  float ia = 1.0 / max(sqrt(na), 1e-9);
  z = sb * ib + sa * ia + sab * ia * ib + daa * ia * ia + dbb * ib * ib - 2.0 * dab * ia * ib;
  cosine = ab * ia * ib;
  return true;
}
`;

/** Props for {@link EarthworkTessera}. */
export type EarthworkTesseraProps = {
  /** r8sint Texture2DArray, 128 layers: TESSERA's first and last year on the chunk's pixels. */
  tesFirst: Texture;
  tesLast: Texture;
  /** r8uint: bit 0 both years embedded, bit 1 its radar came or went, bit 2 not read (AlphaEarth quiet). */
  tesFlags: Texture;
  /** r32float 512 x 1. */
  weightTex: Texture;
  bias: number;
};

const TES_NAME = "earthworkTessera";

/**
 * TESSERA's own layer: the logistic regression on its unit vectors, no gate
 * (as the notebook's). Pixels not embedded in either year are discarded.
 */
export const EarthworkTessera = {
  name: TES_NAME,
  fs: `\
uniform ${TES_NAME}Uniforms {
  float bias;
} ${TES_NAME};
`,
  inject: {
    "fs:#decl": `${HALF_GLSL}
uniform highp isampler2DArray tesFirst;
uniform highp isampler2DArray tesLast;
uniform highp usampler2D tesFlags;
`,
    "fs:DECKGL_FILTER_COLOR": /* glsl */ `
      ivec2 size = textureSize(tesFlags, 0);
      ivec2 px = clamp(ivec2(geometry.uv * vec2(size)), ivec2(0), size - 1);
      if ((texelFetch(tesFlags, px, 0).r & 1u) == 0u) discard;
      float z, ct;
      earthworkHalf(tesFirst, tesLast, px, 128, 0, false, z, ct);
      float score = 1.0 / (1.0 + exp(-(z + ${TES_NAME}.bias)));
      earthworkScore_v = score;
      color = vec4(score, score, score, 1.0);
    `,
  },
  uniformTypes: { bias: "f32" },
  getUniforms: (props: Partial<EarthworkTesseraProps>) => ({
    tesFirst: props.tesFirst,
    tesLast: props.tesLast,
    tesFlags: props.tesFlags,
    weightTex: props.weightTex,
    bias: props.bias ?? 0,
  }),
} as const satisfies ShaderModule<EarthworkTesseraProps>;

/** Props for {@link EarthworkAggregate}. */
export type EarthworkAggregateProps = EarthworkTesseraProps & {
  /** r8sint Texture2DArray, 64 layers: AlphaEarth's first and last year. */
  aefFirst: Texture;
  aefLast: Texture;
  /** r32float 768 x 1: AlphaEarth's 256 weights, then TESSERA's 512. */
  weightTex: Texture;
  /** AlphaEarth's gate (1 - cos): opens, fully open. */
  gateLo: number;
  gateHi: number;
  /** TESSERA's gate (1 - cos): opens, fully open. */
  tLo: number;
  tHi: number;
};

const AGG_NAME = "earthworkAggregate";

/**
 * The aggregate: one logistic regression on both embeddings' pairs, times a
 * gate that opens on either one's change (only AlphaEarth's where TESSERA's
 * radar came or went). Only where both years of both exist.
 */
export const EarthworkAggregate = {
  name: AGG_NAME,
  fs: `\
uniform ${AGG_NAME}Uniforms {
  float bias;
  float gateLo;
  float gateHi;
  float tLo;
  float tHi;
} ${AGG_NAME};
`,
  inject: {
    "fs:#decl": `${HALF_GLSL}
uniform highp isampler2DArray aefFirst;
uniform highp isampler2DArray aefLast;
uniform highp isampler2DArray tesFirst;
uniform highp isampler2DArray tesLast;
uniform highp usampler2D tesFlags;
`,
    "fs:DECKGL_FILTER_COLOR": /* glsl */ `
      ivec2 size = textureSize(tesFlags, 0);
      ivec2 px = clamp(ivec2(geometry.uv * vec2(size)), ivec2(0), size - 1);
      uint fl = texelFetch(tesFlags, px, 0).r;
      float score = 0.0;
      // bit 2: TESSERA not read here, AlphaEarth saw nothing worth it (aefScores): scored 0
      if ((fl & 4u) == 0u) {
        if ((fl & 1u) == 0u) discard;
        float za, ca, zt, ct;
        if (!earthworkHalf(aefFirst, aefLast, px, 64, 0, true, za, ca)) discard;
        earthworkHalf(tesFirst, tesLast, px, 128, 256, false, zt, ct);
        float ga = clamp(((1.0 - ca) - ${AGG_NAME}.gateLo) / (${AGG_NAME}.gateHi - ${AGG_NAME}.gateLo), 0.0, 1.0);
        float gt = clamp(((1.0 - ct) - ${AGG_NAME}.tLo) / max(${AGG_NAME}.tHi - ${AGG_NAME}.tLo, 1e-6), 0.0, 1.0);
        float gate = (fl & 2u) != 0u ? ga : max(ga, gt);
        score = gate / (1.0 + exp(-(za + zt + ${AGG_NAME}.bias)));
      }
      earthworkScore_v = score;
      color = vec4(score, score, score, 1.0);
    `,
  },
  uniformTypes: { bias: "f32", gateLo: "f32", gateHi: "f32", tLo: "f32", tHi: "f32" },
  getUniforms: (props: Partial<EarthworkAggregateProps>) => ({
    aefFirst: props.aefFirst,
    aefLast: props.aefLast,
    tesFirst: props.tesFirst,
    tesLast: props.tesLast,
    tesFlags: props.tesFlags,
    weightTex: props.weightTex,
    bias: props.bias ?? 0,
    gateLo: props.gateLo ?? 0.05,
    gateHi: props.gateHi ?? 0.15,
    tLo: props.tLo ?? 0,
    tHi: props.tHi ?? 1,
  }),
} as const satisfies ShaderModule<EarthworkAggregateProps>;

/**
 * The paint's mask: after a score module, drops the fragments under the
 * layer's cutoff, so a mask drawn with it is exactly the lit pixels.
 */
export const EarthworkCut = {
  name: "earthworkCut",
  fs: `\
uniform earthworkCutUniforms {
  float cut;
} earthworkCut;
`,
  inject: {
    "fs:DECKGL_FILTER_COLOR": /* glsl */ `
      if (earthworkScore_v < earthworkCut.cut) discard;
    `,
  },
  uniformTypes: { cut: "f32" },
  getUniforms: (props: { cut?: number }) => ({ cut: props.cut ?? 0.5 }),
} as const satisfies ShaderModule<{ cut: number }>;

/**
 * Props for the {@link EarthworkScoreTex} shader module.
 */
export type EarthworkScoreTexProps = {
  /** r8unorm: the score scored ahead (chunk.worker.ts), 1 + 254 x score, 0 where a year is missing. */
  scoreTex: Texture;
};

/**
 * The score read from a texture scored ahead on the CPU (the paint's years), in place of
 * EarthworkScore's embeddings: sets earthworkScore_v for EarthworkCut. Missing pixels are discarded.
 */
export const EarthworkScoreTex = {
  name: "earthworkScoreTex",
  inject: {
    "fs:#decl": `
precision highp sampler2D;
uniform highp sampler2D scoreTex;
float earthworkScore_v;
`,
    "fs:DECKGL_FILTER_COLOR": /* glsl */ `
      ivec2 size = textureSize(scoreTex, 0);
      ivec2 px = clamp(ivec2(geometry.uv * vec2(size)), ivec2(0), size - 1);
      float s = texelFetch(scoreTex, px, 0).r * 255.0;
      if (s < 0.5) discard;
      earthworkScore_v = (s - 1.0) / 254.0;
      color = vec4(earthworkScore_v, earthworkScore_v, earthworkScore_v, 1.0);
    `,
  },
  getUniforms: (props: Partial<EarthworkScoreTexProps>) => ({ scoreTex: props.scoreTex }),
} as const satisfies ShaderModule<EarthworkScoreTexProps>;
