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
