struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    //@location(0) frag_uv: vec2<f32>,
}

struct Parameters {
    total1: f32,
    total2: f32,
}

const PI: f32 = 3.14159265359;

var<push_constant> tile_widths: array<u32, 16>;

@group(0) @binding(0) var<uniform> params: Parameters;

@group(1) @binding(0) var lsampl: sampler;
@group(1) @binding(1) var nsampl: sampler;
@group(1) @binding(2) var tex_mask: texture_2d<f32>; // (mip 1 is proper, mip 2 has tile rgb from last frame)
@group(1) @binding(3) var tex_tile: texture_2d<f32>; // (mip 1 is not populated, mip 2 is proper)

const STEP_SIZE: i32 = 1;

fn evalSum(mask: vec3<f32>, tile: vec2<f32>, filter_around: f32) -> f32 {
    return dot(mask.xy, tile - 0.3 * max(0.1 - tile, vec2(0.0))) - dot(tile, tile) * mask.z * filter_around;
}

fn evalTotal(mask: vec3<f32>) -> f32 {
    return dot(mask.xy, mask.xy);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) u32 {
    let v = process(in);
    return pack2x16float(v);
}

@fragment
fn fs_main_debug(in: VertexOutput) -> @location(0) vec2<f32> {
    return process(in);
}

const DETAILED_SCORE_THRESHOLD: f32 = 0.25;
const AROUND_COEFF_1: f32 = 2.0;
const AROUND_COEFF_2: f32 = 1.0;
const MATCHING_SCORE_COEFF: f32 = 1.0;

const N_ZOOMS: u32 = 4;
var<private> zooms: array<f32, N_ZOOMS> = array<f32, N_ZOOMS>(1.0 / 1.5, 1/1.3333, 1 / 1.1666,  1.0);

const ZERO_ARR: array<f32, N_ZOOMS> = array<f32, N_ZOOMS>(0.0, 0.0, 0.0, 0.0);
const EPS_ARR:  array<f32, N_ZOOMS> = array<f32, N_ZOOMS>(1e-5, 1e-5, 1e-5, 1e-5);

fn process(in: VertexOutput) -> vec2<f32> {
    var pixelpos = vec2<i32>(in.position.xy) * STEP_SIZE;    // 1920x1952 = 4*(512-32)x(512-24)
    let dims_mask: vec2<u32> = textureDimensions(tex_mask);  // = 32x24
    let dims_tile  = vec2<f32>(textureDimensions(tex_tile)); // = 2048x2048

    let inv_dims_mask = vec2<f32>(1.0) / vec2<f32>(dims_mask);

    let dims_mask_extended: vec2<u32> = dims_mask * 3u / 2u;

    let tile_idx: vec2<i32>   = pixelpos / (vec2(512) - vec2<i32>(dims_mask)); // = 0..3
    let tile_local: vec2<i32> = pixelpos % (vec2(512) - vec2<i32>(dims_mask)); // = 0..480 x 0..488
    let tile_width: u32       = tile_widths[u32(tile_idx.x) + u32(tile_idx.y) * 4];


    if  (u32(tile_local.x) >= (tile_width - dims_mask_extended.x * 3u / 2u))
     || (u32(tile_local.y) >= (512u       - dims_mask_extended.y * 3u / 2u)) {
        return vec2(-1000.0, 1.0);
    }

    pixelpos = pixelpos + tile_idx * vec2<i32>(dims_mask);

    var matchingScores = ZERO_ARR;

    let origin_m = vec2<f32>(pixelpos);

    for (var y = 0u; y < dims_mask_extended.y / 4; y = y + 1) {
        for (var x = 0u; x < dims_mask_extended.x / 4; x = x + 1) {
            let off = vec2(i32(x), i32(y));
            let p: vec2<i32> = pixelpos / 4 + off;
            let tile_value = textureLoad(tex_tile, p, 2).xyz;

            for (var zoomI = 0u; zoomI < 10; zoomI++) {
                let zoom = zooms[zoomI];

                let mask_pos = vec2<f32>(off * 4) * inv_dims_mask * zoom;
                if (mask_pos.x >= 1.0 || mask_pos.y >= 1.0) {
                    break;
                }
                let mask_value = textureSampleLevel(tex_mask, lsampl, mask_pos, 2.0).xyz;
                let diff = (mask_value - tile_value);
                matchingScores[zoomI]  += dot(diff, diff);
            }

        }
    }
    for (var zoomI = 0u; zoomI < N_ZOOMS; zoomI++) {
        if (textureLoad(tex_mask, vec2(0), 2).x == 0.0) {
            matchingScores[zoomI] = 0.0;
        }

        matchingScores[zoomI] /= f32(dims_mask.x * dims_mask.y / 16);
        matchingScores[zoomI] = MATCHING_SCORE_COEFF * sqrt(matchingScores[zoomI]);
    }

    var sums   = ZERO_ARR;
    var totals = EPS_ARR;

    for (var y = 0u; y < dims_mask_extended.y / 2; y = y + 1) {
        for (var x = 0u; x < dims_mask_extended.x / 2; x = x + 1) {
            let off = vec2(i32(x * 2), i32(y * 2));
            let p: vec2<i32> = pixelpos + off;
            let tile_value = textureLoad(tex_tile, p, 0).xy;

            for (var zoomI = 0u; zoomI < N_ZOOMS; zoomI++) {
                let zoom = zooms[zoomI];

                let mask_pos = vec2<f32>(off) * inv_dims_mask * zoom;
                if (mask_pos.x >= 1.0 || mask_pos.y >= 1.0) {
                    break;
                }
                let mask_value = textureSampleLevel(tex_mask, lsampl, mask_pos, 1.0).xyz;
                sums[zoomI] += evalSum(mask_value, tile_value, AROUND_COEFF_1);
                totals[zoomI] += evalTotal(mask_value);
            }
        }
    }
    var any_has_detailed = false;

    for (var zoomI = 0u; zoomI < N_ZOOMS; zoomI++) {
        sums[zoomI] = sums[zoomI] / totals[zoomI] - matchingScores[zoomI];
        if (sums[zoomI] > DETAILED_SCORE_THRESHOLD) {
            any_has_detailed = true;
            break;
        }
    }


    if (any_has_detailed) {
        var total2 = 0.00001;
        for (var zoomI = 0u; zoomI < N_ZOOMS; zoomI++) {
            sums[zoomI] = 0.0;
            totals[zoomI] = 0.0;
        }

        for (var y = 0u; y < dims_mask_extended.y; y = y + 1) {
            for (var x = 0u; x < dims_mask_extended.x; x = x + 1) {
                let off = vec2(i32(x), i32(y));
                let p: vec2<i32> = pixelpos + off;
                let tile_value = textureLoad(tex_tile, p, 0).xy;

                for (var zoomI = 0u; zoomI < N_ZOOMS; zoomI++) {
                    let zoom = zooms[zoomI];

                    let mask_pos = vec2<f32>(off) * inv_dims_mask * zoom;
                    if (mask_pos.x >= 1.0 || mask_pos.y >= 1.0) {
                        break;
                    }
                    let mask_value = textureSampleLevel(tex_mask, lsampl, mask_pos, 0.0).xyz;
                    sums[zoomI] += evalSum(mask_value, tile_value, AROUND_COEFF_2);
                    totals[zoomI] += evalTotal(mask_value);
                }
            }
        }


        for (var zoomI = 0u; zoomI < N_ZOOMS; zoomI++) {
            sums[zoomI] = sums[zoomI] / totals[zoomI] - matchingScores[zoomI];
        }
    }

    var best_zoom = 0u;
    var best_score = -1000.0;

    for (var zoomI = 0u; zoomI < N_ZOOMS; zoomI++) {
        if (sums[zoomI] > best_score) {
            best_score = sums[zoomI];
            best_zoom = zoomI;
        }
    }

    return vec2<f32>(best_score, 1.0 / zooms[best_zoom]);
}

@vertex
fn vs_main(@builtin(vertex_index) VertexIndex: u32) -> VertexOutput {
    var pos = vec2(0.0, 0.0);
    switch (VertexIndex) {
        case 0u: {pos = vec2(-1.0, -1.0);}
        case 1u: {pos = vec2(3.0, -1.0);}
        case 2u: {pos = vec2(-1.0, 3.0);}
        default: {}
    }

    //let uv = vec2(pos.x * 0.5 + 0.5, 0.5 - pos.y * 0.5);
    return VertexOutput(vec4(pos, 0.0, 1.0));//, uv);
}
