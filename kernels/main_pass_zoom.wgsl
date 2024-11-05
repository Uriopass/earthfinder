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

const ZOOM: f32 = 1.5;
const DETAILED_SCORE_THRESHOLD: f32 = 0.25;
const AROUND_COEFF_1: f32 = 2.0;
const AROUND_COEFF_2: f32 = 1.0;
const MATCHING_SCORE_COEFF: f32 = 1.0;
const ZOOM_BOOST: f32 = 1.15;

fn process(in: VertexOutput) -> vec2<f32> {
    var pixelpos = vec2<i32>(in.position.xy) * STEP_SIZE;    // 1920x1952 = 4*(512-32)x(512-24)
    let dims_mask: vec2<u32> = textureDimensions(tex_mask);  // = 32x24
    let dims_tile  = vec2<f32>(textureDimensions(tex_tile)); // = 2048x2048

    let tile_idx: vec2<i32>   = pixelpos / (vec2(512) - vec2<i32>(dims_mask)); // = 0..3
    let tile_local: vec2<i32> = pixelpos % (vec2(512) - vec2<i32>(dims_mask)); // = 0..480 x 0..488
    let tile_width: u32       = tile_widths[u32(tile_idx.x) + u32(tile_idx.y) * 4];

    if u32(tile_local.x) >= (tile_width - dims_mask.x) {
        return vec2(-1000.0, 0.0);
    }

    pixelpos = pixelpos + tile_idx * vec2<i32>(dims_mask);

    var matchingScore = 0.0;

    for (var y = 0u; y < dims_mask.y / 4; y = y + 1) {
        for (var x = 0u; x < dims_mask.x / 4; x = x + 1) {
            let mask_value = textureLoad(tex_mask, vec2(x, y), 2).xyz;

            let p = (pixelpos) / 4 + vec2<i32>(i32(x), i32(y));
            let tile_value = textureLoad(tex_tile, p, 2).xyz;

            let diff = (mask_value - tile_value);
            matchingScore += dot(diff, diff);
        }
    }

    if (textureLoad(tex_mask, vec2(0), 2).x == 0.0) {
        matchingScore = 0.0;
    }

    matchingScore /= f32(dims_mask.x * dims_mask.y / 16);
    matchingScore = MATCHING_SCORE_COEFF * sqrt(matchingScore);

    var sum = 0.0;
    var total = 0.00001;

    for (var y = 0u; y < dims_mask.y / 2; y = y + 1) {
        for (var x = 0u; x < dims_mask.x / 2; x = x + 1) {
            let mask_value = textureLoad(tex_mask, vec2(x, y), 1).xyz;

            let p = pixelpos + vec2<i32>(i32(x * 2), i32(y * 2));
            let tile_value = textureLoad(tex_tile, p, 0).xy;

            if (tile_value.x == 1.0) {
                return vec2(-1000.0, 0.0);
            }

            sum += evalSum(mask_value, tile_value, AROUND_COEFF_1);
            total += evalTotal(mask_value);
        }
    }

    sum /= total;
    sum -= matchingScore;

    if (sum > DETAILED_SCORE_THRESHOLD) {
        sum = 0.0;
        total = 0.00001;

        for (var y = 0u; y < dims_mask.y; y = y + 1) {
            for (var x = 0u; x < dims_mask.x; x = x + 1) {
                let mask_value = textureLoad(tex_mask, vec2(x, y), 0).xyz;

                let p = pixelpos + vec2<i32>(i32(x), i32(y));
                let tile_value = textureLoad(tex_tile, p, 0).xy;

                if (tile_value.x == 1.0) {
                    return vec2(-1000.0, 0.0);
                }

                sum += evalSum(mask_value, tile_value, AROUND_COEFF_2);
                total += evalTotal(mask_value);
            }
        }

        sum /= total;
        sum -= matchingScore;
    }

    if  (u32(tile_local.x) >= (tile_width - dims_mask.x * 3u / 2u))
     || (u32(tile_local.y) >= (512u       - dims_mask.y * 3u / 2u)) {
        return vec2(sum, 1.0);
    }


    matchingScore = 0.0;
    let origin_m = vec2<f32>(pixelpos);

    for (var y = 0u; y < dims_mask.y / 4; y = y + 1) {
        for (var x = 0u; x < dims_mask.x / 4; x = x + 1) {
            let mask_value = textureLoad(tex_mask, vec2(x, y), 2).xyz;

            let off = vec2<f32>(f32(x) * ZOOM, f32(y) * ZOOM);
            let p: vec2<f32> = (origin_m + off * 4) / (dims_tile);
            let tile_value = textureSampleLevel(tex_tile, lsampl, p, 2.0).xyz;

            let diff = (mask_value - tile_value);
            matchingScore += dot(diff, diff);
        }
    }

    if (textureLoad(tex_mask, vec2(0), 2).x == 0.0) {
        matchingScore = 0.0;
    }

    matchingScore /= f32(dims_mask.x * dims_mask.y / 16);
    matchingScore = MATCHING_SCORE_COEFF * sqrt(matchingScore);

    var sum2 = 0.0;
    var total2 = 0.00001;

    let origin = vec2<f32>(pixelpos);


    for (var y = 0u; y < dims_mask.y / 2; y = y + 1) {
        for (var x = 0u; x < dims_mask.x / 2; x = x + 1) {
            let mask_value = textureLoad(tex_mask, vec2(x, y), 1).xyz;

            let off = vec2<f32>(f32(x) * ZOOM * 2.0, f32(y) * ZOOM * 2.0);
            let p: vec2<f32> = (origin + off) / dims_tile;
            let tile_value = textureSampleLevel(tex_tile, lsampl, p, 0.0).xy;

            if (tile_value.x == 1.0) {
                return vec2(sum, 1.0);
            }

            sum2 += evalSum(mask_value, tile_value, AROUND_COEFF_1);
            total2 += evalTotal(mask_value);
        }
    }

    sum2 /= total2;
    sum2 -= matchingScore;

    if (sum2 > DETAILED_SCORE_THRESHOLD) {
        sum2 = 0.0;
        total2 = 0.00001;

        for (var y = 0u; y < dims_mask.y; y = y + 1) {
            for (var x = 0u; x < dims_mask.x; x = x + 1) {
                let mask_value = textureLoad(tex_mask, vec2(x, y), 0).xyz;

                let off = vec2<f32>(f32(x) * ZOOM, f32(y) * ZOOM);
                let p: vec2<f32> = (origin + off) / dims_tile;
                let tile_value = textureSampleLevel(tex_tile, lsampl, p, 0.0).xy;

                if (tile_value.x == 1.0) {
                    return vec2(sum, 1.0);
                }


                sum2 += evalSum(mask_value, tile_value, AROUND_COEFF_2);
                total2 += evalTotal(mask_value);
            }
        }

        sum2 /= total2;
        sum2 -= matchingScore;
        sum2 *= ZOOM_BOOST;
    }

    if (sum2 > sum) {
        return vec2(sum2, 1.5);
    }
    return vec2(sum, 1.0);
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
