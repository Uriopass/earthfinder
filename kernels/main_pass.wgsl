const STEP_SIZE: i32 = 1;

fn evalSum(mask: vec3<f32>, tile: vec2<f32>, filter_around: f32) -> f32 {
    return dot(mask.xy, tile) - dot(tile, tile) * mask.z * filter_around - 0.15* dot(mask.xy, max(0.1 - tile, vec2(0.0)));
}

fn evalTotal(mask: vec3<f32>) -> f32 {
    return dot(mask.xy, mask.xy);
}

fn process(args: ProcessArgs) -> vec4<f32> {
    // tex0 is mask
    // tex1 is tile data
    // tex2 is last frame
    let dims_mask = textureDimensions(tex0);

    var sum = 0.0;
    var total = 0.0;

    for (var y = 0u; y < dims_mask.y / 2; y = y + 1) {
        for (var x = 0u; x < dims_mask.x / 2; x = x + 1) {
            let mask_value = textureLoad(tex0, vec2(x, y), 1).xyz;

            let p = args.pos * STEP_SIZE + vec2<i32>(i32(x * 2), i32(y * 2));
            let tile_value = textureLoad(tex1, p, 0).xy;

            if (tile_value.x == 1.0) {
                return vec4<f32>(-100.0, 0.0, 0.0, 0.0);
            }

            sum += evalSum(mask_value, tile_value, 3.0);
            total += evalTotal(mask_value);
        }
    }

    if (sum / total < 0.1) {
        return vec4<f32>(sum / total, 0.0, 0.0, 0.0);
    }
    sum = 0.0;
    total = 0.0;

    for (var y = 0u; y < dims_mask.y; y = y + 1) {
        for (var x = 0u; x < dims_mask.x; x = x + 1) {
            let mask_value = textureLoad(tex0, vec2(x, y), 0).xyz;

            let p = args.pos * STEP_SIZE + vec2<i32>(i32(x), i32(y));
            let tile_value = textureLoad(tex1, p, 0).xy;

            if (tile_value.x == 1.0) {
                return vec4<f32>(-100.0, 0.0, 0.0, 0.0);
            }

            sum += evalSum(mask_value, tile_value, 1.5);
            total += evalTotal(mask_value);
        }
    }

    sum /= total;

    return vec4<f32>(sum, 0.0, 0.0, 0.0);
}