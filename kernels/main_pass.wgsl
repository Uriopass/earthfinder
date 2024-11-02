const STEP_SIZE: i32 = 2;

fn process(args: ProcessArgs) -> vec4<f32> {
    // tex0 is mask
    // tex1 is tile data
    let dims_mask = textureDimensions(tex0);

    var sum = 0.0;
    var total = 0.0;

    for (var y = 0u; y < dims_mask.y; y = y + 3) {
        for (var x = 0u; x < dims_mask.x; x = x + 3) {
            let p = args.pos * STEP_SIZE + vec2<i32>(i32(x), i32(y));
            let mask_value = textureLoad(tex0, vec2(x, y), 0).xy;
            let tile_value = textureLoad(tex1, p, 0).x;

            sum += tile_value * (mask_value.x - mask_value.y * 0.3);
            total += mask_value.x;
        }
    }


    if (sum / total < 0.1) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    sum = 0.0;
    total = 0.0;

    for (var y = 0u; y < dims_mask.y; y = y + 1) {
        for (var x = 0u; x < dims_mask.x; x = x + 1) {
            let p = args.pos * STEP_SIZE + vec2<i32>(i32(x), i32(y));
            let mask_value = textureLoad(tex0, vec2(x, y), 0).xy;
            let tile_value = textureLoad(tex1, p, 0).x;

            sum += tile_value * (mask_value.x - mask_value.y * 0.3);
            total += mask_value.x;
        }
    }

    sum /= total;

    return vec4<f32>(sum, 0.0, 0.0, 0.0);
}