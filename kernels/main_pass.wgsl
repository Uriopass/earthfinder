const STEP_SIZE: i32 = 2;

fn process(args: ProcessArgs) -> vec4<f32> {
    // tex0 is mask
    // tex1 is tile data
    let dims_mask = textureDimensions(tex0);

    var sum1 = vec3(0.0);
    var sum1sq = vec3(0.0);
    var sum2 = vec3(0.0);
    var sum2sq = vec3(0.0);

    for (var y = 0u; y < dims_mask.y; y = y + 1) {
        for (var x = 0u; x < dims_mask.x; x = x + 1) {
            let p = args.pos * STEP_SIZE + vec2<i32>(i32(x), i32(y));
            let mask_value = textureLoad(tex0, vec2(x, y), 0).xy;
            let tile_value = textureLoad(tex1, p, 0).rgb;

            sum1 += tile_value * mask_value.x;
            sum1sq += tile_value * tile_value * mask_value.x;

            sum2 += tile_value * mask_value.y;
            sum2sq += tile_value * tile_value * mask_value.y;
        }
    }

    sum1 /= params.total1;
    sum1sq /= params.total1;
    sum2 /= params.total2;
    sum2sq /= params.total2;

    let var1 = sum1sq - sum1 * sum1;
    let var2 = sum2sq - sum2 * sum2;

    let d = (abs(sum1.x - sum2.x) + abs(sum1.y - sum2.y) + abs(sum1.z - sum2.z)) / (0.5 + length(var1) + length(var2));
    return vec4<f32>(d, 0.0, 0.0, 1.0);
}