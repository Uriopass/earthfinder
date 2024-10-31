fn process(args: ProcessArgs) -> vec4<f32> {
    // tex0 is mask
    // tex1 is tile data
    let dims_mask = textureDimensions(tex0);

    var sum1 = vec3(0.0);
    var total1 = 0.0;

    var sum2 = vec3(0.0);
    var total2 = 0.0;

    for (var y = 0u; y < dims_mask.y; y = y + 1) {
        for (var x = 0u; x < dims_mask.x; x = x + 1) {
            let p = args.pos + vec2<i32>(i32(x), i32(y));
            let mask_value = textureLoad(tex0, vec2(x, y), 0).xy;
            let tile_value = textureLoad(tex1, p, 0).rgb;

            sum1 += tile_value * mask_value.x;
            total1 += mask_value.x;

            sum2 += tile_value * mask_value.y;
            total2 += mask_value.y;
        }
    }

    sum1 /= total1;
    sum2 /= total2;

    let d = (abs(sum1.x - sum2.x) + abs(sum1.y - sum2.y) + abs(sum1.z - sum2.z));
    return vec4<f32>(d, d, d, 1.0);
}