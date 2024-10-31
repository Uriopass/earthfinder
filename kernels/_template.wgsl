struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) frag_uv: vec2<f32>,
}

struct ProcessArgs {
    uv: vec2<f32>,
    pos: vec2<i32>,
    halfpixel_pos: vec2<f32>,
}

struct Parameters {
    _unused: f32,
}

const PI: f32 = 3.14159265359;

@group(0) @binding(0) var<uniform> params: Parameters;

{{tex_bindings}}
// @group(1) @binding({}) var tex{}: texture_2d<u8>;



@vertex
fn vs_main(@builtin(vertex_index) VertexIndex: u32) -> VertexOutput {
    var pos = vec2(0.0, 0.0);
    switch VertexIndex {
        case 0u: {pos = vec2(-1.0, -1.0);}
        case 1u: {pos = vec2(3.0, -1.0);}
        case 2u: {pos = vec2(-1.0, 3.0);}
        default: {}
    }

    let uv = vec2(pos.x * 0.5 + 0.5, 0.5 - pos.y * 0.5);
    return VertexOutput(vec4(pos, 0.0, 1.0), uv);
}

{{process_fn}}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let pixelpos = vec2<i32>(in.position.xy);
    let args = ProcessArgs(in.frag_uv, pixelpos, in.position.xy);
    let output = process(args);
    return output;
}