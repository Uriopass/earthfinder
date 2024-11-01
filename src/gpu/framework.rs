#![allow(dead_code)]

use std::collections::hash_map::Entry;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

use lazy_static::lazy_static;
use rustc_hash::FxHashMap;
use wgpu::{
    Buffer, CommandEncoder, ComputePipeline, Device, ErrorFilter, Features, RenderPipeline,
    TextureUsages,
};

#[derive(Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct PtrHash<T: ?Sized + 'static>(pub &'static T);

impl<T: ?Sized + 'static> Clone for PtrHash<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: ?Sized + 'static> Copy for PtrHash<T> {}

impl<T: ?Sized> Hash for PtrHash<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (self.0 as *const T).hash(state);
    }
}
pub struct PassEncoder<'a> {
    device: &'a Device,
    encoder: &'a mut CommandEncoder,
    uni_bg: &'a wgpu::BindGroup,
}

impl<'a> PassEncoder<'a> {
    pub fn new(
        device: &'a Device,
        encoder: &'a mut CommandEncoder,
        uni_bg: &'a wgpu::BindGroup,
    ) -> Self {
        Self {
            device,
            encoder,
            uni_bg,
        }
    }

    pub fn pass(
        &mut self,
        kernel_name: &'static str,
        out_tex: GPUTextureRef,
        in_texs: &[GPUTextureRef],
    ) {
        let pipe = get_pipeline(
            self.device,
            kernel_name,
            out_tex.format,
            in_texs.len() as u32,
        );
        do_pass(
            self.device,
            self.encoder,
            pipe,
            out_tex,
            in_texs,
            self.uni_bg,
        );
    }

    pub fn compute_pass(
        &mut self,
        kernel_name: &'static str,
        dispatch_x: u32,
        dispatch_y: u32,
        in_texs: &[GPUTexture],
        rw_bufs: &[&Buffer],
    ) {
        let pipe = get_compute_pipeline(self.device, kernel_name, in_texs, rw_bufs);
        do_compute_pass(
            self.device,
            self.encoder,
            pipe,
            dispatch_x,
            dispatch_y,
            in_texs,
            rw_bufs,
        );
    }
}

pub fn mk_buffer_src(device: &Device, size: u32) -> Arc<Buffer> {
    Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: size as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    }))
}

pub fn mk_buffer_dst(device: &Device, size: u32) -> Arc<Buffer> {
    Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: size as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }))
}

pub type GPUTexture = &'static GPUTextureInner;
pub type GPUTextureRef<'a> = &'a GPUTextureInner;

pub struct GPUTextureInner {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub format: wgpu::TextureFormat,
}

pub fn mk_tex_general(
    device: &Device,
    (width, height): (u32, u32),
    format: wgpu::TextureFormat,
    layers: u32,
) -> GPUTexture {
    let storage_bindable = format
        .guaranteed_format_features(Features::empty())
        .allowed_usages
        .contains(TextureUsages::STORAGE_BINDING);

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: layers,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: TextureUsages::TEXTURE_BINDING
            | TextureUsages::COPY_DST
            | TextureUsages::COPY_SRC
            | TextureUsages::RENDER_ATTACHMENT
            | if storage_bindable {
                TextureUsages::STORAGE_BINDING
            } else {
                TextureUsages::empty()
            },
        view_formats: &[],
    });

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    Box::leak(Box::new(GPUTextureInner {
        texture,
        view,
        format,
    }))
}

pub fn mk_tex_f32(device: &Device, (width, height): (u32, u32)) -> GPUTexture {
    mk_tex_general(device, (width, height), wgpu::TextureFormat::R32Float, 1)
}

pub fn mk_tex(device: &Device, (width, height): (u32, u32)) -> GPUTexture {
    mk_tex_general(device, (width, height), wgpu::TextureFormat::Rgba8Unorm, 1)
}

pub fn mk_tex_layers(device: &Device, (width, height): (u32, u32), layers: u32) -> GPUTexture {
    mk_tex_general(
        device,
        (width, height),
        wgpu::TextureFormat::Rgba8Unorm,
        layers,
    )
}

#[derive(Default)]
pub struct PipelineCache {
    kernel_update: FxHashMap<PtrHash<str>, bool>,
    render_cache: FxHashMap<u64, &'static RenderPipeline>,
    compute_cache: FxHashMap<u64, &'static ComputePipeline>,
}

fn hash<O: Hash>(o: O) -> u64 {
    use rustc_hash::FxHasher;

    let mut s = FxHasher::default();
    o.hash(&mut s);
    s.finish()
}

lazy_static! {
    pub static ref PIPELINE_CACHE: Mutex<PipelineCache> = Mutex::new(PipelineCache::default());
}

fn get_pipeline(
    device: &Device,
    kernel_name: &'static str,
    out_format: wgpu::TextureFormat,
    n_inputs: u32,
) -> &'static RenderPipeline {
    let cache = &mut PIPELINE_CACHE.lock().unwrap();

    let kernel_should_update = *cache
        .kernel_update
        .entry(PtrHash(kernel_name))
        .or_insert(false);

    let hash = hash((kernel_name, out_format, n_inputs));

    let do_pipeline = |verbose| {
        mk_pipeline(device, kernel_name, out_format, n_inputs, verbose)
            .map(|v| Box::leak(Box::new(v)))
    };

    match cache.render_cache.entry(hash) {
        Entry::Occupied(mut o) => {
            if kernel_should_update {
                if let Some(pipe) = do_pipeline(false) {
                    *o.get_mut() = pipe;
                }
            }
            o.get()
        }
        Entry::Vacant(v) => v.insert(do_pipeline(true).unwrap()),
    }
}

fn get_compute_pipeline(
    device: &Device,
    kernel_name: &'static str,
    in_texs: &[GPUTextureRef],
    rw_bufs: &[&Buffer],
) -> &'static ComputePipeline {
    let cache = &mut PIPELINE_CACHE.lock().unwrap();

    let kernel_should_update = *cache
        .kernel_update
        .entry(PtrHash(kernel_name))
        .or_insert(false);

    let hash = hash(kernel_name);

    let do_pipeline = || {
        mk_compute_pipeline(device, kernel_name, in_texs, rw_bufs).map(|v| Box::leak(Box::new(v)))
    };

    match cache.compute_cache.entry(hash) {
        Entry::Occupied(mut o) => {
            if kernel_should_update {
                if let Some(pipe) = do_pipeline() {
                    *o.get_mut() = pipe;
                }
            }
            o.get()
        }
        Entry::Vacant(v) => v.insert(do_pipeline().unwrap()),
    }
}

fn get_shader_source(kernel_name: &str) -> String {
    std::fs::read_to_string(format!("kernels/{}.wgsl", kernel_name)).expect("Kernel not found")
}

fn mk_compute_pipeline(
    device: &Device,
    kernel_name: &'static str,
    in_texs: &[GPUTextureRef],
    rw_bufs: &[&Buffer],
) -> Option<ComputePipeline> {
    let bglayout = mk_bglayout_compute(device, in_texs, rw_bufs.len() as u32);

    let source = get_shader_source(kernel_name);
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Shader"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });

    Some(
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Pipeline Layout"),
                    bind_group_layouts: &[&bglayout],
                    push_constant_ranges: &[],
                }),
            ),
            module: &shader,
            entry_point: "main",
            compilation_options: Default::default(),
        }),
    )
}

// Define the shader template with macros in Rust
const SHADER_TEMPLATE: &str = include_str!("../../kernels/_template.wgsl");

fn mk_pipeline(
    device: &Device,
    kernel_name: &str,
    out_format: wgpu::TextureFormat,
    n_inputs: u32,
    verbose: bool,
) -> Option<RenderPipeline> {
    let mut tex_bindings = String::new();

    for i in 0..n_inputs {
        tex_bindings += &format!("@group(1) @binding({}) var tex{}: texture_2d<f32>;\n", i, i);
    }

    let process_fn = get_shader_source(kernel_name);

    let shader_source = SHADER_TEMPLATE
        .replace("{{process_fn}}", &process_fn)
        .replace("{{tex_bindings}}", &tex_bindings);

    device.push_error_scope(ErrorFilter::Validation);
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(kernel_name),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });
    let err = pollster::block_on(device.pop_error_scope());
    if let Some(err) = err {
        if verbose {
            eprintln!("error compiling kernel {}: {}", kernel_name, err);
        }
        return None;
    }

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Pipeline Layout"),
        bind_group_layouts: &[
            &device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("User Data Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                }],
            }),
            &mk_bglayout(device, n_inputs),
        ],
        push_constant_ranges: &[],
    });

    Some(
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: out_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        }),
    )
}

fn mk_bglayout_compute(
    device: &Device,
    in_texs: &[GPUTextureRef],
    n_rw_bufs: u32,
) -> wgpu::BindGroupLayout {
    let mut entries = vec![];

    for &tex in in_texs {
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: entries.len() as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                multisampled: false,
                view_dimension: wgpu::TextureViewDimension::D2,
                sample_type: tex
                    .format
                    .sample_type(None, Some(Features::FLOAT32_FILTERABLE))
                    .unwrap(),
            },
            count: None,
        });
    }

    for _ in 0..n_rw_bufs {
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: entries.len() as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
    }

    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Bind Group Layout"),
        entries: &entries,
    })
}

fn mk_bg_compute(
    device: &Device,
    layout: &wgpu::BindGroupLayout,
    texs: &[GPUTextureRef],
    rw_bufs: &[&Buffer],
) -> wgpu::BindGroup {
    let mut entries = vec![];

    for &tex in texs {
        entries.push(wgpu::BindGroupEntry {
            binding: entries.len() as u32,
            resource: wgpu::BindingResource::TextureView(&tex.view),
        });
    }

    for buf in rw_bufs {
        entries.push(wgpu::BindGroupEntry {
            binding: entries.len() as u32,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: buf,
                offset: 0,
                size: None,
            }),
        });
    }

    device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout,
        entries: &entries,
        label: Some("Compute Bind Group"),
    })
}

// Function to create a bind group layout
fn mk_bglayout(device: &Device, n_inputs: u32) -> wgpu::BindGroupLayout {
    let mut entries = vec![];

    for _ in 0..n_inputs {
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: entries.len() as u32,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                multisampled: false,
                view_dimension: wgpu::TextureViewDimension::D2,
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
            },
            count: None,
        });
    }

    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Bind Group Layout"),
        entries: &entries,
    })
}

// Function to create a bind group
fn mk_bg(
    device: &Device,
    layout: &wgpu::BindGroupLayout,
    texs: &[GPUTextureRef],
) -> wgpu::BindGroup {
    let mut entries = vec![];

    for &tex in texs {
        entries.push(wgpu::BindGroupEntry {
            binding: entries.len() as u32,
            resource: wgpu::BindingResource::TextureView(&tex.view),
        });
    }

    device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout,
        entries: &entries,
        label: Some("Bind Group"),
    })
}

// Function to execute a render pass
fn do_pass(
    device: &Device,
    command_encoder: &mut CommandEncoder,
    pipeline: &RenderPipeline,
    out_tex: GPUTextureRef,
    in_texs: &[GPUTextureRef],
    uni_bg: &wgpu::BindGroup,
) {
    let bind_group = mk_bg(device, &pipeline.get_bind_group_layout(1), in_texs);

    {
        let mut render_pass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &out_tex.view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        render_pass.set_pipeline(pipeline);
        render_pass.set_bind_group(0, uni_bg, &[]);
        render_pass.set_bind_group(1, &bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }
}

fn do_compute_pass(
    device: &Device,
    encoder: &mut CommandEncoder,
    pipe: &ComputePipeline,
    dispatch_x: u32,
    dispatch_y: u32,
    in_texs: &[GPUTextureRef],
    rw_bufs: &[&Buffer],
) {
    let bind_group = mk_bg_compute(device, &pipe.get_bind_group_layout(0), in_texs, rw_bufs);

    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("Compute Pass"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipe);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
}
