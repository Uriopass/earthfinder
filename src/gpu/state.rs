use bytemuck::Pod;
use std::sync::{Arc, Mutex};

use wgpu::util::DeviceExt;
use wgpu::{
    Backends, Device, DeviceDescriptor, Features, Instance, InstanceDescriptor, Limits,
    PowerPreference, Queue, RequestAdapterOptions,
};

pub struct WGPUState<U> {
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub user_data: Arc<Mutex<(U, wgpu::Buffer)>>,
    pub uni_bg: Arc<wgpu::BindGroup>,
}

impl<U: Pod + Default> WGPUState<U> {
    pub async fn new() -> Self {
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::PRIMARY,
            flags: wgpu::InstanceFlags::ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER,
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        println!("{:?}", adapter.get_info());

        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: None,
                    required_features: Features::FLOAT32_FILTERABLE,
                    required_limits: Limits::default(),
                },
                None,
            )
            .await
            .unwrap();

        let user_data = U::default();

        let uni_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("User Data Uniform Buffer"),
            contents: bytemuck::cast_slice(std::slice::from_ref(&user_data)),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let uni_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uni_buf.as_entire_binding(),
            }],
            label: Some("User Data Bind Group"),
        });

        Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            user_data: Arc::new(Mutex::new((user_data, uni_buf))),
            uni_bg: Arc::new(uni_bg),
        }
    }

    pub fn modify_user_data(queue: &Queue, ud: &Mutex<(U, wgpu::Buffer)>, f: &dyn Fn(&mut U)) {
        let mut ud = ud.lock().unwrap();
        f(&mut ud.0);
        queue.write_buffer(&ud.1, 0, bytemuck::cast_slice(std::slice::from_ref(&ud.0)));
    }
}
