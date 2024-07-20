use std::{borrow::Cow, path::Path};

use wgpu::util::DeviceExt;

const WIDTH: usize = 512;
const HEIGHT: usize = 512;

async fn run<P: AsRef<Path>>(path: P) {
	let image = image::open(path).unwrap().into_rgba8();

	assert_eq!(image.width() as usize, WIDTH);
	assert_eq!(image.height() as usize, HEIGHT);

	let bytes = image.into_raw();

	let steps = execute_gpu(&bytes).await.unwrap();
}

async fn execute_gpu(image: &[u8]) -> Option<Vec<u32>> {
	let instance = wgpu::Instance::default();

	let adapter = instance
		.request_adapter(&wgpu::RequestAdapterOptions::default())
		.await?;

	let (device, queue) = adapter
		.request_device(
			&wgpu::DeviceDescriptor {
				label: None,
				required_features: wgpu::Features::empty(),
				required_limits: wgpu::Limits::downlevel_defaults(),
			},
			None,
		)
		.await
		.unwrap();

	execute_gpu_inner(&device, &queue, image).await
}

async fn execute_gpu_inner(
	device: &wgpu::Device,
	queue: &wgpu::Queue,
	image: &[u8],
) -> Option<Vec<u32>> {
	// Loads the shader from WGSL
	let module = device.create_shader_module(wgpu::include_wgsl!("../shader.wgsl"));

	let size = std::mem::size_of_val(image) as wgpu::BufferAddress;

	let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
		label: None,
		size,
		usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
		mapped_at_creation: false,
	});

	// Stores the target image, and is overwritten with the first solution
	// under a specified threshold
	let storage_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
		label: Some("ellipses buffer"),
		contents: image,
		usage: wgpu::BufferUsages::STORAGE
			| wgpu::BufferUsages::COPY_DST
			| wgpu::BufferUsages::COPY_SRC,
	});

	let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
		label: None,
		layout: None,
		module: &module,
		entry_point: "cs_main",
		compilation_options: Default::default(),
	});

	let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
		label: None,
		layout: None,
		multiview: None,
		vertex: wgpu::VertexState {
			module: &module,
			entry_point: "vs_main",
			buffers: &[],
			compilation_options: Default::default(),
		},
		fragment: Some(wgpu::FragmentState {
			module: &module,
			entry_point: "fs_main",
			targets: &[Some(wgpu::ColorTargetState {
				format: wgpu::TextureFormat::Rgba8UnormSrgb,
				blend: Some(wgpu::BlendState::REPLACE),
				write_mask: wgpu::ColorWrites::ALL,
			})],
			compilation_options: Default::default(),
		}),
		primitive: Default::default(),
		depth_stencil: None,
		multisample: Default::default(),
	});

	let l = render_pipeline.get_bind_group_layout(0);
	println!("{:?}", l);

	let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
		label: Some("Bind Group Layout"),
		entries: &[
			wgpu::BindGroupLayoutEntry {
				binding: 0,
				visibility: wgpu::ShaderStages::VERTEX_FRAGMENT | wgpu::ShaderStages::COMPUTE,
				ty: wgpu::BindingType::Texture {
					sample_type: wgpu::TextureSampleType::Float { filterable: false },
					view_dimension: wgpu::TextureViewDimension::D2,
					multisampled: false,
				},
				count: None,
			},
			wgpu::BindGroupLayoutEntry {
				binding: 1,
				visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
				ty: wgpu::BindingType::StorageTexture {
					access: wgpu::StorageTextureAccess::ReadWrite,
					format: wgpu::TextureFormat::Rgba8UnormSrgb,
					view_dimension: wgpu::TextureViewDimension::D2,
				},
				count: None,
			},
			wgpu::BindGroupLayoutEntry {
				binding: 2,
				visibility: wgpu::ShaderStages::COMPUTE,
				ty: wgpu::BindingType::Texture {
					multisampled: false,
					view_dimension: wgpu::TextureViewDimension::D2,
					sample_type: wgpu::TextureSampleType::Float { filterable: false },
				},
				count: None,
			},
			wgpu::BindGroupLayoutEntry {
				binding: 3,
				visibility: wgpu::ShaderStages::COMPUTE,
				ty: wgpu::BindingType::Buffer {
					ty: wgpu::BufferBindingType::Storage { read_only: false },
					has_dynamic_offset: false,
					min_binding_size: None,
				},
				count: None,
			},
		],
	});

	let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
		layout: &bind_group_layout,
		entries: &[
			wgpu::BindGroupEntry {
				binding: 0,
				resource: storage_buffer.as_entire_binding(),
			}, /* wgpu::BindGroupEntry {
			    * binding: 1,
			    * resource: wgpu::BindingResource::TextureView(&texture_view),
			    * }, */
			wgpu::BindGroupEntry {
				binding: 3,
				resource: staging_buffer.as_entire_binding(),
			},
		],
		label: Some("Bind Group"),
	});

	let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
	{
		let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
			label: None,
			timestamp_writes: None,
		});
		cpass.set_pipeline(&compute_pipeline);
		cpass.set_bind_group(0, &bind_group, &[]);
		cpass.insert_debug_marker("compute collatz iterations");
		cpass.dispatch_workgroups(256, 1, 1);
	}

	// Sets adds copy operation to command encoder.
	// Will copy data from storage buffer on GPU to staging buffer on CPU.
	encoder.copy_buffer_to_buffer(&storage_buffer, 0, &staging_buffer, 0, size);

	// Submits command encoder for processing
	let index = queue.submit(Some(encoder.finish()));

	// Note that we're not calling `.await` here.
	let buffer_slice = staging_buffer.slice(..);
	// Sets the buffer up for mapping, sending over the result of the mapping back
	// to us when it is finished.
	let (sender, receiver) = flume::bounded(1);
	buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

	// Poll the device in a blocking manner so that our future resolves.
	// In an actual application, `device.poll(...)` should
	// be called in an event loop or on another thread.
	device
		.poll(wgpu::Maintain::wait_for(index))
		.panic_on_timeout();

	// Awaits until `buffer_future` can be read from
	if let Ok(Ok(())) = receiver.recv_async().await {
		// Gets contents of buffer
		let data = buffer_slice.get_mapped_range();
		// Since contents are got in bytes, this converts these bytes back to u32
		let result = bytemuck::cast_slice(&data).to_vec();

		// With the current interface, we have to make sure all mapped views are
		// dropped before we unmap the buffer.
		drop(data);
		staging_buffer.unmap(); // Unmaps buffer from memory
													// If you are familiar with C++ these 2 lines can be thought of similarly to:
													//   delete myPointer;
													//   myPointer = NULL;
													// It effectively frees the memory

		// Returns data from buffer
		Some(result)
	} else {
		panic!("failed to run compute on gpu!")
	}
}

fn main() {
	tracing_subscriber::fmt::init();

	pollster::block_on(run("input.jpg"));
}
