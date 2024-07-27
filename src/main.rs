use std::{borrow::Cow, path::Path};

use rand::{Rng, RngCore};
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

async fn execute_gpu(image: &[u8]) -> Option<[u8; 4000]> {
	let instance = wgpu::Instance::default();

	let adapter = instance
		.request_adapter(&wgpu::RequestAdapterOptions::default())
		.await?;

	let (device, queue) = adapter
		.request_device(
			&wgpu::DeviceDescriptor {
				label: None,
				required_features: wgpu::Features::VERTEX_WRITABLE_STORAGE,
				required_limits: wgpu::Limits::downlevel_defaults(),
				memory_hints: wgpu::MemoryHints::default(),
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
) -> Option<[u8; 4000]> {
	// Loads the shader from WGSL
	let module = device.create_shader_module(wgpu::include_wgsl!("../shader.wgsl"));

	let size = std::mem::size_of_val(image) as wgpu::BufferAddress;

	let target_texture_desc = wgpu::TextureDescriptor {
		label: Some("target texture (what we want)"),
		size: wgpu::Extent3d {
			width: WIDTH as u32,
			height: HEIGHT as u32,
			depth_or_array_layers: 1,
		},
		mip_level_count: 1,
		sample_count: 1,
		dimension: wgpu::TextureDimension::D2,
		format: wgpu::TextureFormat::Rgba8UnormSrgb,
		usage: wgpu::TextureUsages::TEXTURE_BINDING,
		view_formats: &[wgpu::TextureFormat::Rgba8UnormSrgb],
	};

	let target_texture = device.create_texture_with_data(
		queue,
		&target_texture_desc,
		wgpu::util::TextureDataOrder::LayerMajor,
		image,
	);

	// 500 splats, 8 bytes each
	let mut splats = [0u8; 500 * 8];

	// randomize splats
	let mut rng = rand::thread_rng();

	rng.fill_bytes(&mut splats);

	let current_texture_desc = wgpu::TextureDescriptor {
		label: Some("current texture (what we have)"),
		size: wgpu::Extent3d {
			width: WIDTH as u32,
			height: HEIGHT as u32,
			depth_or_array_layers: 1,
		},
		mip_level_count: 1,
		sample_count: 1,
		dimension: wgpu::TextureDimension::D2,
		format: wgpu::TextureFormat::Rgba8UnormSrgb,
		usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::RENDER_ATTACHMENT,
		view_formats: &[wgpu::TextureFormat::Rgba8UnormSrgb],
	};

	// fragment shader draws into this, which is read by compute shader
	// filled with the real image, which will be overwritten before the first check
	let current_texture = device.create_texture_with_data(
		queue,
		&current_texture_desc,
		wgpu::util::TextureDataOrder::LayerMajor,
		image,
	);

	let current_texture_for_compute_desc = wgpu::TextureDescriptor {
		label: Some("current texture for compute"),
		size: wgpu::Extent3d {
			width: WIDTH as u32,
			height: HEIGHT as u32,
			depth_or_array_layers: 1,
		},
		mip_level_count: 1,
		sample_count: 1,
		dimension: wgpu::TextureDimension::D2,
		format: wgpu::TextureFormat::Rgba8UnormSrgb,
		usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
		view_formats: &[wgpu::TextureFormat::Rgba8UnormSrgb],
	};

	let current_texture_for_compute = device.create_texture(&current_texture_for_compute_desc);

	// stores the current parameters being tested by the compute shader
	let current_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
		label: Some("current params buffer"),
		contents: &splats,
		usage: wgpu::BufferUsages::STORAGE
			| wgpu::BufferUsages::COPY_DST
			| wgpu::BufferUsages::COPY_SRC,
	});

	// stores the difference between source and rendered image (written to by
	// compute)
	let difference_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
		label: None,
		contents: bytemuck::cast_slice(&[0u32]),
		usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE,
	});

	let difference_buffer_staging = device.create_buffer(&wgpu::BufferDescriptor {
		label: None,
		size: 4,
		usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
		mapped_at_creation: false,
	});

	let mut best_params = splats;

	let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
		label: Some("Bind Group Layout"),
		entries: &[
			// source texture
			wgpu::BindGroupLayoutEntry {
				binding: 0,
				visibility: wgpu::ShaderStages::COMPUTE,
				ty: wgpu::BindingType::Texture {
					sample_type: wgpu::TextureSampleType::Float { filterable: false },
					view_dimension: wgpu::TextureViewDimension::D2,
					multisampled: false,
				},
				count: None,
			},
			// current parameters
			wgpu::BindGroupLayoutEntry {
				binding: 1,
				visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::VERTEX_FRAGMENT,
				ty: wgpu::BindingType::Buffer {
					ty: wgpu::BufferBindingType::Storage { read_only: false },
					min_binding_size: None,
					has_dynamic_offset: false,
				},
				count: None,
			},
			// similarity buffer
			wgpu::BindGroupLayoutEntry {
				binding: 2,
				visibility: wgpu::ShaderStages::COMPUTE,
				ty: wgpu::BindingType::Buffer {
					ty: wgpu::BufferBindingType::Storage { read_only: false },
					min_binding_size: None,
					has_dynamic_offset: false,
				},
				count: None,
			},
			// current texture (the one being rendered to)
			wgpu::BindGroupLayoutEntry {
				binding: 3,
				visibility: wgpu::ShaderStages::COMPUTE,
				ty: wgpu::BindingType::Texture {
					sample_type: wgpu::TextureSampleType::Float { filterable: false },
					view_dimension: wgpu::TextureViewDimension::D2,
					multisampled: false,
				},
				count: None,
			},
		],
	});

	let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
		label: None,
		bind_group_layouts: &[&bind_group_layout],
		push_constant_ranges: &[],
	});

	let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
		label: None,
		layout: Some(&pipeline_layout),
		module: &module,
		entry_point: "cs_main",
		compilation_options: Default::default(),
		cache: None,
	});

	let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
		label: None,
		layout: Some(&pipeline_layout),
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
		cache: None,
	});

	let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
		layout: &bind_group_layout,
		entries: &[
			// source texture
			wgpu::BindGroupEntry {
				binding: 0,
				resource: wgpu::BindingResource::TextureView(
					&target_texture.create_view(&wgpu::TextureViewDescriptor::default()),
				),
			},
			// current parameters
			wgpu::BindGroupEntry {
				binding: 1,
				resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
					buffer: &current_params_buffer,
					offset: 0,
					size: None,
				}),
			},
			// similarity buffer
			wgpu::BindGroupEntry {
				binding: 2,
				resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
					buffer: &difference_buffer,
					offset: 0,
					size: None,
				}),
			},
			// current texture
			wgpu::BindGroupEntry {
				binding: 3,
				resource: wgpu::BindingResource::TextureView(
					&current_texture_for_compute.create_view(&wgpu::TextureViewDescriptor::default()),
				),
			},
		],
		label: Some("Bind Group"),
	});

	// order of operations:
	// 1. render parameters to texture (fragment)
	// 2. compare texture to source texture (compute)
	// 3. load "similarity" buffer to CPU, if its worse than the last one, copy our
	//    "best" on CPU to overwrite the "current" on the CPU
	// 4. set "similarity" buffer to 0
	// 5. repeat until similarity is lower enough

	let mut similarity = u32::MAX;

	let current_texture_view = current_texture.create_view(&wgpu::TextureViewDescriptor::default());

	// buffer to copy texture to it for debugging
	let debug_buffer = device.create_buffer(&wgpu::BufferDescriptor {
		label: None,
		size,
		usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
		mapped_at_creation: false,
	});

	#[allow(clippy::never_loop)]
	while similarity > 5000000 {
		println!("similarity: {}", similarity);

		let splat = rng.gen_range(0..500);
		let offset = (splat * 8) as wgpu::BufferAddress;

		let random_splat = rng.gen::<[u8; 8]>();

		// write to some splats
		queue.write_buffer(&current_params_buffer, offset, &random_splat);

		// first, render to texture
		let mut encoder =
			device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

		{
			let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
				label: None,
				timestamp_writes: None,
				color_attachments: &[Some(wgpu::RenderPassColorAttachment {
					view: &current_texture_view,
					resolve_target: None,
					ops: wgpu::Operations {
						load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
						store: wgpu::StoreOp::Store,
					},
				})],
				depth_stencil_attachment: None,
				occlusion_query_set: None,
			});

			rpass.set_pipeline(&render_pipeline);
			rpass.set_bind_group(0, &bind_group, &[]);
			// draw a quad
			rpass.draw(0..6, 0..1);
		}

		encoder.copy_texture_to_buffer(
			wgpu::ImageCopyTexture {
				texture: &current_texture,
				mip_level: 0,
				origin: wgpu::Origin3d::ZERO,
				aspect: wgpu::TextureAspect::All,
			},
			wgpu::ImageCopyBuffer {
				buffer: &debug_buffer,
				layout: wgpu::ImageDataLayout {
					offset: 0,
					bytes_per_row: Some(WIDTH as u32 * 4),
					rows_per_image: Some(HEIGHT as u32),
				},
			},
			wgpu::Extent3d {
				width: WIDTH as u32,
				height: HEIGHT as u32,
				depth_or_array_layers: 1,
			},
		);

		encoder.copy_texture_to_texture(
			wgpu::ImageCopyTexture {
				texture: &current_texture,
				mip_level: 0,
				origin: wgpu::Origin3d::ZERO,
				aspect: wgpu::TextureAspect::All,
			},
			wgpu::ImageCopyTexture {
				texture: &current_texture_for_compute,
				mip_level: 0,
				origin: wgpu::Origin3d::ZERO,
				aspect: wgpu::TextureAspect::All,
			},
			wgpu::Extent3d {
				width: WIDTH as u32,
				height: HEIGHT as u32,
				depth_or_array_layers: 1,
			},
		);

		// Submits command encoder for processing
		let index = queue.submit(Some(encoder.finish()));

		// debug the texture we wrote to to make sure it looks right
		let buffer_slice = debug_buffer.slice(..);

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
			let result: Cow<[u8]> = bytemuck::cast_slice(&data).into();

			// write to debug.png
			image::save_buffer(
				"debug.png",
				&result,
				WIDTH as u32,
				HEIGHT as u32,
				image::ColorType::Rgba8,
			)
			.unwrap();

			drop(data);
			debug_buffer.unmap();
		} else {
			panic!("failed to run fragment on gpu!")
		}

		panic!("end");

		// once it renders, do the compute pass!
		let mut encoder =
			device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

		{
			let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
				label: None,
				timestamp_writes: None,
			});

			cpass.set_pipeline(&compute_pipeline);
			cpass.set_bind_group(0, &bind_group, &[]);
			cpass.dispatch_workgroups(1, 1, 1);
		}

		encoder.copy_buffer_to_buffer(&difference_buffer, 0, &difference_buffer_staging, 0, 4);

		// Submits command encoder for processing
		let index = queue.submit(Some(encoder.finish()));

		let buffer_slice = difference_buffer_staging.slice(..);
		let (sender, receiver) = flume::bounded(1);
		buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

		// Poll the device in a blocking manner so that our future resolves.
		// In an actual application, `device.poll(...)` should
		// be called in an event loop or on another thread.
		device
			.poll(wgpu::Maintain::wait_for(index))
			.panic_on_timeout();

		// read the similarity buffer

		let last_similarity = similarity;

		// Awaits until `buffer_future` can be read from
		if let Ok(Ok(())) = receiver.recv_async().await {
			// Gets contents of buffer
			let data = buffer_slice.get_mapped_range();
			// Since contents are got in bytes, this converts these bytes back to u32
			let result: &[u32] = bytemuck::cast_slice(&data);

			similarity = result[0];

			drop(data);
			difference_buffer_staging.unmap();
		} else {
			panic!("failed to run compute on gpu!")
		}

		// if the new similarity is strictly worse, copy the best splat to the
		// one we just tried
		if similarity > last_similarity {
			queue.write_buffer(
				&current_params_buffer,
				offset,
				&best_params[offset as usize..offset as usize + 8],
			);

			similarity = last_similarity;
		} else {
			println!("improved {} -> {}", last_similarity, similarity);
			// otherwise, we want to update our new best params with the one we just
			// determined was better
			best_params[offset as usize..offset as usize + 8].copy_from_slice(&random_splat);
		}
	}

	Some(best_params)
}

fn main() {
	tracing_subscriber::fmt::init();

	pollster::block_on(run("input.jpg"));
}
