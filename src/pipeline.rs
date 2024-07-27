use rand::{Rng, RngCore};
use wgpu::util::DeviceExt;

pub struct Pipeline {
	device: wgpu::Device,
	queue: wgpu::Queue,

	render: wgpu::RenderPipeline,
	compute: wgpu::ComputePipeline,

	current: wgpu::Texture,
	current_view: wgpu::TextureView,
	current_for_compute: wgpu::Texture,

	/// The current best splat configuration that results
	/// in an image closest to the target image.
	splats: Box<[u8]>,
	splat_buffer: wgpu::Buffer,

	/// The current difference between the target image and the
	/// image rendered by the fragment shader.
	difference: wgpu::Buffer,
	difference_staging: wgpu::Buffer,
	lowest_difference: Difference,

	bind_group: wgpu::BindGroup,
	rng: rand::rngs::ThreadRng,

	debug_buffer: wgpu::Buffer,
}

#[derive(Debug, Clone, Copy)]
pub struct Difference(pub u32);

impl Difference {
	pub fn splat(self, rng: &mut rand::rngs::ThreadRng) -> [u8; 8] {
		let mut bytes = [0u8; 8];
		rng.fill_bytes(&mut bytes);
		let stage = self.stage();

		bytes.map(|b| b >> stage)
	}

	/// Returns the "stage", which approaches 7 as the difference
	/// reaches 0, and 0 as it reaches `WIDTH * HEIGHT * 3 * 255`.
	///
	/// It should be approximately logarithmic, but not exactly.
	pub fn stage(self) -> u8 {
		const MAX_INPUT: u32 = Pipeline::WIDTH as u32 * Pipeline::HEIGHT as u32 * 3 * 255;
		const MIN_INPUT: u32 = MAX_INPUT / 64;
		const BREAK_POINT: u32 = MAX_INPUT / 4;

		const MAX_OUTPUT: u8 = 3;
		const MIN_OUTPUT: u8 = 0;

		if self.0 >= BREAK_POINT {
			return 0;
		}

		// Ensure x is within the expected range
		let x = self.0.min(MIN_INPUT);

		// Calculate the logarithmic scale factor
		let scale_factor = (BREAK_POINT - MIN_INPUT) as f64 / (MAX_OUTPUT as f64).ln();

		// Apply the logarithmic transformation
		let log_value = (MAX_OUTPUT as f64 * (1.0 - (x as f64 / scale_factor).ln())).max(0.0);

		// Convert the result to u32
		(log_value.round() as u8 * 3 / 5).clamp(MIN_OUTPUT, MAX_OUTPUT)
	}
}

impl Pipeline {
	pub const HEIGHT: usize = 512;
	pub const WIDTH: usize = 512;

	pub fn difference(&self) -> u32 {
		self.lowest_difference.0
	}

	pub fn solution(&self) -> &[u8] {
		&self.splats
	}

	/// Constructs a pipeline used to target a certain image.
	///
	/// The image must be a [`Self::HEIGHT`] x [`Self::WIDTH`] image in RGBA
	/// format.
	pub async fn new(image: &[u8], splats: Option<[u8; 4000]>) -> Option<Self> {
		let mut rng = rand::thread_rng();
		let splats = if let Some(splats) = splats {
			splats
		} else {
			let mut splats = [0u8; 500 * 8];
			rng.fill_bytes(&mut splats);
			splats
		};

		let (device, queue) = create_device_queue().await.unwrap();

		let module = device.create_shader_module(wgpu::include_wgsl!("../shader.wgsl"));

		let target = create_target_texture(&device, &queue, image);
		let (current, current_for_compute) = create_render_textures(&device, &queue, image);

		let splat_buffer = create_splat_buffer(&device, &splats);

		let (difference, difference_staging) = create_difference_buffer(&device);

		let layout = bind_group_layout(&device);
		let group = bind_group(
			&device,
			&layout,
			&target,
			&splat_buffer,
			&difference,
			&current_for_compute,
		);

		let (render, compute) = create_pipelines(&device, &module, &layout);

		let debug_buffer = device.create_buffer(&wgpu::BufferDescriptor {
			label: None,
			size: std::mem::size_of_val(image) as wgpu::BufferAddress,
			usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
			mapped_at_creation: false,
		});

		Some(Self {
			device,
			queue,
			render,
			compute,
			current_view: current.create_view(&wgpu::TextureViewDescriptor::default()),
			current,
			current_for_compute,
			splats: splats.into(),
			splat_buffer,
			difference,
			difference_staging,
			lowest_difference: Difference(u32::MAX),
			bind_group: group,
			rng,
			debug_buffer,
		})
	}

	/// Renders the current splats to the image texture.
	pub fn render(&self) {
		// first, render to texture
		let mut encoder = self
			.device
			.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

		{
			let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
				label: None,
				timestamp_writes: None,
				color_attachments: &[Some(wgpu::RenderPassColorAttachment {
					view: &self.current_view,
					resolve_target: None,
					ops: wgpu::Operations {
						load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
						store: wgpu::StoreOp::Store,
					},
				})],
				depth_stencil_attachment: None,
				occlusion_query_set: None,
			});

			rpass.set_pipeline(&self.render);
			rpass.set_bind_group(0, &self.bind_group, &[]);
			// draw a quad
			rpass.draw(0..6, 0..1);
		}

		encoder.copy_texture_to_texture(
			wgpu::ImageCopyTexture {
				texture: &self.current,
				mip_level: 0,
				origin: wgpu::Origin3d::ZERO,
				aspect: wgpu::TextureAspect::All,
			},
			wgpu::ImageCopyTexture {
				texture: &self.current_for_compute,
				mip_level: 0,
				origin: wgpu::Origin3d::ZERO,
				aspect: wgpu::TextureAspect::All,
			},
			wgpu::Extent3d {
				width: Self::WIDTH as u32,
				height: Self::HEIGHT as u32,
				depth_or_array_layers: 1,
			},
		);

		// Submits command encoder for processing
		let index = self.queue.submit(Some(encoder.finish()));

		// Poll the device in a blocking manner so that our future resolves.
		// In an actual application, `device.poll(...)` should
		// be called in an event loop or on another thread.
		self
			.device
			.poll(wgpu::Maintain::wait_for(index))
			.panic_on_timeout();
	}

	/// Runs one pass and returns the difference
	pub async fn pass(&mut self) {
		let splat = self.rng.gen_range(0..500);
		let offset = (splat * 8) as wgpu::BufferAddress;

		// generate random bytes. we will use the current difference
		// to determine how much to nudge by (this doesn't "overwrite",
		// but rather "nudges" the current splat)
		let mut splat_nudge = self.lowest_difference.splat(&mut self.rng);
		let random_splat = {
			let base_splat = &self.splats[offset as usize..offset as usize + 8];

			for i in 0..8 {
				splat_nudge[i] = splat_nudge[i].wrapping_add(base_splat[i]);
			}

			splat_nudge
		};

		// write to some splats
		self
			.queue
			.write_buffer(&self.splat_buffer, offset, &random_splat);

		// render the current splats
		self.render();

		// reset the difference counter
		self
			.queue
			.write_buffer(&self.difference, 0, bytemuck::cast_slice(&[0u32]));

		// once it renders, do the compute pass!
		let mut encoder = self
			.device
			.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

		{
			let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
				label: None,
				timestamp_writes: None,
			});

			cpass.set_pipeline(&self.compute);
			cpass.set_bind_group(0, &self.bind_group, &[]);
			cpass.dispatch_workgroups(32, 32, 1);
		}

		encoder.copy_buffer_to_buffer(&self.difference, 0, &self.difference_staging, 0, 4);

		// Submits command encoder for processing
		let index = self.queue.submit(Some(encoder.finish()));

		let buffer_slice = self.difference_staging.slice(..);
		let (sender, receiver) = flume::bounded(1);
		buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

		// Poll the device in a blocking manner so that our future resolves.
		// In an actual application, `device.poll(...)` should
		// be called in an event loop or on another thread.
		self
			.device
			.poll(wgpu::Maintain::wait_for(index))
			.panic_on_timeout();

		// Awaits until `buffer_future` can be read from
		let difference = if let Ok(Ok(())) = receiver.recv_async().await {
			// Gets contents of buffer
			let data = buffer_slice.get_mapped_range();
			// Since contents are got in bytes, this converts these bytes back to u32
			let result: &[u32] = bytemuck::cast_slice(&data);

			let difference = result[0];

			drop(data);
			self.difference_staging.unmap();

			difference
		} else {
			panic!("failed to run compute on gpu!")
		};

		// if the new difference is strictly worse, copy the best splat to the
		// one we just tried
		if difference > self.lowest_difference.0 {
			self.queue.write_buffer(
				&self.splat_buffer,
				offset,
				&self.splats[offset as usize..offset as usize + 8],
			);
		} else {
			// otherwise, we want to update our new best params with the one we just
			// determined was better
			self.splats[offset as usize..offset as usize + 8].copy_from_slice(&random_splat);

			if difference == self.lowest_difference.0 {
				return;
			}

			tracing::info!(
				"improved difference from {} to {}",
				self.lowest_difference.0,
				difference
			);

			self.lowest_difference.0 = difference;

			self.render_to_image().await.save("debug.png").unwrap();
		}
	}

	pub async fn render_to_image(&self) -> image::RgbaImage {
		let mut encoder = self
			.device
			.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

		encoder.copy_texture_to_buffer(
			wgpu::ImageCopyTexture {
				texture: &self.current,
				mip_level: 0,
				origin: wgpu::Origin3d::ZERO,
				aspect: wgpu::TextureAspect::All,
			},
			wgpu::ImageCopyBuffer {
				buffer: &self.debug_buffer,
				layout: wgpu::ImageDataLayout {
					offset: 0,
					bytes_per_row: Some(Self::WIDTH as u32 * 4),
					rows_per_image: Some(Self::HEIGHT as u32),
				},
			},
			wgpu::Extent3d {
				width: Self::WIDTH as u32,
				height: Self::HEIGHT as u32,
				depth_or_array_layers: 1,
			},
		);

		let index = self.queue.submit(Some(encoder.finish()));

		// debug the texture we wrote to to make sure it looks right
		let buffer_slice = self.debug_buffer.slice(..);

		let (sender, receiver) = flume::bounded(1);

		buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

		self
			.device
			.poll(wgpu::Maintain::wait_for(index))
			.panic_on_timeout();

		// Awaits until `buffer_future` can be read from
		if let Ok(Ok(())) = receiver.recv_async().await {
			// Gets contents of buffer
			let data = buffer_slice.get_mapped_range();
			let result: &[u8] = &data;

			let image =
				image::RgbaImage::from_raw(Self::WIDTH as u32, Self::HEIGHT as u32, result.to_vec())
					.unwrap();

			drop(data);
			self.debug_buffer.unmap();

			image
		} else {
			panic!("failed to run fragment on gpu!")
		}
	}
}

fn create_pipelines(
	device: &wgpu::Device,
	module: &wgpu::ShaderModule,
	layout: &wgpu::BindGroupLayout,
) -> (wgpu::RenderPipeline, wgpu::ComputePipeline) {
	let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
		label: None,
		bind_group_layouts: &[layout],
		push_constant_ranges: &[],
	});

	let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
		label: None,
		layout: Some(&pipeline_layout),
		multiview: None,
		vertex: wgpu::VertexState {
			module,
			entry_point: "vs_main",
			buffers: &[],
			compilation_options: Default::default(),
		},
		fragment: Some(wgpu::FragmentState {
			module,
			entry_point: "fs_main",
			targets: &[Some(wgpu::ColorTargetState {
				format: wgpu::TextureFormat::Rgba8Unorm,
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

	let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
		label: None,
		layout: Some(&pipeline_layout),
		module,
		entry_point: "cs_main",
		compilation_options: Default::default(),
		cache: None,
	});

	(render_pipeline, compute_pipeline)
}

fn bind_group(
	device: &wgpu::Device,
	layout: &wgpu::BindGroupLayout,
	target_texture: &wgpu::Texture,
	current_params_buffer: &wgpu::Buffer,
	difference_buffer: &wgpu::Buffer,
	current_texture_for_compute: &wgpu::Texture,
) -> wgpu::BindGroup {
	device.create_bind_group(&wgpu::BindGroupDescriptor {
		layout,
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
					buffer: current_params_buffer,
					offset: 0,
					size: None,
				}),
			},
			// similarity buffer
			wgpu::BindGroupEntry {
				binding: 2,
				resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
					buffer: difference_buffer,
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
	})
}

fn bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
	device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
	})
}

/// Creates the target texture that the pipeline will try to match.
///
/// This will never be modified once created.
fn create_target_texture(
	device: &wgpu::Device,
	queue: &wgpu::Queue,
	image: &[u8],
) -> wgpu::Texture {
	let target_texture_desc = wgpu::TextureDescriptor {
		label: Some("target texture (what we want)"),
		size: wgpu::Extent3d {
			width: Pipeline::WIDTH as u32,
			height: Pipeline::HEIGHT as u32,
			depth_or_array_layers: 1,
		},
		mip_level_count: 1,
		sample_count: 1,
		dimension: wgpu::TextureDimension::D2,
		format: wgpu::TextureFormat::Rgba8Unorm,
		usage: wgpu::TextureUsages::TEXTURE_BINDING,
		view_formats: &[wgpu::TextureFormat::Rgba8Unorm],
	};

	device.create_texture_with_data(
		queue,
		&target_texture_desc,
		wgpu::util::TextureDataOrder::LayerMajor,
		image,
	)
}

/// Creates two textures: one for the fragment shader to draw into, and one for
/// the compute shader to read from (with a `texture_to_texture` copy in
/// between)
fn create_render_textures(
	device: &wgpu::Device,
	queue: &wgpu::Queue,
	image: &[u8],
) -> (wgpu::Texture, wgpu::Texture) {
	let desc = wgpu::TextureDescriptor {
		label: None,
		size: wgpu::Extent3d {
			width: Pipeline::WIDTH as u32,
			height: Pipeline::HEIGHT as u32,
			depth_or_array_layers: 1,
		},
		mip_level_count: 1,
		sample_count: 1,
		dimension: wgpu::TextureDimension::D2,
		format: wgpu::TextureFormat::Rgba8Unorm,
		view_formats: &[wgpu::TextureFormat::Rgba8Unorm],
		usage: wgpu::TextureUsages::empty(),
	};

	let current_texture_desc = wgpu::TextureDescriptor {
		label: Some("current render"),
		usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::RENDER_ATTACHMENT,
		..desc
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
		label: Some("current compute"),
		usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
		..desc
	};

	let current_texture_for_compute = device.create_texture(&current_texture_for_compute_desc);

	(current_texture, current_texture_for_compute)
}

async fn create_device_queue() -> Option<(wgpu::Device, wgpu::Queue)> {
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

	Some((device, queue))
}

fn create_splat_buffer(device: &wgpu::Device, splats: &[u8]) -> wgpu::Buffer {
	device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
		label: Some("current params buffer"),
		contents: splats,
		usage: wgpu::BufferUsages::STORAGE
			| wgpu::BufferUsages::COPY_DST
			| wgpu::BufferUsages::COPY_SRC,
	})
}

/// Constructs two buffers: one for the compute shader to write the difference
/// into, and one for the CPU to read the difference from.
fn create_difference_buffer(device: &wgpu::Device) -> (wgpu::Buffer, wgpu::Buffer) {
	// stores the difference between source and rendered image (written to by
	// compute)
	let difference_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
		label: None,
		contents: bytemuck::cast_slice(&[0u32]),
		usage: wgpu::BufferUsages::COPY_SRC
			| wgpu::BufferUsages::COPY_DST
			| wgpu::BufferUsages::STORAGE,
	});

	let difference_buffer_staging = device.create_buffer(&wgpu::BufferDescriptor {
		label: None,
		size: 4,
		usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
		mapped_at_creation: false,
	});

	(difference_buffer, difference_buffer_staging)
}
