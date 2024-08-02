// Source texture, likely 512x512, that we want to match closely
@group(0) @binding(0) var t_source: texture_2d<f32>;
// A heatmap used to weigh the importance of each pixel, from 1 to 21.
@group(0) @binding(1) var t_heatmap: texture_2d<i32>;

// A bunch of bytes that represent the ellipses we want to draw
// These are just in a big buffer, since they don't correlate to
// the size of the image.
@group(0) @binding(2) var<storage, read_write> ellipses: array<u32>;

const TAU: f32 = 6.283185307179586;

fn gaussian(p: vec2<f32>, c: vec2<f32>, r: vec2<f32>, a: f32) -> f32 {
	var p1 = p - c;

	let an = a * TAU / 256.0;
	let co = cos(an);
	let si = sin(an);
	p1 = mat2x2<f32>(co, -si, si, co) * p1;
	p1 /= r;
	return exp(-4.0 * dot(p1, p1));
}

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> @builtin(position) vec4<f32> {
	var positions = array<vec2<f32>, 6>(
		vec2<f32>(-1.0, -1.0), // bottom left
		vec2<f32>( 1.0, -1.0), // bottom right
		vec2<f32>(-1.0,  1.0), // top left
		vec2<f32>(-1.0,  1.0), // top left
		vec2<f32>( 1.0, -1.0), // bottom right
		vec2<f32>( 1.0,  1.0)  // top right
	);

	let position = positions[in_vertex_index];
	return vec4<f32>(position, 0.0, 1.0);
}

// A fragment shader that draws the ellipses
@fragment
fn fs_main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
	let p = vec2<f32>(position.x, position.y);

	var col: vec3<f32> = vec3<f32>(255.0);

	for (var i = 0u; i < arrayLength(&ellipses) / 2; i++) {
		let whag = ellipses[i * 2];
		let xyrb = ellipses[i * 2 + 1];

		let w = f32(whag & 255);
		let h = f32((whag >> 8) & 255);
		let a = f32((whag >> (8 + 8)) & 255);
		let g = f32((whag >> (8 + 8 + 8)) & 255);
		// 0 to 8
		let x = f32(xyrb & 511);
		// 9 to 17
		let y = f32((xyrb >> 9) & 511);
		// 18 to 24 (with LSB always 0)
		let r = f32((xyrb >> (9 + 9 - 1)) & 254);
		// 25 to 31 (with LSB always 0)
		let b = f32((xyrb >> (9 + 9 + 7 - 1)) & 254);

		let f = gaussian(p, vec2<f32>(x, y), vec2<f32>(w, h), a);

		col = mix(col, vec3<f32>(r, g, b), f);
	}

	return vec4<f32>(col / 255.0, 1.0);
}

@group(0) @binding(3)
var<storage, read_write> similarity: atomic<u32>;

// The current texture we have, same size as the source
@group(0) @binding(4) var t_current: texture_2d<f32>;

// A compute shader that calculates how close `t_current` is to `t_source`
// then randomly modifies part of `ellipses` to try and get closer.
//
// This shader is called once per 16x16 block of pixels in the image.
@compute
@workgroup_size(1, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
	let tl = global_id.xy * 16u;

	// check all the way to global_id.x + 16 and global_id.y + 16
	// it is assumed the image is a multiple of 16x16
	for (var x = tl.x; x < 16 + tl.x; x++) {
		for (var y = tl.y; y < 16 + tl.y; y++) {
			let source = vec4<i32>(textureLoad(t_source, vec2<u32>(x, y), 0) * 255.0);
			let current = vec4<i32>(textureLoad(t_current, vec2<u32>(x, y), 0) * 255.0);

			let diff = abs(source - current) * textureLoad(t_heatmap, vec2<u32>(x, y), 0).r;

			// we don't normalize as 512x512x3x256*WEIGHT still fits within a u32
			atomicAdd(&similarity, u32(diff.r + diff.g + diff.b));
		}
	}
}
