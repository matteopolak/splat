// Source texture, likely 512x512, that we want to match closely
@group(0) @binding(0) var t_source: texture_2d<f32>;

// The current texture we have, same size as the source
@group(0) @binding(1) var t_current: texture_2d<f32>;

// A bunch of bytes that represent the ellipses we want to draw
// These are just in a big buffer, since they don't correlate to
// the size of the image.
@group(0) @binding(2) var<storage, read_write> ellipses: array<u32>;

const TAU: f32 = 6.283185307179586;

fn gaussian(p: vec2<f32>, c: vec2<f32>, r: vec2<f32>, a: f32) -> f32 {
	p -= c;

	let an = a * (TAU / 256.0);
	let co = cos(an);
	let si = sin(an);
	p = mat2<f32>(co, -si, si, co) * p;
	p /= r;
	exp(-4.0 * dot(p, p))
}

// A fragment shader that draws the ellipses
@fragment
fn fs_main(@builtin(position) position: vec4<f32>) -> vec4<f32> {
	let p = vec2<f32>(position.x, 511.0 - position.y);

	var col: vec3<f32> = vec3<f32>(1.0, 1.0, 1.0);

	for i in 0..500 {
		let whag = ellipses[i * 2];
		let xyrb = ellipses[i * 2 + 1];

		let x = f32(xyrb & 511);
		let y = f32((xyrb >> 9) & 511);
		let w = f32(whag & 255);
		let h = f32((whag >> 8) & 255);
		let a = f32((whag >> 16) & 255);
		let g = u8((whag >> 24) & 255);
		let u = f32((xyrb >> 16) & 508);
		let v = f32((xyrb >> 23) & 508);
		let r = f32(clamp(g as f32 + u - 256.0, 0.0, 255.0));
		let b = f32(clamp(g as f32 + v - 256.0, 0.0, 255.0));

		let f = gaussian(p, vec2<f32>(x, y), vec2<f32>(w, h), a);

		col = mix(col, vec3<f32>(r, f32(g), b), f);
	}

	if abs(p.x - 256.0) > 255.0 {
		col = vec3<f32>(0.0);
	}

	return vec4<f32>(col, 1.0);
}

@group(0) @binding(3)
var<storage, read_write> similarity: atomic<f32> = 0.0;

// A compute shader that calculates how close `t_current` is to `t_source`
// then randomly modifies part of `ellipses` to try and get closer.
//
// The comparison is a regular diff of the rgb channels, then divides by the dimensions
// of the image to get a value between 0 and 1.
//
// This shader is called once per 16x16 block of pixels in the image.

@compute
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
	let dimensions = textureDimensions(t_source);

	// check all the way to global_id.x + 16 and global_id.y + 16
	// it is assumed the image is a multiple of 16x16
	for (var x = global_id.x; x < 16; x++) {
		for (var y = global_id.y; y < 16; y++) {
			let source = textureLoad(t_source, vec2<u32>(x, y));
			let current = textureLoad(t_current, vec2<u32>(x, y));

			let diff = abs(source - current);

			// once it's all done, we want it to be normalized. so divide by the dimensions
			atomicAdd(similarity, f32(diff.r + diff.g + diff.b) / (dimensions.x * dimensions.y));
		}
	}
}
