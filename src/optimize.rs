use std::cmp::Ordering;

use rand::{rngs::ThreadRng, Rng};

use crate::{model::Splat, pipeline::Pipeline};

/// The current stage of the optimizer. This number is entirely
/// based on how far the current image is to the target image.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Stage(u8);

impl Stage {
	pub fn into_inner(self) -> u8 {
		self.0
	}

	/// Once this returns `true`, the optimizer should stop moving
	/// everything around randomly and instead nudge existing
	/// parameters.
	pub fn should_finetune(self) -> bool {
		self.0 >= 3
	}

	/// The difference starts at around 70 million, quickly decreases
	/// to around 40 million, then ends at around 3-5 million.
	///
	/// This stage takes into account the above values and returns
	/// the current stage, from 0 (the start) to 5 (the end).
	pub fn from_difference(difference: u32) -> Self {
		// TODO: figure out good values for this
		match difference {
			..5_000_000 => Self(5),
			5_000_000..10_000_000 => Self(4),
			10_000_000..17_000_000 => Self(3),
			17_000_000..30_000_000 => Self(2),
			30_000_000..40_000_000 => Self(1),
			40_000_000.. => Self(0),
		}
	}
}

/// The optimizer is responsible for optimizing the splat placement.
#[derive(Debug)]
pub struct Optimizer {
	heatmap: HeatMap,
	scaled: ScaledImage,
	stage: Stage,
	lowest_difference: u32,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Change {
	Improved,
	ImprovedAndStageChanged,
	Stagnant,
	Degraded,
}

impl Change {
	pub fn is_improved(&self) -> bool {
		matches!(self, Self::Improved | Self::ImprovedAndStageChanged)
	}

	pub fn is_stagnant(&self) -> bool {
		matches!(self, Self::Stagnant)
	}

	pub fn stage_changed(&self) -> bool {
		matches!(self, Self::ImprovedAndStageChanged)
	}
}

impl Optimizer {
	pub fn from_image(image: image::RgbaImage, radius: u32) -> Self {
		let heatmap = HeatMap::from_image(&image, radius);
		let scaled = ScaledImage::from_image(image);
		let stage = Stage::default();

		Self {
			heatmap,
			scaled,
			stage,
			lowest_difference: u32::MAX,
		}
	}

	#[inline]
	pub fn difference(&self) -> u32 {
		self.lowest_difference
	}

	/// Mutates a splat using information from the optimizer.
	///
	/// 88887799
	///
	/// In order from left to right:
	/// 32                        0
	/// green, angle, height, width
	/// 32            0
	/// blue, red, y, x
	pub fn transform_splat(&self, rng: &mut ThreadRng, splat: &mut [u8; 8]) {
		if self.stage.should_finetune() {
			let mut s = Splat::from_bytes(*splat);

			// move around the position up to 64 pixels
			let dx = rng.gen_range(-64..=64);
			let dy = rng.gen_range(-64..=64);

			s.x = s
				.x
				.saturating_add_signed(dx)
				.min(Pipeline::WIDTH as u16 - 1);
			s.y = s
				.y
				.saturating_add_signed(dy)
				.min(Pipeline::HEIGHT as u16 - 1);

			// change the width and height up to 16 pixels
			let dw = rng.gen_range(-16..=16);
			let dh = rng.gen_range(-16..=16);

			s.width = s.width.saturating_add_signed(dw);
			s.height = s.height.saturating_add_signed(dh);

			// change the angle up to 32 degrees
			let da = rng.gen_range(-32..=32);

			s.angle = s.angle.saturating_add_signed(da);

			// re-calculate the colour
			let [red, green, blue] = self.colour_at(s.x as u32, s.y as u32);

			s.red = red & 0b1111_1110;
			s.blue = blue & 0b1111_1110;
			s.green = green;

			*splat = s.to_bytes();
		} else {
			let (x, y) = self.heatmap.weighted_pixel(rng);
			let [red, green, blue] = self.colour_at(x, y);

			splat[1..4].copy_from_slice(&rng.gen::<[u8; 3]>());
			splat[0] = green;

			let xyrb = (blue as u32) << 24 | (red as u32) << 17 | y << 9 | x;

			splat[4..8].copy_from_slice(&xyrb.to_be_bytes());
		}
	}

	/// Iterates over every pixel, calling `colour_at` for each one
	/// and writing it out.
	pub fn debug_flush(&self) {
		let mut image = image::RgbaImage::new(Pipeline::WIDTH as u32, Pipeline::HEIGHT as u32);

		for y in 0..Pipeline::HEIGHT as u32 {
			for x in 0..Pipeline::WIDTH as u32 {
				let colour = self.colour_at(x, y);
				let pixel = image::Rgba([colour[0], colour[1], colour[2], 255]);

				image.put_pixel(x, y, pixel);
			}
		}

		image.save("flush.png").unwrap();
	}

	pub fn colour_at(&self, x: u32, y: u32) -> [u8; 3] {
		let rgba = self.scaled.current_image().get_pixel(x, y);
		let red = rgba[0] & 0b1111_1110;
		let green = rgba[1];
		let blue = rgba[2] & 0b1111_1110;

		[red, green, blue]
	}

	pub fn heatmap(&self) -> &HeatMap {
		&self.heatmap
	}

	pub fn current(&self) -> &image::RgbaImage {
		self.scaled.current_image()
	}

	/// The original image.
	pub fn original(&self) -> &image::RgbaImage {
		&self.scaled.original
	}

	pub fn original_bytes(&self) -> &[u8] {
		self.scaled.original.as_raw()
	}

	/// Update the difference. This is assumed to always be smaller
	/// than the previous time it was called.
	///
	/// Returns `true` if the difference was updated (i.e. it was better or the
	/// same).
	pub fn update_difference(&mut self, difference: u32) -> Change {
		let ordering = difference.cmp(&self.lowest_difference);

		if ordering == Ordering::Greater {
			return Change::Degraded;
		}

		let stage = Stage::from_difference(difference);

		if stage == self.stage {
			self.lowest_difference = difference;
		} else {
			tracing::info!(
				"Stage changed from {} to {}",
				self.stage.into_inner(),
				stage.into_inner()
			);

			self.stage = stage;
			// Rescale the image
			self.scaled.rescale();
			self.lowest_difference = u32::MAX;

			return Change::ImprovedAndStageChanged;
		}

		if ordering == Ordering::Equal {
			Change::Stagnant
		} else {
			Change::Improved
		}
	}
}

/// A map that represents the complexity of an image at each pixel.
///
/// The complexity of a pixel is the weighted sum of the differences
/// between the pixel and some radius of its neighbours.
#[derive(Debug)]
pub struct HeatMap {
	/// The complexity of each pixel, divided by the number of pixels in the
	/// radius, then normalized to the entire range of u16.
	pub weights: Vec<u16>,
	/// `weights.iter().sum()`
	pub weights_sum: u64,
	/// The image that the heatmap was generated from.
	pub dimensions: (u32, u32),
}

impl HeatMap {
	/// Randomly selects a pixel from the heatmap, with a higher chance of
	/// selecting a pixel with a higher complexity.
	pub fn weighted_pixel(&self, rng: &mut ThreadRng) -> (u32, u32) {
		let mut value = rng.gen_range(0..self.weights_sum);
		let mut index = 0;

		for (i, &complexity) in self.weights.iter().enumerate() {
			if value < complexity as u64 {
				index = i;
				break;
			}

			value -= complexity as u64;
		}

		let x = index as u32 % self.dimensions.0;
		let y = index as u32 / self.dimensions.0;

		(x, y)
	}

	/// Constructs a heatmap image from the heatmap.
	/// with red indicating high complexity and blue indicating low complexity.
	pub fn to_image(&self) -> image::RgbImage {
		let (width, height) = self.dimensions;
		let mut heatmap = image::RgbImage::new(width, height);

		for y in 0..heatmap.height() {
			for x in 0..heatmap.width() {
				let complexity = self.weights[(y * heatmap.width() + x) as usize] as f64 / u16::MAX as f64;

				let color = image::Rgb([
					((1.0 - complexity).powf(2.0) * 255.0) as u8,
					((1.0 - complexity).powf(1.0) * 255.0) as u8,
					((1.0 - complexity).powf(0.5) * 255.0) as u8,
				]);

				heatmap.put_pixel(x, y, color);
			}
		}

		heatmap
	}

	pub fn from_image(image: &image::RgbaImage, radius: u32) -> Self {
		let mut heatmap = vec![0.0; image.width() as usize * image.height() as usize];

		let mut min = f64::INFINITY;
		let mut max = f64::NEG_INFINITY;

		let mut map_sum = 0u64;

		for y in 0..image.height() {
			for x in 0..image.width() {
				let mut complexity = 0.0;
				let mut weight = 0u32;

				let ry = y.saturating_sub(radius)..=y.saturating_add(radius).min(image.height() - 1);
				let rx = x.saturating_sub(radius)..=x.saturating_add(radius).min(image.width() - 1);

				for ny in ry {
					for nx in rx.clone() {
						let pixel = image.get_pixel(nx, ny);
						let pixel = [
							pixel[0] as f64 / 255.0,
							pixel[1] as f64 / 255.0,
							pixel[2] as f64 / 255.0,
						];

						let current_pixel = image.get_pixel(x, y);
						let current_pixel = [
							current_pixel[0] as f64 / 255.0,
							current_pixel[1] as f64 / 255.0,
							current_pixel[2] as f64 / 255.0,
						];

						let diff = [
							(pixel[0] - current_pixel[0]).abs(),
							(pixel[1] - current_pixel[1]).abs(),
							(pixel[2] - current_pixel[2]).abs(),
						];

						let diff = diff.iter().sum::<f64>();

						complexity += diff;
						weight += 1;
					}
				}

				let value = complexity / weight as f64;

				min = min.min(value);
				max = max.max(value);
				heatmap[(y * image.width() + x) as usize] = value;
			}
		}

		Self {
			weights: heatmap
				.into_iter()
				.map(|value| (value - min) / (max - min) * u16::MAX as f64)
				.map(|value| {
					map_sum += value as u64;
					value as u16
				})
				.collect(),
			weights_sum: map_sum,
			dimensions: image.dimensions(),
		}
	}
}

/// A representation of a rescaled image, starting at a low resolution,
/// and gradually increasing the resolution to guide the training process.
#[derive(Debug)]
pub struct ScaledImage {
	/// The image data. 512x512
	pub original: image::RgbaImage,
	/// The current scale of the image, starting at 16x16 and increasing to
	/// 512x512.
	pub scale: (u32, u32),
	/// The resized image (downsized then upsized) of the original image
	/// according to the `current_scale` value.
	pub current: image::RgbaImage,
}

impl ScaledImage {
	pub const HEIGHT: usize = Pipeline::HEIGHT;
	pub const START_HEIGHT: usize = 16;
	pub const START_WIDTH: usize = 16;
	pub const WIDTH: usize = Pipeline::WIDTH;

	pub fn from_image(image: image::RgbaImage) -> Self {
		let mut scaled = Self {
			original: image,
			// Start at 16/2 since we immediately double the scale
			scale: (Self::START_WIDTH as u32 / 2, Self::START_HEIGHT as u32 / 2),
			current: image::RgbaImage::new(0, 0),
		};

		scaled.rescale();
		scaled
	}

	/// Get the current image.
	pub fn current_image(&self) -> &image::RgbaImage {
		&self.current
	}

	/// Rescale the image to the next scale.
	pub fn rescale(&mut self) {
		if self.scale.0 == Self::WIDTH as u32 || self.scale.1 == Self::HEIGHT as u32 {
			return;
		}

		self.scale = (self.scale.0 * 2, self.scale.1 * 2);
		self.current = image::imageops::resize(
			&self.original,
			self.scale.0,
			self.scale.0,
			image::imageops::FilterType::Lanczos3,
		);

		// Then scale back up to 512x512
		self.current = image::imageops::resize(
			&self.current,
			Self::WIDTH as u32,
			Self::HEIGHT as u32,
			image::imageops::FilterType::Lanczos3,
		);

		self
			.current
			.save(format!("rescaled{}.png", self.scale.0))
			.unwrap();
	}
}
