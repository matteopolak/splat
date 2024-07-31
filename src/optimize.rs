use std::{cmp::Ordering, ops::RangeInclusive};

use rand::{rngs::ThreadRng, Rng};

use crate::{model::Splat, pipeline::Pipeline};

/// The current stage of the optimizer. This number is entirely
/// based on how far the current image is to the target image.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd)]
pub struct Stage(u8);

impl Stage {
	pub fn into_inner(self) -> u8 {
		self.0
	}

	pub fn finetune_range(self) -> Option<(RangeInclusive<i16>, RangeInclusive<i8>)> {
		if self.0 < 3 {
			return None;
		}

		Some(match self.0 {
			3 => (-64..=64, -64..=64),
			4 => (-32..=32, -24..=24),
			_ => (-16..=16, -16..=16),
		})
	}

	/// The difference starts at around 70 million, quickly decreases
	/// to around 40 million, then ends at around 3-5 million.
	///
	/// This stage takes into account the above values and returns
	/// the current stage, from 0 (the start) to 4 (the end).
	pub fn from_difference(difference: u32) -> Self {
		// TODO: figure out good values for this
		match difference {
			..10_000_000 => Self(4),
			10_000_000..20_000_000 => Self(3),
			20_000_000..25_000_000 => Self(2),
			25_000_000..30_000_000 => Self(1),
			30_000_000.. => Self(0),
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
	pub fn from_image(image: image::RgbaImage) -> Self {
		let heatmap = HeatMap::from_image(&image);
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

	#[inline]
	pub fn stage(&self) -> Stage {
		self.stage
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
		let mut s = Splat::from_bytes(*splat);

		if let Some((position, attrib)) = self.stage.finetune_range() {
			// move around the position up to 64 pixels
			let dx = rng.gen_range(position.clone());
			let dy = rng.gen_range(position.clone());

			s.x = s
				.x
				.saturating_add_signed(dx)
				.min(Pipeline::WIDTH as u16 - 1);
			s.y = s
				.y
				.saturating_add_signed(dy)
				.min(Pipeline::HEIGHT as u16 - 1);

			let dw = rng.gen_range(attrib.clone());
			let dh = rng.gen_range(attrib.clone());

			s.width = s.width.saturating_add_signed(dw);
			s.height = s.height.saturating_add_signed(dh);

			let da = rng.gen_range(attrib);

			s.angle = s.angle.saturating_add_signed(da);

			// re-calculate the colour
			let [red, green, blue] = self.colour_at(s.x as u32, s.y as u32);

			s.red = red & 0b1111_1110;
			s.blue = blue & 0b1111_1110;
			s.green = green;
		} else {
			let (x, y) = self.heatmap.weighted_pixel(rng);
			let [red, green, blue] = self.colour_at(x, y);

			s.x = x as u16;
			s.y = y as u16;
			s.red = red & 0b1111_1110;
			s.blue = blue & 0b1111_1110;
			s.green = green;

			s.angle = rng.gen();
			s.width = rng.gen();
			s.height = rng.gen();
		}

		*splat = s.to_bytes();
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

		if stage > self.stage {
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

		self.lowest_difference = difference;

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
	pub weights: Vec<u8>,
	/// `weights.iter().sum()`
	pub weights_sum: u32,
	/// The image that the heatmap was generated from.
	pub dimensions: (u32, u32),
	/// The number of unique colours in the image (reduced to 255 colours).
	pub num_colours: u8,
}

impl HeatMap {
	/// Randomly selects a pixel from the heatmap, with a higher chance of
	/// selecting a pixel with a higher complexity.
	pub fn weighted_pixel(&self, rng: &mut ThreadRng) -> (u32, u32) {
		let mut value = rng.gen_range(0..self.weights_sum);
		let mut index = 0;

		for (i, &complexity) in self.weights.iter().enumerate() {
			if value < complexity as u32 {
				index = i;
				break;
			}

			value -= complexity as u32;
		}

		let x = index as u32 % self.dimensions.0;
		let y = index as u32 / self.dimensions.0;

		(x, y)
	}

	/// Determines a reasonable splat count based on the average complexity of the
	/// heatmap.
	///
	/// This takes into account the min and max complexity, and the average
	/// complexity.
	///
	/// A decent number is around 256 for a lower bound, with an upper bound
	/// of 2048.
	pub fn recommended_splat_count(&self) -> usize {
		let average = self.weights_sum as f64 / self.weights.len() as f64;

		let lower = ((average - 32.0) / 32.0 * 256.0).round();
		let upper = ((average - 32.0) / 32.0 * 2048.0).round();

		let base = (lower + upper) / 2.0;

		// incorporate the number of colours
		let colour_factor = self.num_colours as f64 / 32.0;

		((base * colour_factor) as usize).clamp(256, 2048)
	}

	/// Constructs a heatmap image from the heatmap.
	/// with red indicating high complexity and blue indicating low complexity.
	pub fn to_image(&self) -> image::RgbImage {
		let (width, height) = self.dimensions;
		let mut image = image::RgbImage::new(width, height);

		for (i, &complexity) in self.weights.iter().enumerate() {
			let x = i as u32 % self.dimensions.0;
			let y = i as u32 / self.dimensions.0;

			let value = (complexity as f64 / u8::MAX as f64 * 255.0) as u8;
			let pixel = image::Rgb([value, 0, 255 - value]);

			image.put_pixel(x, y, pixel);
		}

		image
	}

	pub fn from_image(image: &image::RgbaImage) -> Self {
		let radius = 32;
		let mut heatmap = vec![0; image.width() as usize * image.height() as usize];

		let mut min = u32::MAX;
		let mut max = u32::MIN;

		let mut map_sum = 0u32;

		let mut colour_count = vec![0u8; 256];

		for y in 0..image.height() {
			for x in 0..image.width() {
				let mut complexity = 0;
				let mut weight = 0;

				let current_pixel = image.get_pixel(x, y);
				let current_pixel = [current_pixel[0], current_pixel[1], current_pixel[2]];

				// squash the pixel to an 8-bit colour
				let eight_bit = current_pixel[0] as u32 * 6 / 256 * 36
					+ current_pixel[1] as u32 * 6 / 256 * 6
					+ current_pixel[2] as u32 * 6 / 256;

				// increment colour
				let count = &mut colour_count[eight_bit as usize];
				*count = count.saturating_add(1);

				let ry = y.saturating_sub(radius)..=y.saturating_add(radius).min(image.height() - 1);
				let rx = x.saturating_sub(radius)..=x.saturating_add(radius).min(image.width() - 1);

				for ny in ry {
					for nx in rx.clone() {
						let pixel = image.get_pixel(nx, ny);
						let pixel = [pixel[0], pixel[1], pixel[2]];

						let diff = [
							pixel[0].abs_diff(current_pixel[0]) as u32,
							pixel[1].abs_diff(current_pixel[1]) as u32,
							pixel[2].abs_diff(current_pixel[2]) as u32,
						];

						let diff = diff.iter().sum::<u32>();

						complexity += diff;
						weight += 1;
					}
				}

				let value = complexity / weight;

				min = min.min(value);
				max = max.max(value);
				heatmap[(y * image.width() + x) as usize] = value;
			}
		}

		Self {
			weights: heatmap
				.into_iter()
				.map(|value| (value - min) as f64 / (max - min) as f64 * u8::MAX as f64)
				.map(|value| value.clamp(32.0, 64.0))
				.map(|value| {
					map_sum += value as u32;
					value as u8
				})
				.collect(),
			weights_sum: map_sum,
			dimensions: image.dimensions(),
			// only count it as a colour if there are at least 128 pixels of it
			num_colours: colour_count.iter().filter(|&&c| c > 128).count() as u8,
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
	pub const START_HEIGHT: usize = 64;
	pub const START_WIDTH: usize = 64;
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
		self.current.save("rescaled.png").unwrap();
	}
}
