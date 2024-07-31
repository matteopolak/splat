#[derive(Debug, Clone, Copy)]
pub struct Splat {
	// 9 bits
	pub x: u16,
	pub y: u16,
	// 7 bits
	pub red: u8,
	// 8 bits
	pub green: u8,
	// 7 bits
	pub blue: u8,
	// 8 bits
	pub width: u8,
	pub height: u8,
	pub angle: u8,
}

impl Splat {
	pub fn from_bytes(bytes: [u8; 8]) -> Self {
		let w = bytes[0];
		let h = bytes[1];
		let a = bytes[2];
		let g = bytes[3];

		let xyrb = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);

		Splat {
			x: (xyrb & 0b1_1111_1111) as u16,
			y: (xyrb >> 9 & 0b1_1111_1111) as u16,
			red: (xyrb >> (9 + 9 - 1) & 0b1111_1110) as u8,
			blue: (xyrb >> (9 + 9 + 7 - 1) & 0b1111_1110) as u8,
			width: w,
			height: h,
			angle: a,
			green: g,
		}
	}

	pub fn to_bytes(self) -> [u8; 8] {
		let mut xyrb = 0u32;
		xyrb |= self.x as u32;
		xyrb |= (self.y as u32) << 9;
		xyrb |= (self.red as u32) << (9 + 9 - 1);
		xyrb |= (self.blue as u32) << (9 + 9 + 7 - 1);

		[
			self.width,
			self.height,
			self.angle,
			self.green,
			xyrb as u8,
			(xyrb >> 8) as u8,
			(xyrb >> 16) as u8,
			(xyrb >> 24) as u8,
		]
	}
}

#[cfg(test)]
mod test {
	use rand::Rng;

	use super::*;

	#[test]
	fn test_roundtrip() {
		let mut rng = rand::thread_rng();

		let bytes = rng.gen();
		let splat = Splat::from_bytes(bytes);

		assert_eq!(splat.to_bytes(), bytes);
	}
}
