pub struct Splat {
	// 9 bits
	pub x: u16,
	pub y: u16,
	// 7 bits
	pub red: u8,
	pub blue: u8,
	// 8 bits
	pub width: u8,
	pub height: u8,
	pub angle: u8,
	pub green: u8,
}

impl Splat {
	pub fn from_bytes(bytes: [u8; 8]) -> Self {
		let w = bytes[3];
		let h = bytes[2];
		let a = bytes[1];
		let g = bytes[0];

		let xyrb = u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);

		Splat {
			x: (xyrb & 0b1_1111_1111) as u16,
			y: (xyrb >> 9 & 0b1_1111_1111) as u16,
			red: (xyrb >> 17 & 0b1111_1110) as u8,
			blue: (xyrb >> 24 & 0b1111_1110) as u8,
			width: w,
			height: h,
			angle: a,
			green: g,
		}
	}

	pub fn to_bytes(&self) -> [u8; 8] {
		let mut xyrb = 0u32;
		xyrb |= self.x as u32;
		xyrb |= (self.y as u32) << 9;
		xyrb |= (self.red as u32) << 17;
		xyrb |= (self.blue as u32) << 24;

		[
			self.green,
			self.angle,
			self.height,
			self.width,
			xyrb as u8,
			(xyrb >> 8) as u8,
			(xyrb >> 16) as u8,
			(xyrb >> 24) as u8,
		]
	}
}
