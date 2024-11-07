use crate::data;
use image::RgbaImage;
use std::ops::{Deref, DerefMut};

pub struct Mask {
    pub img: RgbaImage,
}

impl Mask {
    pub fn new(mask_idx: u32) -> Self {
        let img = data::mask_i(mask_idx);

        Self { img }
    }

    pub fn dot(&self, b: &Mask) -> f32 {
        let mut sum_exact = 0u64;
        let mut norm1_exact = 0u64;
        let mut norm2_exact = 0u64;

        for (p_a, p_b) in self.pixels().zip(b.pixels()) {
            let p_a_0 = p_a.0[0] as u64;
            let p_a_1 = p_a.0[1] as u64;

            let p_b_0 = p_b.0[0] as u64;
            let p_b_1 = p_b.0[1] as u64;

            sum_exact += p_a_0 * p_b_0 + p_a_1 * p_b_1;
            norm1_exact += p_a_0 * p_a_0 + p_a_1 * p_a_1;
            norm2_exact += p_b_0 * p_b_0 + p_b_1 * p_b_1;
        }

        let norm1 = (norm1_exact as f64).sqrt();
        let norm2 = (norm2_exact as f64).sqrt();

        ((sum_exact as f64) / (norm1 * norm2)) as f32
    }
}

impl Deref for Mask {
    type Target = RgbaImage;

    fn deref(&self) -> &Self::Target {
        &self.img
    }
}

impl DerefMut for Mask {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.img
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_lol() {
        let mask1 = super::Mask::new(2349);
        let mask2 = super::Mask::new(2350);

        println!("dot self: {}", mask1.dot(&mask1));
        println!("dot lol: {}", mask1.dot(&mask2));
    }

    #[test]
    fn test_mask_dot() {
        fastrand::seed(0);

        let _ = std::fs::remove_dir_all("data/debug");
        let _ = std::fs::create_dir_all("data/debug");

        for i in 0..1000 {
            let r1 = fastrand::u32(1..6100);
            let dr = fastrand::u32(1..60);

            let mask1 = super::Mask::new(r1);
            let mask2 = super::Mask::new(r1 + dr);

            let diff = mask1.dot(&mask2);
            if diff > 0.8 {
                continue;
            }

            let mut image_both = image::RgbaImage::new(mask1.width() * 2, mask1.height());

            for (x, y, p) in mask1.enumerate_pixels() {
                image_both.put_pixel(x, y, *p);
            }

            for (x, y, p) in mask2.enumerate_pixels() {
                image_both.put_pixel(x + mask1.width(), y, *p);
            }

            image_both
                .save(format!("data/debug/mask_{}.png", i))
                .unwrap();
        }
    }
}
