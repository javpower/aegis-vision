//! 确定性随机源（xorshift64*）：合成数据生成与实验种子控制用，无第三方依赖。

pub struct XorShift {
    s: u64,
}

impl XorShift {
    pub fn new(seed: u64) -> Self {
        Self { s: seed.max(1) }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.s;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.s = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// [0, 1) 均匀分布。
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// [lo, hi) 均匀分布。
    pub fn next_range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_f32()
    }

    /// [0, n) 整数。
    pub fn next_usize(&mut self, n: usize) -> usize {
        (self.next_u64() % (n.max(1) as u64)) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_sequence() {
        let mut a = XorShift::new(42);
        let mut b = XorShift::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn ranges_respect_bounds() {
        let mut r = XorShift::new(7);
        for _ in 0..1000 {
            let v = r.next_range(1.5, 2.5);
            assert!((1.5..2.5).contains(&v));
            assert!(r.next_usize(3) < 3);
        }
    }
}
