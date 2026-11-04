use super::VM;
use crate::LustInt;
use crate::bytecode::Value;
use rand::TryRng;
use rand::rngs::SysRng;

/// Portable xoshiro256** state used by the stdlib random functions.
///
/// The algorithm and SplitMix64 seed expansion are kept local so explicit
/// seeds have stable output independent of `rand`'s `StdRng` policy.
#[derive(Clone, Debug)]
pub(super) struct PortableRng {
    state: [u64; 4],
}

impl PortableRng {
    fn from_seed(seed: u64) -> Self {
        let mut splitmix = seed;
        let mut state = [0; 4];
        for word in &mut state {
            splitmix = splitmix.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut value = splitmix;
            value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            *word = value ^ (value >> 31);
        }
        if state.iter().all(|word| *word == 0) {
            state[0] = 1;
        }
        Self { state }
    }

    fn from_state(state: [u64; 4]) -> Result<Self, String> {
        if state.iter().all(|word| *word == 0) {
            return Err("host entropy produced an invalid all-zero RNG seed".to_string());
        }
        Ok(Self { state })
    }

    fn from_entropy() -> Result<Self, String> {
        let mut bytes = [0; 32];
        SysRng
            .try_fill_bytes(&mut bytes)
            .map_err(|error| format!("failed to obtain random seed from host entropy: {error}"))?;
        let mut state = [0; 4];
        for (word, bytes) in state.iter_mut().zip(bytes.chunks_exact(8)) {
            *word = u64::from_le_bytes(bytes.try_into().expect("8-byte seed word"));
        }
        Self::from_state(state)
    }

    fn next_u64(&mut self) -> u64 {
        let [s0, s1, s2, s3] = &mut self.state;
        let product = s1.wrapping_mul(5);
        let result = product.rotate_left(7).wrapping_mul(9);
        let t = *s1 << 17;
        *s2 ^= *s0;
        *s3 ^= *s1;
        *s1 ^= *s2;
        *s0 ^= *s3;
        *s2 ^= t;
        *s3 = s3.rotate_left(45);
        result
    }

    fn random_float(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1_u64 << 53) as f64
    }

    fn random_int_inclusive(&mut self, low: LustInt, high: LustInt) -> LustInt {
        let width = (high as u64).wrapping_sub(low as u64).wrapping_add(1);
        if width == 0 {
            return self.next_u64() as LustInt;
        }

        // Rejection sampling avoids modulo bias while preserving the inclusive
        // interval semantics of math.random(min, max).
        let threshold = width.wrapping_neg() % width;
        loop {
            let value = self.next_u64();
            if value >= threshold {
                return low.wrapping_add((value % width) as LustInt);
            }
        }
    }
}

impl VM {
    /// Set the script-visible random stream to an explicit deterministic seed.
    /// This is reproducible, but does not make the stream cryptographically safe.
    pub fn set_random_seed(&mut self, seed: u64) {
        self.random_rng = Some(PortableRng::from_seed(seed));
    }

    pub(crate) fn random_value(
        &mut self,
        lower: Option<&Value>,
        upper: Option<&Value>,
    ) -> Result<Value, String> {
        if self.random_rng.is_none() {
            self.random_rng = Some(PortableRng::from_entropy()?);
        }
        let rng = self.random_rng.as_mut().expect("RNG was initialized");

        match (lower, upper) {
            (None, _) => Ok(Value::Float(rng.random_float())),
            (Some(max), None) => {
                let hi = coerce_int(max).unwrap_or(1);
                let upper_bound = if hi < 1 { 1 } else { hi };
                Ok(Value::Int(rng.random_int_inclusive(1, upper_bound)))
            }
            (Some(min), Some(max)) => {
                let lo = coerce_int(min).unwrap_or(1);
                let hi = coerce_int(max).unwrap_or(lo);
                let (start, end) = if lo <= hi { (lo, hi) } else { (hi, lo) };
                Ok(Value::Int(rng.random_int_inclusive(start, end)))
            }
        }
    }
}

fn coerce_int(value: &Value) -> Option<LustInt> {
    match value {
        Value::Int(value) => Some(*value),
        Value::Float(value) => Some(*value as LustInt),
        Value::Bool(value) => Some(if *value { 1 } else { 0 }),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::PortableRng;

    #[test]
    fn seed_zero_matches_portable_golden_vector() {
        let mut rng = PortableRng::from_seed(0);
        assert_eq!(
            [
                rng.next_u64(),
                rng.next_u64(),
                rng.next_u64(),
                rng.next_u64()
            ],
            [
                11_091_344_671_253_066_420,
                13_793_997_310_169_335_082,
                1_900_383_378_846_508_768,
                7_684_712_102_626_143_532,
            ]
        );
    }

    #[test]
    fn random_float_and_integer_conversion_match_golden_values() {
        let mut rng = PortableRng::from_seed(0);
        assert_eq!(rng.random_float(), 0.601_262_999_417_904_8);

        let mut rng = PortableRng::from_seed(0);
        assert_eq!(rng.random_int_inclusive(-10, 10), 1);

        let mut rng = PortableRng::from_seed(0);
        assert_eq!(
            rng.random_int_inclusive(i64::MIN, i64::MAX),
            -7_355_399_402_456_485_196
        );
    }

    #[test]
    fn all_zero_entropy_state_is_rejected() {
        assert!(PortableRng::from_state([0; 4]).is_err());
    }
}
