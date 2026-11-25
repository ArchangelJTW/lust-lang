#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct TelemetrySignalMask(pub u32);

impl TelemetrySignalMask {
    pub const NONE: Self = Self(0);
    pub const GAS: Self = Self(1 << 0);
    pub const MEMORY: Self = Self(1 << 1);
    pub const HOST_LEDGER: Self = Self(1 << 2);
    pub const VM: Self = Self(Self::GAS.0 | Self::MEMORY.0 | Self::HOST_LEDGER.0);
    pub const JIT: Self = Self(1 << 3);
    pub const EFFECTS: Self = Self(1 << 4);
    pub const USER: Self = Self(1 << 5);
    pub const ALL: Self = Self(Self::VM.0 | Self::JIT.0 | Self::EFFECTS.0 | Self::USER.0);

    #[inline]
    pub const fn contains(&self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    #[inline]
    pub const fn union(&self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    #[inline]
    pub const fn intersection(&self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    #[inline]
    pub fn insert(&mut self, other: Self) {
        self.0 |= other.0;
    }

    #[inline]
    pub fn remove(&mut self, other: Self) {
        self.0 &= !other.0;
    }

    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.0 == 0
    }

    #[inline]
    pub const fn bits(&self) -> u32 {
        self.0
    }

    #[inline]
    pub const fn from_bits_truncate(bits: u32) -> Self {
        Self(bits & Self::ALL.0)
    }
}

impl core::ops::BitOr for TelemetrySignalMask {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self::Output {
        self.union(rhs)
    }
}

impl core::ops::BitAnd for TelemetrySignalMask {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self::Output {
        self.intersection(rhs)
    }
}

impl core::ops::BitOrAssign for TelemetrySignalMask {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.insert(rhs);
    }
}
