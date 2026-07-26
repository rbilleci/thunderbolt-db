//! Feature-gated analysis instrumentation (docs/proposals/feature-gated-instrumentation.md).
//!
//! With the `probe-timing` feature OFF (the default), [`Probe`] is a zero-sized type and every method body
//! is empty, so the compiler elides all of it -- no branch, no symbol, no perturbation of the very thing
//! being measured. (A Cargo feature, not `#[cfg(debug_assertions)]`: we measure in `--release`, where the
//! debug-assertions cfg is stripped.) With the feature ON, `start()` stamps the clock and `mark`/`lap`
//! print durations to stderr.
//!
//! The point: measurement *points* are the hard-won part -- keep probes here PERMANENTLY instead of
//! writing-then-reverting them. CI builds `--features probe-timing` to keep them from bit-rotting.
//!
//! ```ignore
//! use gpu_db_execution::Probe;
//!
//! let p = Probe::start();
//! // ... compare kernel ...
//! p.lap("compare");   // prints the compare duration, resets the lap clock
//! // ... compact kernel ...
//! p.lap("compact");   // prints the compact duration
//! ```

#[cfg(feature = "probe-timing")]
pub struct Probe {
    start: std::time::Instant,
    last: std::cell::Cell<std::time::Instant>,
}

#[cfg(not(feature = "probe-timing"))]
pub struct Probe;

impl Probe {
    /// Start a timer. A no-op ZST constructor when `probe-timing` is off (fully elided).
    #[inline(always)]
    pub fn start() -> Self {
        #[cfg(feature = "probe-timing")]
        {
            let now = std::time::Instant::now();
            Self {
                start: now,
                last: std::cell::Cell::new(now),
            }
        }
        #[cfg(not(feature = "probe-timing"))]
        {
            Self
        }
    }

    /// Print `label` + the cumulative elapsed time since [`start`](Self::start). Empty when off.
    #[inline(always)]
    pub fn mark(&self, _label: &str) {
        #[cfg(feature = "probe-timing")]
        eprintln!("[probe] {_label} {}us", self.start.elapsed().as_micros());
    }

    /// Print `label` + the time since the LAST `lap`/`start`, then reset the lap clock -- one line per
    /// phase in a multi-phase span. Empty when off.
    #[inline(always)]
    pub fn lap(&self, _label: &str) {
        #[cfg(feature = "probe-timing")]
        {
            let now = std::time::Instant::now();
            eprintln!(
                "[probe] {_label} {}us",
                now.duration_since(self.last.get()).as_micros()
            );
            self.last.set(now);
        }
    }

    /// Print a measured scalar beside the wall-clock phases (for example a CUDA event duration). Empty
    /// when `probe-timing` is off. Keeping scalar reporting here gives permanent probes one stable output
    /// format without making production callers carry feature gates.
    #[inline(always)]
    pub fn value(_label: &str, _value: u64, _unit: &str) {
        #[cfg(feature = "probe-timing")]
        eprintln!("[probe] {_label} {_value}{_unit}");
    }

    /// A drop-scoped timer: prints `label` + the elapsed time when the returned guard drops (e.g. at the end
    /// of the enclosing function/block). Convenient for whole-function timing without a trailing `mark`.
    #[inline(always)]
    pub fn scope(_label: &'static str) -> ProbeScope {
        #[cfg(feature = "probe-timing")]
        {
            ProbeScope {
                start: std::time::Instant::now(),
                label: _label,
            }
        }
        #[cfg(not(feature = "probe-timing"))]
        {
            ProbeScope
        }
    }
}

#[cfg(feature = "probe-timing")]
pub struct ProbeScope {
    start: std::time::Instant,
    label: &'static str,
}
#[cfg(not(feature = "probe-timing"))]
pub struct ProbeScope;

#[cfg(feature = "probe-timing")]
impl Drop for ProbeScope {
    fn drop(&mut self) {
        eprintln!(
            "[probe] {} {}us",
            self.label,
            self.start.elapsed().as_micros()
        );
    }
}
