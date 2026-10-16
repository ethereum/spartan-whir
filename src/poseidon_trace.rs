use alloc::{sync::Arc, vec::Vec};
use core::fmt;
use std::sync::Mutex;

use p3_challenger::{
    CanFinalizeDigest, CanObserve, CanSample, CanSampleBits, CanSampleUniformBits,
    DuplexChallenger, FieldChallenger, GrindingChallenger, ResamplingError,
};
use p3_field::{BasedVectorSpace, Field, PrimeField32};
use p3_symmetric::{CryptographicPermutation, Hash, MerkleCap};
use serde::{Deserialize, Serialize};

use crate::engine::{Poseidon16, F};

type InnerPoseidonChallenger<P> = DuplexChallenger<F, P, 16, 8>;

/// One operation performed by the Poseidon Fiat-Shamir challenger.
///
/// Values are canonical KoalaBear representatives. An extension-field sample
/// stores its basis coefficients in ascending order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case")]
pub enum PoseidonTranscriptEvent {
    Observe {
        values: Vec<u32>,
    },
    ObserveCommitment {
        roots: usize,
        values: Vec<u32>,
    },
    Sample {
        values: Vec<u32>,
    },
    SampleBits {
        bits: usize,
        value: usize,
    },
    SampleUniformBits {
        bits: usize,
        resample: bool,
        value: Option<usize>,
    },
    Grind {
        bits: usize,
        witness: u32,
    },
    CheckWitness {
        bits: usize,
        witness: u32,
        accepted: bool,
    },
    Finalize {
        digest: Vec<u32>,
    },
}

/// Trace controls shared by the supported Poseidon transcript profiles.
pub trait PoseidonTranscriptTrace: Sized {
    fn with_trace(self) -> Self;
    fn transcript_trace(&self) -> Vec<PoseidonTranscriptEvent>;
}

/// The normal Poseidon2 challenger with optional operation tracing.
///
/// Tracing does not participate in challenger state. Clones share the same
/// trace sink so transcript branches are recorded in execution order.
#[derive(Clone)]
pub struct TraceablePoseidonChallenger<P = Poseidon16>
where
    P: CryptographicPermutation<[F; 16]>,
{
    inner: InnerPoseidonChallenger<P>,
    trace: Option<Arc<Mutex<Vec<PoseidonTranscriptEvent>>>>,
}

impl<P> fmt::Debug for TraceablePoseidonChallenger<P>
where
    P: CryptographicPermutation<[F; 16]>,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TraceablePoseidonChallenger")
            .field("tracing", &self.trace.is_some())
            .finish_non_exhaustive()
    }
}

impl<P> TraceablePoseidonChallenger<P>
where
    P: CryptographicPermutation<[F; 16]>,
{
    pub fn new(permutation: P) -> Self {
        Self {
            inner: InnerPoseidonChallenger::new(permutation),
            trace: None,
        }
    }

    /// Enable recording and discard any prior trace sink.
    pub fn with_trace(mut self) -> Self {
        self.trace = Some(Arc::new(Mutex::new(Vec::new())));
        self
    }

    /// Preserve the cryptographic state without sharing a transcript recorder.
    pub(crate) fn without_trace(mut self) -> Self {
        self.trace = None;
        self
    }

    /// Return a stable snapshot of all operations recorded so far.
    pub fn transcript_trace(&self) -> Vec<PoseidonTranscriptEvent> {
        self.trace
            .as_ref()
            .map(|trace| {
                trace
                    .lock()
                    .expect("transcript trace lock poisoned")
                    .clone()
            })
            .unwrap_or_default()
    }

    fn record(&self, event: PoseidonTranscriptEvent) {
        if let Some(trace) = &self.trace {
            trace
                .lock()
                .expect("transcript trace lock poisoned")
                .push(event);
        }
    }
}

impl<P> PoseidonTranscriptTrace for TraceablePoseidonChallenger<P>
where
    P: CryptographicPermutation<[F; 16]>,
{
    fn with_trace(self) -> Self {
        TraceablePoseidonChallenger::with_trace(self)
    }

    fn transcript_trace(&self) -> Vec<PoseidonTranscriptEvent> {
        TraceablePoseidonChallenger::transcript_trace(self)
    }
}

fn canonical_values(values: &[F]) -> Vec<u32> {
    values.iter().map(PrimeField32::as_canonical_u32).collect()
}

impl<P> CanObserve<F> for TraceablePoseidonChallenger<P>
where
    P: CryptographicPermutation<[F; 16]>,
{
    fn observe(&mut self, value: F) {
        self.inner.observe(value);
        self.record(PoseidonTranscriptEvent::Observe {
            values: vec![value.as_canonical_u32()],
        });
    }

    fn observe_slice(&mut self, values: &[F]) {
        self.inner.observe_slice(values);
        self.record(PoseidonTranscriptEvent::Observe {
            values: canonical_values(values),
        });
    }
}

impl<P, const N: usize> CanObserve<[F; N]> for TraceablePoseidonChallenger<P>
where
    P: CryptographicPermutation<[F; 16]>,
{
    fn observe(&mut self, values: [F; N]) {
        self.inner.observe(values);
        self.record(PoseidonTranscriptEvent::Observe {
            values: canonical_values(&values),
        });
    }
}

impl<P, const N: usize> CanObserve<Hash<F, F, N>> for TraceablePoseidonChallenger<P>
where
    P: CryptographicPermutation<[F; 16]>,
{
    fn observe(&mut self, values: Hash<F, F, N>) {
        let canonical = values.as_ref().to_vec();
        self.inner.observe(values);
        self.record(PoseidonTranscriptEvent::Observe {
            values: canonical_values(&canonical),
        });
    }
}

impl<P, const N: usize> CanObserve<&MerkleCap<F, [F; N]>> for TraceablePoseidonChallenger<P>
where
    P: CryptographicPermutation<[F; 16]>,
{
    fn observe(&mut self, cap: &MerkleCap<F, [F; N]>) {
        let values = cap
            .roots()
            .iter()
            .flatten()
            .map(PrimeField32::as_canonical_u32)
            .collect();
        self.inner.observe(cap);
        self.record(PoseidonTranscriptEvent::ObserveCommitment {
            roots: cap.num_roots(),
            values,
        });
    }
}

impl<P, const N: usize> CanObserve<MerkleCap<F, [F; N]>> for TraceablePoseidonChallenger<P>
where
    P: CryptographicPermutation<[F; 16]>,
{
    fn observe(&mut self, cap: MerkleCap<F, [F; N]>) {
        self.observe(&cap);
    }
}

impl<P> CanObserve<Vec<Vec<F>>> for TraceablePoseidonChallenger<P>
where
    P: CryptographicPermutation<[F; 16]>,
{
    fn observe(&mut self, values: Vec<Vec<F>>) {
        let canonical = values
            .iter()
            .flatten()
            .map(PrimeField32::as_canonical_u32)
            .collect();
        self.inner.observe(values);
        self.record(PoseidonTranscriptEvent::Observe { values: canonical });
    }
}

impl<P, A> CanSample<A> for TraceablePoseidonChallenger<P>
where
    A: BasedVectorSpace<F>,
    P: CryptographicPermutation<[F; 16]>,
{
    fn sample(&mut self) -> A {
        let value = <InnerPoseidonChallenger<P> as CanSample<A>>::sample(&mut self.inner);
        self.record(PoseidonTranscriptEvent::Sample {
            values: canonical_values(value.as_basis_coefficients_slice()),
        });
        value
    }
}

impl<P> CanSampleBits<usize> for TraceablePoseidonChallenger<P>
where
    P: CryptographicPermutation<[F; 16]>,
{
    fn sample_bits(&mut self, bits: usize) -> usize {
        let value = self.inner.sample_bits(bits);
        self.record(PoseidonTranscriptEvent::SampleBits { bits, value });
        value
    }
}

impl<P> CanSampleUniformBits<F> for TraceablePoseidonChallenger<P>
where
    P: CryptographicPermutation<[F; 16]>,
{
    fn sample_uniform_bits<const RESAMPLE: bool>(
        &mut self,
        bits: usize,
    ) -> Result<usize, ResamplingError> {
        let result = self.inner.sample_uniform_bits::<RESAMPLE>(bits);
        self.record(PoseidonTranscriptEvent::SampleUniformBits {
            bits,
            resample: RESAMPLE,
            value: result.as_ref().ok().copied(),
        });
        result
    }
}

impl<P> FieldChallenger<F> for TraceablePoseidonChallenger<P> where
    P: CryptographicPermutation<[F; 16]>
{
}

impl<P> GrindingChallenger for TraceablePoseidonChallenger<P>
where
    P: CryptographicPermutation<[F; 16]>,
    P: CryptographicPermutation<[<F as Field>::Packing; 16]>,
{
    type Witness = F;

    fn grind(&mut self, bits: usize) -> Self::Witness {
        let witness = self.inner.grind(bits);
        self.record(PoseidonTranscriptEvent::Grind {
            bits,
            witness: witness.as_canonical_u32(),
        });
        witness
    }

    fn check_witness(&mut self, bits: usize, witness: Self::Witness) -> bool {
        let accepted = self.inner.check_witness(bits, witness);
        self.record(PoseidonTranscriptEvent::CheckWitness {
            bits,
            witness: witness.as_canonical_u32(),
            accepted,
        });
        accepted
    }
}

impl<P> CanFinalizeDigest for TraceablePoseidonChallenger<P>
where
    P: CryptographicPermutation<[F; 16]>,
{
    type Digest = [F; 8];

    fn finalize(self) -> Self::Digest {
        let trace = self.trace.clone();
        let digest = self.inner.finalize();
        if let Some(trace) = trace {
            trace.lock().expect("transcript trace lock poisoned").push(
                PoseidonTranscriptEvent::Finalize {
                    digest: canonical_values(&digest),
                },
            );
        }
        digest
    }
}
