use core::marker::PhantomData;

use p3_field::{
    extension::{BinomialExtensionField, QuinticTrinomialExtensionField},
    BasedVectorSpace, ExtensionField, Field, TwoAdicField,
};
use p3_koala_bear::{
    default_koalabear_poseidon1_16, default_koalabear_poseidon2_16, default_koalabear_poseidon2_24,
    KoalaBear, Poseidon1KoalaBear, Poseidon2KoalaBear,
};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};

use crate::hashers::{Keccak256NodeCompress, KeccakFieldLeafHasher};
use crate::SpartanWhirEngine;
use crate::{CanonicalKeccakChallenger32, KeccakByteChallenger};

pub type F = KoalaBear;
pub type QuarticBinExtension = BinomialExtensionField<F, 4>;
pub type OcticBinExtension = BinomialExtensionField<F, 8>;
pub type QuinticExtension = QuinticTrinomialExtensionField<F>;
pub type KeccakFieldHash = KeccakFieldLeafHasher;
pub type KeccakNodeCompress = Keccak256NodeCompress;
pub type KeccakChallenger = CanonicalKeccakChallenger32<F>;
pub type Poseidon16 = Poseidon2KoalaBear<16>;
pub type Poseidon24 = Poseidon2KoalaBear<24>;
pub type PoseidonFieldHash = PaddingFreeSponge<Poseidon24, 24, 16, 8>;
pub type PoseidonNodeCompress = TruncatedPermutation<Poseidon16, 2, 8, 16>;
pub type PoseidonChallenger = crate::poseidon_trace::TraceablePoseidonChallenger;
pub type PoseidonZkChallenger = PoseidonChallenger;
pub type Poseidon1Permutation = Poseidon1KoalaBear<16>;
pub type Poseidon1FieldHash = PaddingFreeSponge<Poseidon1Permutation, 16, 8, 8>;
pub type Poseidon1NodeCompress = TruncatedPermutation<Poseidon1Permutation, 2, 8, 16>;
pub type Poseidon1Challenger =
    crate::poseidon_trace::TraceablePoseidonChallenger<Poseidon1Permutation>;

pub trait ExtField:
    ExtensionField<F> + BasedVectorSpace<F> + TwoAdicField + Copy + Send + Sync
{
}

pub trait Plonky3PoseidonEngine:
    SpartanWhirEngine<F = F, W = F, PackedF = <F as Field>::Packing, PackedW = <F as Field>::Packing>
{
    fn challenger() -> Self::Challenger;
    fn merkle_hash() -> Self::Hash;
    fn merkle_compress() -> Self::Compress;
}

impl<Ext> ExtField for Ext where
    Ext: ExtensionField<F> + BasedVectorSpace<F> + TwoAdicField + Copy + Send + Sync
{
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct KeccakEngine<EF>(PhantomData<EF>);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PoseidonEngine<EF>(PhantomData<EF>);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Poseidon1Engine<EF>(PhantomData<EF>);

pub type KeccakQuarticEngine = KeccakEngine<QuarticBinExtension>;
pub type KeccakOcticEngine = KeccakEngine<OcticBinExtension>;
pub type KeccakQuinticEngine = KeccakEngine<QuinticExtension>;
pub type PoseidonQuarticEngine = PoseidonEngine<QuarticBinExtension>;
pub type PoseidonOcticEngine = PoseidonEngine<OcticBinExtension>;
/// Quartic and quintic engines are intentionally distinct types at the API boundary.
///
/// ```compile_fail
/// use spartan_whir::{
///     Plonky3WhirPcs, PoseidonQuarticEngine, PoseidonQuinticEngine, SpartanProof,
/// };
///
/// fn needs_quintic(_proof: SpartanProof<PoseidonQuinticEngine, Plonky3WhirPcs>) {}
///
/// let take_quartic: fn(SpartanProof<PoseidonQuarticEngine, Plonky3WhirPcs>) = needs_quintic;
/// ```
pub type PoseidonQuinticEngine = PoseidonEngine<QuinticExtension>;
pub type Poseidon1QuarticEngine = Poseidon1Engine<QuarticBinExtension>;
pub type Poseidon1OcticEngine = Poseidon1Engine<OcticBinExtension>;
pub type Poseidon1QuinticEngine = Poseidon1Engine<QuinticExtension>;

impl<Ext: ExtField> SpartanWhirEngine for KeccakEngine<Ext> {
    type F = F;
    type EF = Ext;
    type Challenger = KeccakChallenger;
    type Hash = KeccakFieldHash;
    type Compress = KeccakNodeCompress;
    type W = u64;
    type PackedF = F;
    type PackedW = u64;

    const DIGEST_ELEMS: usize = 4;
}

impl<Ext: ExtField> SpartanWhirEngine for PoseidonEngine<Ext> {
    type F = F;
    type EF = Ext;
    type Challenger = PoseidonChallenger;
    type Hash = PoseidonFieldHash;
    type Compress = PoseidonNodeCompress;
    type W = F;
    type PackedF = <F as Field>::Packing;
    type PackedW = <F as Field>::Packing;

    const DIGEST_ELEMS: usize = 8;
}

impl<Ext: ExtField> SpartanWhirEngine for Poseidon1Engine<Ext> {
    type F = F;
    type EF = Ext;
    type Challenger = Poseidon1Challenger;
    type Hash = Poseidon1FieldHash;
    type Compress = Poseidon1NodeCompress;
    type W = F;
    type PackedF = <F as Field>::Packing;
    type PackedW = <F as Field>::Packing;

    const DIGEST_ELEMS: usize = 8;
}

impl<Ext: ExtField> Plonky3PoseidonEngine for PoseidonEngine<Ext> {
    fn challenger() -> Self::Challenger {
        poseidon_challenger()
    }

    fn merkle_hash() -> Self::Hash {
        poseidon_merkle_hash()
    }

    fn merkle_compress() -> Self::Compress {
        poseidon_merkle_compress()
    }
}

impl<Ext: ExtField> Plonky3PoseidonEngine for Poseidon1Engine<Ext> {
    fn challenger() -> Self::Challenger {
        poseidon1_challenger()
    }

    fn merkle_hash() -> Self::Hash {
        poseidon1_merkle_hash()
    }

    fn merkle_compress() -> Self::Compress {
        poseidon1_merkle_compress()
    }
}

pub fn keccak_challenger() -> KeccakChallenger {
    KeccakChallenger::new(KeccakByteChallenger::default())
}

pub fn poseidon_merkle_hash() -> PoseidonFieldHash {
    PoseidonFieldHash::new(default_koalabear_poseidon2_24())
}

pub fn poseidon_merkle_compress() -> PoseidonNodeCompress {
    PoseidonNodeCompress::new(default_koalabear_poseidon2_16())
}

pub fn poseidon_challenger() -> PoseidonChallenger {
    PoseidonChallenger::new(default_koalabear_poseidon2_16())
}

/// Poseidon2 challenger retained for the existing full-ZK API names.
pub fn poseidon_zk_challenger() -> PoseidonZkChallenger {
    poseidon_challenger()
}

pub fn poseidon1_merkle_hash() -> Poseidon1FieldHash {
    Poseidon1FieldHash::new(default_koalabear_poseidon1_16())
}

pub fn poseidon1_merkle_compress() -> Poseidon1NodeCompress {
    Poseidon1NodeCompress::new(default_koalabear_poseidon1_16())
}

pub fn poseidon1_challenger() -> Poseidon1Challenger {
    Poseidon1Challenger::new(default_koalabear_poseidon1_16())
}
