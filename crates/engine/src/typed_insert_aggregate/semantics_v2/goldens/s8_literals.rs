//! Independently captured B/C/D retained-response envelope literals.
//!
//! These test-only values deliberately do not call the fixture reframe helpers, so a
//! self-consistent builder drift cannot update the accepted envelope identities.

pub(super) struct S8EnvelopeLiteral {
    pub(super) s8_bytes: usize,
    pub(super) s8_sha256: &'static str,
    pub(super) s8_payload_digest: &'static str,
    pub(super) s8_section_root: &'static str,
    pub(super) response_root: &'static str,
    pub(super) aggregate_root: &'static str,
    pub(super) status_hex: &'static str,
    pub(super) status_deadline: u64,
    pub(super) status_artifact_count: u32,
}

pub(super) const B_ENVELOPE_LITERAL: S8EnvelopeLiteral = S8EnvelopeLiteral {
    s8_bytes: 800,
    s8_sha256: "8a3d3fd01fc9b9e411f6f31bd0846a7fcd375e05390c0cf072c34282f3537247",
    s8_payload_digest: "f1ff30e95999ca5c15454a8c02d6662dfd385d1fd5f201c9861c745679fa191e",
    s8_section_root: "ca3bb4c78b4a8a46b40a38122dab68f3c4de0cec8528618122a7d47d359f9ab4",
    response_root: "27cb903e4e7493ce556295ecff4ad3b2212cb2bcac8db622104986a7e201cfb7",
    aggregate_root: "975eb375d2a8d037eb3276896a81c417cc039d1082d38ca22db23c4fdca3e224",
    status_hex: "475055444253544154555332a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2b100000000000000f98849a75458d7dce64f41df0d1ca59fb2b47cf8b0edebe84adf6b12222955fa0101000000000000840300000000000002000000010000003be47bd3d65a56fecc81f0bc49150f39c2957b777646ce056e507c8a2ad003ed27cb903e4e7493ce556295ecff4ad3b2212cb2bcac8db622104986a7e201cfb7975eb375d2a8d037eb3276896a81c417cc039d1082d38ca22db23c4fdca3e224",
    status_deadline: 900,
    status_artifact_count: 1,
};

pub(super) const C_ENVELOPE_LITERAL: S8EnvelopeLiteral = S8EnvelopeLiteral {
    s8_bytes: 1900,
    s8_sha256: "fc534af7b5217e2f7252d66d1e1a5d9e999e0256b7b715ecc4a8615b8f1856ea",
    s8_payload_digest: "e99a9582056c42aeb83321ab6da48752fe0ac9845066bebd1e81f73d84f71044",
    s8_section_root: "623c6fc522389b4ad31ea69ef5f291eee61d27cdfb4f0b61fcbb051da296770f",
    response_root: "13814345cbddfe17a4750644b2edef2eb3028b51a250bc7d47864b76cbf610f7",
    aggregate_root: "a3ae39a29d6a3890c36d29128f8170bb031b48b2989dba29deb84f68ebff5b5b",
    status_hex: "475055444253544154555332a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2b1000000000000006e12148f8efa26c46916921d23d3336d257f11100c42c480d54168cf10890a5c01010000000000008403000000000000030000000200000067f2a36d0e60419a6ff314243d7f878fc5d5a5466ac232686b6c00750e2aa17d13814345cbddfe17a4750644b2edef2eb3028b51a250bc7d47864b76cbf610f7a3ae39a29d6a3890c36d29128f8170bb031b48b2989dba29deb84f68ebff5b5b",
    status_deadline: 900,
    status_artifact_count: 2,
};

pub(super) const D_ENVELOPE_LITERAL: S8EnvelopeLiteral = S8EnvelopeLiteral {
    s8_bytes: 1016,
    s8_sha256: "5456138ab92b0755703deb8a3717ff77b3233dd1e84440456b39c2490ab9bf12",
    s8_payload_digest: "a2785267c9dc23b55724a29798131a41f889a165db51743d75f833942bf3d445",
    s8_section_root: "b43c6fd14a406ed5ee3e6b46e02cf5d593ad016cbd857532042c70f4541cb53e",
    response_root: "c7698d1dd22c3c9babed831100baa8f8cc8da49e010b753f4e168756416faf22",
    aggregate_root: "e995a862d711c01371f4b25a80b798f771c2615b66a4ea22617d42f5d00819d1",
    status_hex: "475055444253544154555332a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2b1000000000000006e12148f8efa26c46916921d23d3336d257f11100c42c480d54168cf10890a5c0101000000000000840300000000000003000000010000004f0f2c4dda108c93f993dfb5820d65df0126b77e4175e1722032967289651fa3c7698d1dd22c3c9babed831100baa8f8cc8da49e010b753f4e168756416faf22e995a862d711c01371f4b25a80b798f771c2615b66a4ea22617d42f5d00819d1",
    status_deadline: 900,
    status_artifact_count: 1,
};
