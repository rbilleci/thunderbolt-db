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
    s8_sha256: "71fe7518654a41b39407f529044a2222bb32cbed0e43611272a8b5c3261ba860",
    s8_payload_digest: "6739453f2dcd48a7718120603a2d74cd5329a7c454afdbcbe97d17899fb819ea",
    s8_section_root: "013e4c0a5a9c80645d3a6732060a586439cb38397964c511f2f94116d446cff6",
    response_root: "c187f28537b4cf5e8ecc159f9667d990e923a54f50730f2dfb90f86de9554c52",
    aggregate_root: "b25493210865c7624a02f3d5d46ce58d23bd9a6759248e280808518366120897",
    status_hex: "475055444253544154555332a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2b100000000000000f98849a75458d7dce64f41df0d1ca59fb2b47cf8b0edebe84adf6b12222955fa0101000000000000840300000000000002000000010000003be47bd3d65a56fecc81f0bc49150f39c2957b777646ce056e507c8a2ad003edc187f28537b4cf5e8ecc159f9667d990e923a54f50730f2dfb90f86de9554c52b25493210865c7624a02f3d5d46ce58d23bd9a6759248e280808518366120897",
    status_deadline: 900,
    status_artifact_count: 1,
};

pub(super) const C_ENVELOPE_LITERAL: S8EnvelopeLiteral = S8EnvelopeLiteral {
    s8_bytes: 1900,
    s8_sha256: "0993f4378117d4da5ded86972af44a0f3d30cb6e26b22fa053f3f77072318f46",
    s8_payload_digest: "7f7f023e86e8faf05176876807872b582cfcfab770baa93c80d344ed1fd7c0a5",
    s8_section_root: "3fe2291bc0691975102ce1d81da306c5a28e03b0730f638c66e10c51f2075402",
    response_root: "f5b0f2efb4a09a6a0b4496759b2cea23cac0294104d51f5bff824e7bcc2fbd24",
    aggregate_root: "2a15e01f3ecfb00e038d9dc612bdbe749f7d058f93ae71033c688c012f44bac2",
    status_hex: "475055444253544154555332a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2b1000000000000006e12148f8efa26c46916921d23d3336d257f11100c42c480d54168cf10890a5c01010000000000008403000000000000030000000200000067f2a36d0e60419a6ff314243d7f878fc5d5a5466ac232686b6c00750e2aa17df5b0f2efb4a09a6a0b4496759b2cea23cac0294104d51f5bff824e7bcc2fbd242a15e01f3ecfb00e038d9dc612bdbe749f7d058f93ae71033c688c012f44bac2",
    status_deadline: 900,
    status_artifact_count: 2,
};

pub(super) const D_ENVELOPE_LITERAL: S8EnvelopeLiteral = S8EnvelopeLiteral {
    s8_bytes: 1016,
    s8_sha256: "001a2283fd6024e3cadfd6d84959766dea8266bcfd66e034da4062d93f0a58e4",
    s8_payload_digest: "bc1c446397f2668df592f30ff5cd58484559111208ec3bb6655a027c7d25b8c5",
    s8_section_root: "c94c539e39fc0dbedb7131a4ae9209d06b96549e40057a6cf78ce913675ff749",
    response_root: "be0b435214207c27bc36491b063f140b53ca2d23111203f8cf6e13df3b1fea51",
    aggregate_root: "fdcb7e016959318843f6abe3a5743be817a5953b66861d5b4105a17516249ac6",
    status_hex: "475055444253544154555332a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2b1000000000000006e12148f8efa26c46916921d23d3336d257f11100c42c480d54168cf10890a5c0101000000000000840300000000000003000000010000004f0f2c4dda108c93f993dfb5820d65df0126b77e4175e1722032967289651fa3be0b435214207c27bc36491b063f140b53ca2d23111203f8cf6e13df3b1fea51fdcb7e016959318843f6abe3a5743be817a5953b66861d5b4105a17516249ac6",
    status_deadline: 900,
    status_artifact_count: 1,
};
