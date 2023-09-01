// Copyright 2023 RISC Zero, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

pub const RECURSION_CONTROL_IDS: [&str; 29] = [
    "f0b7cc120dc3c34b4548d34f688c1424846d0600a3331277f6710c113fce4e72",
    "3a8de34cd9c65059ff59a5311e038556d63345070749052e6110576d5e36c83c",
    "1987251f3589eb2eaac24f23dd35345c28e388752b49760a3dc80574f822891c",
    "10c6e26dca834a38b1d7836961372945f04a481e34779e6123fe286d54a8b64d",
    "26fdac1de5970b2b05d4b320a550fc47c942a960db500877bbe96c187a965a74",
    "6491e73707b9dc76e7b00224d5fd472058c7ce3a06dc7064ec63ab6e20e2e851",
    "bf7f2031bed5337296093a54e7f3d636811c50187c00c51d9a386f3ba2600f0d",
    "9058842f3d182a4f0c5837089302116f2d757902ae0d7414ee3ea7284f27ad34",
    "2b3aa1752bfb62157005d95c7650394f0978236b118c4338bc833557c52f7624",
    "162fbe4c178a3633b49beb6f16ee6e709d2be7682998526dc17273359e674745",
    "6af58c4e7f2ebd61d5e8560ad154ec4b7c762428d9c8ad74facfc1471d863515",
    "63645f59076d4124742d5b5a8c325d6c45dbd166a881cf6996f4d5316f2e5071",
    "e569af067d234020b6d0d715d1bc0c5b82b54264255b272241a616573184fb47",
    "a69d8761f3ab76093195571860f0262635e8253bc2324771d6768266277e5d36",
    "b7068836a9f6ba4af9f9390cfdc5d24d14331a4a4feadb16f0ab332c1f667465",
    "38b78e49ae4a6b6f487c6418d043586297cd9a2a5a16cf40d1f25f1f383e9b5d",
    "411bac17ad8e59302258cb6d43553a1690907d1263c2df1d0cf7c0339ef2d440",
    "791d821eeb20be3fcecb6811311322568d568a54e5d78a09ec64f4133a1f931c",
    "5e913c5f0fe74e4bf59d1a41f6b4c5280f077f1f7d62fb1ffae9c31b08798b13",
    "65543f379e502224bcbf184b7cc91a0f5fe24870dfc2130e9f99844f4f801d19",
    "06950b424f3aa66f7753b26b03e7a53dae76732e5444e81194917c28079cc320",
    "ae009311cd50aa5fc11c6d22ec064037a103c249df7c5c060f5bff7105ab0f67",
    "d212413d6ced8712ffae393b13c8ee53ef265f5719934729c22e1f5cd48f6416",
    "3a344d749f6fc635c570721cf83d160e657ef1632da7c65c1b430a37beea2712",
    "164c092422f9b44f09d4654fcbd5b22747e3b368bb8252501c0d075b196e6f60",
    "d88d14387aea843f4304475a9be72e76627a8761a82baf714f96821ae9dcf261",
    "6b1dc9639d3ed521de3b7b1494286110ebdadc60902dc075c75bce521e525c45",
    "93ddfd13ad326c73317033379bda035dd1e97a056aaed01c5beca909d764d213",
    "6b6a5660631ea4764bec404ae28c4c73281dd81e9ea54b63f43055737425961f",
];

/// Merkle root of the RECURSION_CONTROL_IDS
pub const ALLOWED_IDS_ROOT: &str =
    "b32a0567a799174e9f49dc3d8b2de4683192d345045b5828a0e045164f680238";
