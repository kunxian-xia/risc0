// Copyright 2024 RISC Zero, Inc.
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

use criterion::BatchSize;
use criterion::{criterion_group, criterion_main, Criterion};
use rand::thread_rng;
use risc0_core::field::{baby_bear::BabyBearElem, Elem};
#[cfg(feature = "cuda")]
use risc0_zkp::hal::cuda::CudaHalPoseidon2;
use risc0_zkp::{
    core::hash::{
        poseidon::{poseidon_mix, CELLS as POSEIDON_CELLS},
        poseidon2::{poseidon2_mix, Poseidon2HashSuite, CELLS as POSEIDON2_CELLS},
    },
    hal::{cpu::CpuHal, Hal},
};
use std::any::type_name;

fn benchmark_poseidon_mix(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let mut cells = [BabyBearElem::random(&mut rng); POSEIDON_CELLS];
    c.bench_function("poseidon_mix", |b| b.iter(|| poseidon_mix(&mut cells)));
}

fn benchmark_poseidon2_mix(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let mut cells = [BabyBearElem::random(&mut rng); POSEIDON2_CELLS];
    c.bench_function("poseidon2_mix", |b| b.iter(|| poseidon2_mix(&mut cells)));
}

fn bench_hash_rows_internal<H: Hal<Elem = BabyBearElem>>(
    c: &mut Criterion,
    h: H,
    num_polys: usize,
    num_rows: usize,
) {
    let rng = &mut thread_rng();
    let rows = vec![BabyBearElem::random(rng); num_polys * num_rows];
    c.bench_function(
        &format!("hash_rows_{}_{}_{}", type_name::<H>(), num_rows, num_polys),
        |b| {
            b.iter_batched(
                || {},
                |_| {
                    let leave_hashes = h.alloc_digest("leaves", num_rows);
                    let matrix = h.copy_from_elem("matrix", &rows);

                    h.hash_rows(&leave_hashes, &matrix)
                },
                criterion::BatchSize::SmallInput,
            );
        },
    );
}

fn bench_hash_fold_internal<H: Hal<Elem = BabyBearElem>>(c: &mut Criterion, h: H, num_rows: usize) {
    c.bench_function(
        &format!("hash_fold_{}_{}", type_name::<H>(), num_rows),
        |b| {
            b.iter_batched(
                || h.alloc_digest("digests", num_rows * 2),
                |digests| {
                    let input_size = num_rows;
                    let output_size = num_rows / 2;
                    h.hash_fold(&digests, input_size, output_size);
                },
                BatchSize::SmallInput,
            );
        },
    );
}

fn bench_hasher(c: &mut Criterion) {
    for n_polys in [1, 5, 20, 50, 200] {
        for logn in [19, 20, 21, 22] {
            bench_hash_rows_internal(
                c,
                CpuHal::new(Poseidon2HashSuite::new_suite()),
                n_polys,
                1 << logn,
            );
            #[cfg(feature = "cuda")]
            bench_hash_rows_internal(c, CudaHalPoseidon2::new(), n_polys, 1 << logn);

            bench_hash_fold_internal(c, CpuHal::new(Poseidon2HashSuite::new_suite()), 1 << logn);
            #[cfg(feature = "cuda")]
            bench_hash_fold_internal(c, CudaHalPoseidon2::new(), 1 << logn);
        }
    }
}

criterion_group!(
    benches,
    bench_hasher,
    benchmark_poseidon_mix,
    benchmark_poseidon2_mix
);
criterion_main!(benches);
