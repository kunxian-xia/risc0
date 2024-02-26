use criterion::BatchSize;
use criterion::{criterion_group, criterion_main, Criterion};
use rand::thread_rng;
use risc0_core::field::{baby_bear::BabyBearElem, Elem};
use risc0_zkp::core::hash::poseidon2::Poseidon2HashSuite;
use risc0_zkp::hal::cpu::CpuHal;
#[cfg(feature = "cuda")]
use risc0_zkp::hal::cuda::CudaHalPoseidon2;
use risc0_zkp::hal::Hal;
use std::any::type_name;

fn bench_ntt_internal<H: Hal<Elem = BabyBearElem>>(
    c: &mut Criterion,
    h: H,
    log_n: usize,
    batch: usize,
) {
    let rng = &mut thread_rng();
    let polys = (0..(1 << log_n) * batch)
        .map(|_| BabyBearElem::random(rng))
        .collect::<Vec<_>>();

    c.bench_function(
        &format!("ntt_{}_{}_{}", type_name::<H>(), log_n, batch),
        |b| {
            b.iter_batched(
                //|| h.copy_from_elem("a", &polys),
                || {},
                |_| {
                    let buf = h.copy_from_elem("a", &polys);
                    h.batch_interpolate_ntt(&buf, batch);
                },
                BatchSize::SmallInput,
            );
        },
    );
}

fn bench_ntt(c: &mut Criterion) {
    for log_n in [19, 20, 21, 22] {
        for batch in [1, 5, 20, 50] {
            bench_ntt_internal(
                c,
                CpuHal::new(Poseidon2HashSuite::new_suite()),
                log_n,
                batch,
            );
            #[cfg(feature = "cuda")]
            bench_ntt_internal(c, CudaHalPoseidon2::new(), log_n, batch);
        }
    }
}

criterion_group!(benches, bench_ntt);
criterion_main!(benches);
