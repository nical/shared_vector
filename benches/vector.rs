use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use shared_vector::{AtomicSharedVector, SharedVector, UniqueVector, SharedChunkVector, UniqueChunkVector};

criterion_group!(vector, vector_push);
criterion_main!(vector);

fn push_shared(n: u32, initial_cap: usize) {
    let mut v = SharedVector::with_capacity(initial_cap);
    for i in 0..n {
        v.push(i);
    }
    black_box(v);
}

fn push_atomic(n: u32, initial_cap: usize) {
    let mut v = AtomicSharedVector::with_capacity(initial_cap);
    for i in 0..n {
        v.push(i);
    }
    black_box(v);
}

fn push_unique(n: u32, initial_cap: usize) {
    let mut v = UniqueVector::with_capacity(initial_cap);
    for i in 0..n {
        v.push(i);
    }
    black_box(v);
}

fn push_std(n: u32, initial_cap: usize) {
    let mut v = Vec::with_capacity(initial_cap);
    for i in 0..n {
        v.push(i);
    }
    black_box(v);
}

fn push_chunks(n: u32, chunk_size: usize) {
    let mut v = SharedChunkVector::new(chunk_size);
    for i in 0..n {
        v.push(i);
    }
    black_box(v);
}

fn push_chunks_unique(n: u32, chunk_size: usize) {
    let mut v = UniqueChunkVector::new(chunk_size);
    for i in 0..n {
        v.push(i);
    }
    black_box(v);
}

fn vector_push(c: &mut Criterion) {
    let mut g = c.benchmark_group("push");

    for item_count in [1000, 10_000] {
        for initial_cap in [1024, 256, 32] {
            g.bench_with_input(BenchmarkId::new(&format!("shared({initial_cap})"), &item_count), &item_count, |b, item_count| b.iter (||push_shared(*item_count, black_box(initial_cap))));
            g.bench_with_input(BenchmarkId::new(&format!("atomic({initial_cap})"), &item_count), &item_count, |b, item_count| b.iter (||push_atomic(*item_count, black_box(initial_cap))));
            g.bench_with_input(BenchmarkId::new(&format!("unique({initial_cap})"), &item_count), &item_count, |b, item_count| b.iter(||push_unique(*item_count, black_box(initial_cap))));
            g.bench_with_input(BenchmarkId::new(&format!("std({initial_cap})"), &item_count), &item_count, |b, item_count| b.iter(||push_std(*item_count, black_box(initial_cap))));
            g.bench_with_input(BenchmarkId::new(&format!("chunk({initial_cap})"), &item_count), &item_count, |b, item_count| b.iter(||push_chunks(*item_count, black_box(initial_cap))));
            g.bench_with_input(BenchmarkId::new(&format!("chunk_unique({initial_cap})"), &item_count), &item_count, |b, item_count| b.iter(||push_chunks_unique(*item_count, black_box(initial_cap))));
        }
    }
}
