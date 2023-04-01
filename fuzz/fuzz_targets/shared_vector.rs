#![no_main]

use libfuzzer_sys::fuzz_target;
use shared_vector::SharedVector;

mod cmd;
use cmd::*;

fuzz_target!(|cmds: Vec<Cmd>| {

    let mut vectors: [SharedVector<Box<u32>>; 4] = [
        SharedVector::new(),
        SharedVector::new(),
        SharedVector::new(),
        SharedVector::new(),
    ];

    for cmd in cmds {
        match cmd {
            Cmd::AddRef { src_idx, dst_idx } => {
                vectors[slot(dst_idx)] = vectors[slot(src_idx)].new_ref();
            }
            Cmd::DropVec { idx } => {
                vectors[slot(idx)] = SharedVector::new();
            }
            Cmd::Clear { idx } => {
                vectors[slot(idx)].clear();
            }
            Cmd::Push { idx, val } => {
                vectors[slot(idx)].push(Box::new(val));
            }
            Cmd::PushWithinCapacity { idx, val } => {
                let _ = vectors[slot(idx)].push_within_capacity(Box::new(val));
            }
            Cmd::Pop { idx } => {
                vectors[slot(idx)].pop();
            }
            Cmd::ExtendFromSlice { idx } => {
                vectors[slot(idx)].extend_from_slice(&[Box::new(1), Box::new(2), Box::new(3)]);
            }
            Cmd::CloneBuffer { src_idx, dst_idx } => {
                vectors[slot(dst_idx)] = vectors[slot(src_idx)].clone_buffer();
                }
            Cmd::Iter { idx } => {
                let _: u32 = vectors[slot(idx)]
                    .iter()
                    .fold(0, |a, b| a.wrapping_add(**b));
            }
            Cmd::IterMut { idx } => {
                for elt in vectors[slot(idx)].iter_mut() { *elt = Box::new(1337); };
            }
            Cmd::EnsureUnique { idx } => {
                vectors[slot(idx)].ensure_unique();
            }
            Cmd::Append { src_idx, dst_idx } => {
                let mut v = std::mem::replace(&mut vectors[slot(src_idx)], SharedVector::new());
                vectors[slot(dst_idx)].append(&mut v);
            }
            Cmd::WithCapacity { idx, cap } => {
                vectors[slot(idx)] = SharedVector::with_capacity(cap % 1024);
            }
            Cmd::FromSlice { src_idx, dst_idx } => {
                vectors[slot(dst_idx)] = SharedVector::from_slice(vectors[slot(src_idx)].as_slice());
            }
            Cmd::AsMutSlice { idx } => {
                for v in vectors[slot(idx)].as_mut_slice() { *v = Box::new(42); };
            }
            Cmd::First { idx } => {
                let _ = vectors[slot(idx)].first();
            }
            Cmd::Last { idx } => {
                let _ = vectors[slot(idx)].last();
            }
            Cmd::FirstMut { idx } => {
                if let Some(elt) = vectors[slot(idx)].first_mut() { *elt = Box::new(1); }
            }
            Cmd::LastMut { idx } => {
                if let Some(elt) = vectors[slot(idx)].last_mut() { *elt = Box::new(2); }
            }
            Cmd::Reserve { idx, val } => {
                vectors[slot(idx)].reserve(reserve_max(vectors[slot(idx)].len(), val));
            }
            Cmd::Convert { idx } => {
                let a = std::mem::replace(&mut vectors[slot(idx)], SharedVector::new());
                vectors[slot(idx)] = a.into_unique().into_shared();
            }
            Cmd::Swap { idx, offsets } => {
                let vec = &mut vectors[slot(idx)];
                let len = vec.len();
                if !vec.is_empty() {
                    vec.swap(offsets.0 % len, offsets.1 % len)
                }
            }
            Cmd::SwapRemove { idx, offset } => {
                let vec = &mut vectors[slot(idx)];
                if vec.is_empty() {
                    return;
                }
                vec.swap_remove(offset % vec.len());
            }
            Cmd::ShrinkTo { idx, cap } => {
                vectors[slot(idx)].shrink_to(cap);
            }
            Cmd::ShrinkToFit { idx } => {
                vectors[slot(idx)].shrink_to_fit();
            }
            Cmd::Drain { .. } => {
                // TODO
            }
            Cmd::Splice { .. } => {
                // TODO
            }
        }
    }
});
