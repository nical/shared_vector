#![no_main]

use libfuzzer_sys::fuzz_target;
use shared_vector::UniqueVector;

mod cmd;
use cmd::*;

fuzz_target!(|cmds: Vec<Cmd>| {

    let mut vectors: [UniqueVector<Box<u32>>; 4] = [
        UniqueVector::new(),
        UniqueVector::new(),
        UniqueVector::new(),
        UniqueVector::new(),
    ];

    for cmd in cmds {
        match cmd {
            Cmd::AddRef { .. } => {
                //vectors[slot(dst_idx)] = vectors[slot(src_idx)].new_ref();
            }
            Cmd::DropVec { idx } => {
                vectors[slot(idx)] = UniqueVector::new();
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
            Cmd::PushSlice { idx } => {
                vectors[slot(idx)].push_slice(&[Box::new(1), Box::new(2), Box::new(3)]);
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
            Cmd::EnsureUnique { .. } => {}
            Cmd::Append { src_idx, dst_idx } => {
                let mut v = std::mem::replace(&mut vectors[slot(src_idx)], UniqueVector::new());
                vectors[slot(dst_idx)].append(&mut v);
            }
            Cmd::WithCapacity { idx, cap } => {
                vectors[slot(idx)] = UniqueVector::with_capacity(cap % 1024);
            }
            Cmd::FromSlice { src_idx, dst_idx } => {
                vectors[slot(dst_idx)] = UniqueVector::from_slice(vectors[slot(src_idx)].as_slice());
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
                let a = std::mem::replace(&mut vectors[slot(idx)], UniqueVector::new());
                vectors[slot(idx)] = a.into_shared().into_unique();
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
        }
    }
});
