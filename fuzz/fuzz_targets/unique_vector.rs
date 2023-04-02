#![no_main]

use libfuzzer_sys::fuzz_target;
use shared_vector::Vector;

mod cmd;
use cmd::*;

fuzz_target!(|cmds: Vec<Cmd>| {

    //fn print() {
    //    use Cmd::*;
    //    cmds_to_src("Vector", &[]);
    //}
    //print();

    let mut vectors: [Vector<Box<u32>>; 4] = [
        Vector::new(),
        Vector::new(),
        Vector::new(),
        Vector::new(),
    ];

    for cmd in cmds {
        match cmd {
            Cmd::AddRef { .. } => {
                //vectors[slot(dst_idx)] = vectors[slot(src_idx)].new_ref();
            }
            Cmd::DropVec { idx } => {
                vectors[slot(idx)] = Vector::new();
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
            Cmd::EnsureUnique { .. } => {}
            Cmd::Append { src_idx, dst_idx } => {
                let mut v = std::mem::replace(&mut vectors[slot(src_idx)], Vector::new());
                vectors[slot(dst_idx)].append(&mut v);
            }
            Cmd::WithCapacity { idx, cap } => {
                vectors[slot(idx)] = Vector::with_capacity(cap % 1024);
            }
            Cmd::FromSlice { src_idx, dst_idx } => {
                vectors[slot(dst_idx)] = Vector::from_slice(vectors[slot(src_idx)].as_slice());
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
                let a = std::mem::replace(&mut vectors[slot(idx)], Vector::new());
                vectors[slot(idx)] = a.into_shared().into_unique();
            }
            Cmd::Swap { idx, offsets } => {
                let vec = &mut vectors[slot(idx)];
                let len = vec.len();
                if !vec.is_empty() {
                    vec.swap(offsets.0 % len, offsets.1 % len)
                }
            }
            Cmd::Remove { idx, offset } => {
                let vec = &mut vectors[slot(idx)];
                if vec.is_empty() {
                    return;
                }
                vec.remove(offset % vec.len());
            }
            Cmd::SwapRemove { idx, offset } => {
                let vec = &mut vectors[slot(idx)];
                if vec.is_empty() {
                    return;
                }
                vec.swap_remove(offset % vec.len());
            }
            Cmd::Insert { idx, offset, val } => {
                let len = vectors[slot(idx)].len();
                vectors[slot(idx)].insert(offset % len.max(1), Box::new(val));
            }
            Cmd::ShrinkTo { idx, cap } => {
                vectors[slot(idx)].shrink_to(cap);
            }
            Cmd::ShrinkToFit { idx } => {
                vectors[slot(idx)].shrink_to_fit();
            }
            Cmd::Drain { idx, start, count } => {
                let vec = &mut vectors[slot(idx)];
                let len = vec.len();
                let start = if len > 0 { start % len} else { 0 };
                let end = (start + (count % 5)).min(len);
                vectors[slot(idx)].drain(start..end);
            }
            Cmd::Splice { idx, start, rem_count, val, add_count } => {
                let vec = &mut vectors[slot(idx)];
                let len = vec.len();
                let start = if len > 0 { start % len } else { 0 };
                let end = (start + (rem_count % 5)).min(len);
                let items = vec![Box::new(val); add_count % 10];
                vectors[slot(idx)].splice(start..end, items.into_iter());
            }
            Cmd::Retain { idx, bits } => {
                let mut i = 0;
                vectors[slot(idx)].retain(&mut |_: &Box<u32>| { i += 1; bits & (1 << i.min(31)) != 0 });
            }
        }
    }
});
