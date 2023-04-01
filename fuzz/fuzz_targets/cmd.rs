use arbitrary::Arbitrary;

pub fn slot(idx: usize) -> usize { idx % 4 }
pub fn reserve_max(len: usize, additional: usize) -> usize {
    (additional % 1024).min(2048 - len.min(2048))
}

#[derive(Arbitrary, Copy, Clone, Debug)]
pub enum Cmd {
    AddRef { src_idx: usize, dst_idx: usize },
    DropVec { idx: usize },
    Clear { idx: usize },
    Push { idx: usize, val: u32 },
    PushWithinCapacity { idx: usize, val: u32 },
    ExtendFromSlice { idx: usize },
    Pop { idx: usize },
    CloneBuffer { src_idx: usize, dst_idx: usize },
    Iter { idx: usize },
    IterMut { idx: usize },
    EnsureUnique { idx: usize },
    Append { src_idx: usize, dst_idx: usize },
    WithCapacity { idx: usize, cap: usize },
    FromSlice { src_idx: usize, dst_idx: usize },
    AsMutSlice { idx: usize },
    First { idx: usize },
    Last { idx: usize, },
    FirstMut { idx: usize },
    LastMut { idx: usize, },
    Reserve { idx: usize, val: usize },
    Convert { idx: usize },
    Swap { idx: usize, offsets: (usize, usize) },
    Remove { idx: usize, offset: usize },
    SwapRemove { idx: usize, offset: usize },
    ShrinkTo { idx: usize, cap: usize },
    ShrinkToFit { idx: usize },
    Drain { idx: usize, start: usize, count: usize },
    Splice { idx: usize, start: usize, rem_count: usize, val: u32, add_count: usize },
}

fn cmd_to_string(vec_type: &str, cmd: Cmd) -> String {
    match cmd {
        Cmd::AddRef { src_idx, dst_idx } => {
            if vec_type == "UniqueVector" {
                return String::new();
            }
            let src_idx = slot(src_idx);
            let dst_idx = slot(dst_idx);
            format!("vectors[{dst_idx}] = vectors[{src_idx}].new_ref();")
        }
        Cmd::DropVec { idx } => {
            format!("vectors[{}] = {vec_type}::new();", slot(idx))
        }
        Cmd::Clear { idx } => {
            format!("vectors[{}].clear();", slot(idx))
        }
        Cmd::Push { idx, val } => {
            format!("vectors[{}].push(Box::new({val}));", slot(idx))
        }
        Cmd::PushWithinCapacity { idx, val } => {
            format!("vectors[{}].push_within_capacity(Box::new({val}));", slot(idx))
        }
        Cmd::ExtendFromSlice { idx } => {
            format!("vectors[{}].extend_from_slice(&[Box::new(1), Box::new(2), Box::new(3)]);", slot(idx)) 
        }
        Cmd::Pop { idx } => {
            format!("vectors[{}].pop();", slot(idx))
        }
        Cmd::CloneBuffer { src_idx, dst_idx } => {
            let src_idx = slot(src_idx);
            let dst_idx = slot(dst_idx);
            format!("vectors[{dst_idx}] = vectors[{src_idx}].clone_buffer();")
        }
        Cmd::Iter { idx } => {
            format!("let _: u32 = vectors[{}].iter().fold(0, |a, b| a.wrapping_add(**b));", slot(idx))
        }
        Cmd::IterMut { idx } => {
            format!("for elt in vectors[{}].iter_mut() {{ *elt = Box::new(1337); }}", slot(idx))
        }
        Cmd::EnsureUnique { idx } => {
            if vec_type == "UniqueVector" {
                return String::new();
            }
            format!("vectors[{}].ensure_unique();", slot(idx))
        }
        Cmd::Append { src_idx, dst_idx } => {
            let a = format!("let mut v = take(&mut vectors[{}]);", slot(src_idx));
            let b = format!("vectors[{}].append(&mut v);", slot(dst_idx));
            format!("{a}\n    {b}")
        }
        Cmd::WithCapacity { idx, cap } => {
            format!("vectors[{}] = {vec_type}::with_capacity({});", slot(idx), cap % 1024)
        }
        Cmd::FromSlice { src_idx, dst_idx } => {
            format!("vectors[{}] = {vec_type}::from_slice(vectors[{}].as_slice());", slot(dst_idx), slot(src_idx))
        }
        Cmd::AsMutSlice { idx } => {
            format!("for v in vectors[{}].as_mut_slice() {{ *v = Box::new(42); }};", slot(idx))
        }
        Cmd::First { idx } => {
            format!("let _ = vectors[{}].first();", slot(idx))
        }
        Cmd::Last { idx } => {
            format!("let _ = vectors[{}].last();", slot(idx))
        }
        Cmd::FirstMut { idx } => {
            format!("if let Some(elt) = vectors[{}].first_mut() {{ *elt = Box::new(1); }}", slot(idx))
        }
        Cmd::LastMut { idx } => {
            format!("if let Some(elt) = vectors[{}].last_mut() {{ *elt = Box::new(2); }} ", slot(idx))
        }
        Cmd::Reserve { idx, val } => {
            let val = val % 1024;
            let idx = slot(idx);
            format!("vectors[{idx}].reserve(reserve_max(vectors[{idx}].len(), {val}));")
        }
        Cmd::Convert { idx } => {
            let conv = match vec_type {
                "SharedVector" => "into_unique",
                "AtomicSharedVector" => "into_unique",
                "UniqueVector" => "into_shared",
                _ => panic!("unknwon type {vec_type}"),
            };
            let inv = match vec_type {
                "SharedVector" => "into_shared",
                "AtomicSharedVector" => "into_shared",
                "UniqueVector" => "into_unique",
                _ => panic!("unknwon type {vec_type}"),
            };
            let idx = slot(idx);
            let a = format!("let a = take(&mut vectors[{idx}]);");
            let b = format!("vectors[{idx}] = a.{conv}().{inv}();");
            format!("{a}\n    {b}")
        }
        Cmd::Swap { idx, offsets } => {
            let a = format!("let vec = &mut vectors[{}];", slot(idx));
            let b = format!("let len = vec.len();");
            let c = format!("if !vec.is_empty() {{ vec.swap({} % len, {} % len) }}", offsets.0, offsets.1);
            format!("{a}\n    {b}\n    {c}")
        }
        Cmd::Remove { idx, offset } => {
            let a = format!("let vec = &mut vectors[{}];", slot(idx));
            let b = format!("if !vec.is_empty() {{ vec.remove({offset} % vec.len()); }}");
            format!("{a}\n    {b}")
        }
        Cmd::SwapRemove { idx, offset } => {
            let a = format!("let vec = &mut vectors[{}];", slot(idx));
            let b = format!("if !vec.is_empty() {{ vec.swap_remove({offset} % vec.len()); }}");
            format!("{a}\n    {b}")
        }
        Cmd::ShrinkTo { idx, cap } => {
            format!("vectors[{}].shrink_to({cap});", slot(idx))
        }
        Cmd::ShrinkToFit { idx } => {
            format!("vectors[{}].shrink_to_fit();", slot(idx))
        }
        Cmd::Drain { idx, start, count } => {
            let s1 = format!("let vec = &mut vectors[{}];", slot(idx));
            let s2 = format!("let len = vec.len();");
            let s3 = format!("let start = if len > 0 {{ {start} % len }} else {{ 0 }};");
            let s4 = format!("let end = {}.min(len);", (start + (count % 5)));
            let s5 = format!("vectors[{}].drain(start..end);", slot(idx));
            format!("{s1}\n    {s2}\n    {s3}\n    {s4}\n    {s5}")
        }
        Cmd::Splice { idx, start, rem_count, val, add_count } => {
            let s1 = format!("let vec = &mut vectors[{}];", slot(idx));
            let s2 = format!("let len = vec.len();");
            let s3 = format!("let start = if len > 0 {{ {start} % len }} else {{ 0 }};");
            let s4 = format!("let end = (start + {}).min(len);", rem_count % 5);
            let s5 = format!("let items = vec![Box::new({val}); {}];", add_count % 10);
            let s6 = format!("vectors[{}].splice(start..end, items.into_iter());", slot(idx));
            format!("{s1}\n    {s2}\n    {s3}\n    {s4}\n    {s5}\n    {s6}")
        }
    }
}

#[allow(unused)]
pub fn cmds_to_src(vec_type: &str, cmds: &[Cmd]) {
    println!("// -------");
    println!("    fn reserve_max(len: usize, additional: usize) -> usize {{");
    println!("        additional.min(2048 - len.min(2048))");
    println!("    }}");
    println!("    fn take<T>(place: &mut {vec_type}<T>) -> {vec_type}<T> {{");
    println!("        std::mem::replace(place, {vec_type}::new())");
    println!("    }}");
    println!("    let mut vectors: [{vec_type}<Box<u32>>; 4] = [");
    println!("        {vec_type}::new(),");
    println!("        {vec_type}::new(),");
    println!("        {vec_type}::new(),");
    println!("        {vec_type}::new(),");
    println!("    ];");
    for cmd in cmds {
        println!("    {}", cmd_to_string(vec_type, *cmd));
    }
    println!("// -------");
}
