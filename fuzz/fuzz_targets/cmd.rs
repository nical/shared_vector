use arbitrary::Arbitrary;

pub fn slot(idx: usize) -> usize { idx % 4 }

#[derive(Arbitrary, Copy, Clone, Debug)]
pub enum Cmd {
    AddRef { src_idx: usize, dst_idx: usize },
    DropVec { idx: usize },
    Clear { idx: usize },
    Push { idx: usize, val: u32 },
    PushSlice { idx: usize },
    Pop { idx: usize },
    CloneBuffer { src_idx: usize, dst_idx: usize },
    Iter { idx: usize },
    IterMut { idx: usize },
    EnsureUnique { idx: usize },
    PushVector { src_idx: usize, dst_idx: usize },
    WithCapacity { idx: usize, cap: usize },
    FromSlice { src_idx: usize, dst_idx: usize },
    AsMutSlice { idx: usize },
    First { idx: usize },
    Last { idx: usize, },
    FirstMut { idx: usize },
    LastMut { idx: usize, },
    Reserve { idx: usize, val: usize },
    Convert { idx: usize },
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
        Cmd::PushSlice { idx } => {
            format!("vectors[{}].push_slice(&[Box::new(1), Box::new(2), Box::new(3)]);", slot(idx)) 
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
        Cmd::PushVector { src_idx, dst_idx } => {
            format!("vectors[{}].push_vector(std::mem::replace(&mut vectors[{}], {vec_type}::new()));", slot(dst_idx), slot(src_idx))
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
            format!("vectors[{}].reserve(({} % 1024).min(2048 - vectors[{}].len()));", slot(idx), val, slot(idx))
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
            let a = format!("let a = std::mem::replace(&mut vectors[{idx}], {vec_type}::new());");
            let b = format!("vectors[{idx}] = a.{conv}().{inv}();");
            format!("{a}\n{b}")
        }
    }
}

#[allow(unused)]
pub fn cmds_to_src(vec_type: &str, cmds: &[Cmd]) {
    println!("// -------");
    println!("let mut vectors: [{vec_type}<Box<u32>>; 4] = [");
    println!("  {vec_type}::new(),");
    println!("  {vec_type}::new(),");
    println!("  {vec_type}::new(),");
    println!("  {vec_type}::new(),");
    println!("];");
    for cmd in cmds {
        println!("{}", cmd_to_string(vec_type, *cmd));
    }
    println!("// -------");
}
