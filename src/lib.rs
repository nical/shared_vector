#![doc = include_str!("../README.md")]

mod raw;
mod shared;
mod unique;

use raw::AllocError;
pub use raw::{BufferSize};
pub use unique::UniqueVector;
pub use shared::{SharedVector, AtomicSharedVector, RefCountedVector};


pub trait ReferenceCount {
    type Header: raw::BufferHeader;
}

pub struct DefaultRefCount;
pub struct AtomicRefCount;

impl ReferenceCount for DefaultRefCount { type Header = raw::Header; }
impl ReferenceCount for AtomicRefCount { type Header = raw::AtomicHeader; }

pub(crate) fn grow_amortized(len: usize, additional: usize) -> usize {
    let required = len.saturating_add(additional);
    let cap = len.saturating_add(len).max(required).max(8);

    const MAX: usize = BufferSize::MAX as usize;

    if cap > MAX {
        if required <= MAX {
            return required;
        }

        panic!("Required allocation size is too large");
    }

    cap
}

#[macro_export]
macro_rules! vector {
    (@one $x:expr) => (1usize);
    ($elem:expr; $n:expr) => ({
        $crate::UniqueVector::from_elem($elem, $n)
    });
    ($($x:expr),*$(,)*) => ({
        let count = 0usize $(+ $crate::vector!(@one $x))*;
        let mut vec = $crate::UniqueVector::with_capacity(count);
        $(vec.push($x);)*
        vec
    });
}

#[macro_export]
macro_rules! rc_vector {
    ($elem:expr; $n:expr) => ({
        $crate::SharedVector::from_elem($elem, $n)
    });
    ($($x:expr),*$(,)*) => ({
        let count = 0usize $(+ $crate::vector!(@one $x))*;
        let mut vec = $crate::SharedVector::with_capacity(count);
        $(vec.push($x);)*
        vec
    });
}

#[macro_export]
macro_rules! arc_vector {
    ($elem:expr; $n:expr) => ({
        $crate::AtomicSharedVector::from_elem($elem, $n)
    });
    ($($x:expr),*$(,)*) => ({
        let count = 0usize $(+ $crate::vector!(@one $x))*;
        let mut vec = $crate::AtomicSharedVector::with_capacity(count);
        $(vec.push($x);)*
        vec
    });
}

#[test]
fn vector_macro() {
    let v1: UniqueVector<u32> = vector![0, 1, 2, 3, 4, 5];
    let v2: UniqueVector<u32> = vector![42, 10];
    let vec1: Vec<u32> = v1.iter().cloned().collect();
    let vec2: Vec<u32> = v2.iter().cloned().collect();
    assert_eq!(vec1, vec![0, 1, 2, 3, 4, 5]);
    assert_eq!(vec2, vec![42, 10]);


    let v1: SharedVector<u32> = rc_vector![0, 1, 2, 3, 4, 5];
    let v2: SharedVector<u32> = rc_vector![42, 10];
    let vec1: Vec<u32> = v1.iter().cloned().collect();
    let vec2: Vec<u32> = v2.iter().cloned().collect();
    assert_eq!(vec1, vec![0, 1, 2, 3, 4, 5]);
    assert_eq!(vec2, vec![42, 10]);

    let v1: AtomicSharedVector<u32> = arc_vector![0, 1, 2, 3, 4, 5];
    let v2: AtomicSharedVector<u32> = arc_vector![42, 10];
    let vec1: Vec<u32> = v1.iter().cloned().collect();
    let vec2: Vec<u32> = v2.iter().cloned().collect();
    assert_eq!(vec1, vec![0, 1, 2, 3, 4, 5]);
    assert_eq!(vec2, vec![42, 10]);
}
