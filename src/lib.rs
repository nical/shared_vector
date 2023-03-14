#![cfg_attr(feature = "nightly", feature(allocator_api))]
#![doc = include_str!("../README.md")]

mod raw;
mod shared;
mod unique;

pub use raw::{BufferSize, RefCount, AtomicRefCount, DefaultRefCount};
pub use unique::Vector;
pub use shared::{SharedVector, AtomicSharedVector, RefCountedVector};

pub mod alloc {
    pub use allocator_api2::alloc::{Allocator, Global, AllocError};
}

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
    (@one@ $x:expr) => (1usize);
    ($elem:expr; $n:expr) => ({
        $crate::Vector::from_elem($elem, $n)
    });
    ($($x:expr),*$(,)*) => ({
        let count = 0usize $(+ $crate::vector!(@one@ $x))*;
        let mut vec = $crate::Vector::with_capacity(count);
        $(vec.push($x);)*
        vec
    });
    ([$($x:expr),*$(,)*] in $allocator:expr) => ({
        let count = 0usize $(+ $crate::vector!(@one@ $x))*;
        let mut vec = $crate::Vector::try_with_capacity_in(count, $allocator).unwrap();
        $(vec.push($x);)*
        vec
    });
    ([$x:expr;$n:expr] in $allocator:expr) => ({
        let mut vec = $crate::Vector::try_with_capacity_in($n, $allocator).unwrap();
        for _ in 0..$n { vec.push($x.clone()); }
        vec
    });
}

#[macro_export]
macro_rules! rc_vector {
    ($elem:expr; $n:expr) => ({
        let mut vec = $crate::SharedVector::with_capacity($n);
        for _ in 0..$n { vec.push($elem.clone()); }
        vec
    });
    ($($x:expr),*$(,)*) => ({
        let count = 0usize $(+ $crate::vector!(@one@ $x))*;
        let mut vec = $crate::SharedVector::with_capacity(count);
        $(vec.push($x);)*
        vec
    });
    ([$($x:expr),*$(,)*] in $allocator:expr) => ({
        let count = 0usize $(+ $crate::vector!(@one@ $x))*;
        let mut vec = $crate::SharedVector::try_with_capacity_in(count, $allocator).unwrap();
        $(vec.push($x);)*
        vec
    });
    ([$elem:expr;$n:expr] in $allocator:expr) => ({
        let mut vec = $crate::SharedVector::try_with_capacity_in($n, $allocator).unwrap();
        for _ in 0..$n { vec.push($elem.clone()); }
        vec
    });
}

#[macro_export]
macro_rules! arc_vector {
    ($elem:expr; $n:expr) => ({
        let mut vec = $crate::AtomicSharedVector::with_capacity($n);
        for _ in 0..$n { vec.push($elem.clone()); }
        vec
    });
    ($($x:expr),*$(,)*) => ({
        let count = 0usize $(+ $crate::vector!(@one@ $x))*;
        let mut vec = $crate::AtomicSharedVector::with_capacity(count);
        $(vec.push($x);)*
        vec
    });
    ([$($x:expr),*$(,)*] in $allocator:expr) => ({
        let count = 0usize $(+ $crate::vector!(@one@ $x))*;
        let mut vec = $crate::AtomicSharedVector::try_with_capacity_in(count, $allocator).unwrap();
        $(vec.push($x);)*
        vec
    });
    ([$elem:expr;$n:expr] in $allocator:expr) => ({
        let mut vec = $crate::AtomicSharedVector::try_with_capacity_in($n, $allocator).unwrap();
        for _ in 0..$n { vec.push($elem.clone()); }
        vec
    });
}

#[test]
fn vector_macro() {
    pub use allocator_api2::alloc::{Allocator, Global};

    let v1: Vector<u32> = vector![0, 1, 2, 3, 4, 5];
    let v2: Vector<u32> = vector![2; 4];
    let v3: Vector<u32> = vector!([6, 7] in Global);
    assert_eq!(v1.as_slice(), &[0, 1, 2, 3, 4, 5]);
    assert_eq!(v2.as_slice(), &[2, 2, 2, 2]);
    assert_eq!(v3.as_slice(), &[6, 7]);


    let v1: SharedVector<u32> = rc_vector![0, 1, 2, 3, 4, 5];
    let v2: SharedVector<u32> = rc_vector![3; 5];
    let v3: SharedVector<u32> = rc_vector!([4; 3] in Global);
    assert_eq!(v1.as_slice(), &[0, 1, 2, 3, 4, 5]);
    assert_eq!(v2.as_slice(), &[3, 3, 3, 3, 3]);
    assert_eq!(v3.as_slice(), &[4, 4, 4]);

    let v1: AtomicSharedVector<u32> = arc_vector![0, 1, 2, 3, 4, 5];
    let v2: AtomicSharedVector<u32> = arc_vector![1; 4];
    let v3: AtomicSharedVector<u32> = arc_vector![[3, 2, 1] in Global];
    assert_eq!(v1.as_slice(), &[0, 1, 2, 3, 4, 5]);
    assert_eq!(v2.as_slice(), &[1, 1, 1, 1]);
    assert_eq!(v3.as_slice(), &[3, 2, 1]);
}
