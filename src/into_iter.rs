use core::fmt::Debug;
use core::iter::FusedIterator;
use core::{mem, ptr};

use crate::alloc::{Allocator, Global};
use crate::vector::{RawVector, Vector};

/// An iterator that moves out of a vector.
///
/// This `struct` is created by the `into_iter` method on [`Vector`]
/// (provided by the [`IntoIterator`] trait).
pub struct IntoIter<T, A: Allocator = Global> {
    // The elements in the `start..end` range are still owned by the iterator,
    // the rest of the buffer has been moved out. `raw`'s length is set to zero
    // so that it never drops items on its own.
    raw: RawVector<T>,
    allocator: A,
    start: usize,
    end: usize,
}

impl<T, A: Allocator> IntoIter<T, A> {
    pub(crate) fn new(mut vector: Vector<T, A>) -> Self {
        unsafe {
            let end = vector.len();
            let allocator = ptr::read(&vector.allocator);
            let mut raw = mem::take(&mut vector.raw);
            raw.header.len = 0;

            // The allocator was moved out of the vector, so it must not be dropped.
            mem::forget(vector);

            IntoIter {
                raw,
                allocator,
                start: 0,
                end,
            }
        }
    }

    /// Returns the remaining items of this iterator as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        unsafe { core::slice::from_raw_parts(self.ptr(self.start), self.len()) }
    }

    /// Returns the remaining items of this iterator as a mutable slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { core::slice::from_raw_parts_mut(self.ptr(self.start), self.len()) }
    }

    /// Returns a reference to the underlying allocator.
    #[inline]
    pub fn allocator(&self) -> &A {
        &self.allocator
    }

    #[inline]
    fn len(&self) -> usize {
        self.end - self.start
    }

    #[inline]
    unsafe fn ptr(&self, idx: usize) -> *mut T {
        self.raw.data_ptr().add(idx)
    }
}

impl<T, A: Allocator> Iterator for IntoIter<T, A> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        if self.start == self.end {
            return None;
        }

        unsafe {
            let item = ptr::read(self.ptr(self.start));
            self.start += 1;

            Some(item)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }
}

impl<T, A: Allocator> DoubleEndedIterator for IntoIter<T, A> {
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        if self.start == self.end {
            return None;
        }

        unsafe {
            self.end -= 1;

            Some(ptr::read(self.ptr(self.end)))
        }
    }
}

impl<T, A: Allocator> ExactSizeIterator for IntoIter<T, A> {
    #[inline]
    fn len(&self) -> usize {
        IntoIter::len(self)
    }
}

impl<T, A: Allocator> FusedIterator for IntoIter<T, A> {}

impl<T, A: Allocator> AsRef<[T]> for IntoIter<T, A> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T: Debug, A: Allocator> Debug for IntoIter<T, A> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> Result<(), core::fmt::Error> {
        f.debug_tuple("IntoIter").field(&self.as_slice()).finish()
    }
}

unsafe impl<T: Send, A: Allocator + Send> Send for IntoIter<T, A> {}
unsafe impl<T: Sync, A: Allocator + Sync> Sync for IntoIter<T, A> {}

impl<T, A: Allocator> Drop for IntoIter<T, A> {
    fn drop(&mut self) {
        unsafe {
            ptr::drop_in_place(self.as_mut_slice());
            // The raw vector's length is zero, so this only deallocates the buffer.
            self.raw.deallocate(&self.allocator);
        }
    }
}

#[test]
fn into_iter() {
    fn num(val: u32) -> Box<u32> {
        Box::new(val)
    }

    let mut v = Vector::new();
    v.push(num(0));
    v.push(num(1));
    v.push(num(2));

    let mut iter = v.into_iter();
    assert_eq!(iter.len(), 3);
    assert_eq!(iter.as_slice(), &[num(0), num(1), num(2)]);
    assert_eq!(iter.next(), Some(num(0)));
    assert_eq!(iter.next_back(), Some(num(2)));
    assert_eq!(iter.as_slice(), &[num(1)]);
    assert_eq!(iter.next(), Some(num(1)));
    assert_eq!(iter.next(), None);
    assert_eq!(iter.next_back(), None);

    // The remaining items are dropped with the iterator.
    let mut v = Vector::new();
    v.push(num(0));
    v.push(num(1));
    let mut iter = v.into_iter();
    assert_eq!(iter.next(), Some(num(0)));
    mem::drop(iter);

    // Empty and unallocated vectors.
    let v: Vector<Box<u32>> = Vector::new();
    assert_eq!(v.into_iter().next(), None);
    let v: Vector<Box<u32>> = Vector::with_capacity(16);
    assert_eq!(v.into_iter().count(), 0);

    // Zero-sized types.
    let mut v = Vector::new();
    v.push(());
    v.push(());
    assert_eq!(v.into_iter().collect::<std::vec::Vec<()>>().len(), 2);

    // In a for loop, with a non-global allocator.
    let mut v: Vector<Box<u32>, &Global> = Vector::new_in(&Global);
    v.push(num(1));
    v.push(num(2));
    let mut sum = 0;
    for item in v {
        sum += *item;
    }
    assert_eq!(sum, 3);
}
