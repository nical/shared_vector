use std::{ptr, mem};
use std::ops::{Index, IndexMut, Deref, DerefMut};
use std::ptr::NonNull;
use std::fmt::Debug;

use crate::raw::{self, AllocError, BufferSize, HeaderBuffer, VecHeader};
use crate::shared::{AtomicSharedVector, SharedVector};
use crate::grow_amortized;

/// A heap allocated, mutable contiguous buffer containing elements of type `T`.
///
/// <svg width="280" height="120" viewBox="0 0 74.08 31.75" xmlns:xlink="http://www.w3.org/1999/xlink" xmlns="http://www.w3.org/2000/svg"><defs><linearGradient id="a"><stop offset="0" stop-color="#491c9c"/><stop offset="1" stop-color="#d54b27"/></linearGradient><linearGradient xlink:href="#a" id="b" gradientUnits="userSpaceOnUse" x1="6.27" y1="34.86" x2="87.72" y2="13.24" gradientTransform="translate(-2.64 -18.48)"/></defs><rect width="10.57" height="10.66" x="7.94" y="18.45" ry="1.37" fill="#3dbdaa"/><circle cx="12.7" cy="18.48" r=".79" fill="#666"/><path d="M12.7 18.48c0-3.93 7.14-1.28 7.14-5.25" fill="none" stroke="#999" stroke-width=".86" stroke-linecap="round"/><rect width="68.79" height="10.58" x="2.65" y="2.68" ry="1.37" fill="url(#b)"/><rect width="15.35" height="9.51" x="3.18" y="3.21" ry=".9" fill="#78a2d4"/><rect width="9.26" height="9.51" x="19.85" y="3.2" ry=".9" fill="#eaa577"/><rect width="9.26" height="9.51" x="29.64" y="3.22" ry=".9" fill="#eaa577"/><rect width="9.26" height="9.51" x="39.43" y="3.22" ry=".9" fill="#eaa577"/><rect width="9.26" height="9.51" x="49.22" y="3.21" ry=".9" fill="#eaa577"/><circle cx="62.84" cy="7.97" r=".66" fill="#eaa577"/><circle cx="64.7" cy="7.97" r=".66" fill="#eaa577"/><circle cx="66.55" cy="7.97" r=".66" fill="#eaa577"/></svg>
///
/// Similar in principle to a `Vec<T>`.
/// It can be converted for free into an immutable `SharedVector<T>` or `AtomicSharedVector<T>`.
///
/// Unique and shared vectors expose similar functionality. `UniqueVector` takes advantage of
/// the guaranteed uniqueness at the type level to provide overall faster operations than its
/// shared counterparts, while its memory layout makes it very cheap to convert to a shared vector
/// (involving not allocation or copy).
///
/// # Internal representation
///
/// `UniqueVector` stores its length and capacity inline and points to the first element of the
/// allocated buffer. Room for a 16 bytes header is left before the first element so that the
/// vector can be converted into a `SharedVector` or `AtomicSharedVector` without reallocating
/// the storage.
pub struct UniqueVector<T> {
    pub(crate) data: NonNull<T>,
    pub(crate) len: BufferSize,
    pub(crate) cap: BufferSize,
}

impl<T> UniqueVector<T> {
    /// Creates an empty vector.
    ///
    /// This does not allocate memory.
    pub fn new() -> Self {
        UniqueVector {
            data: NonNull::dangling(),
            len: 0,
            cap: 0,
        }
    }

    /// Creates an empty pre-allocated vector with a given storage capacity.
    ///
    /// Does not allocate memory if `cap` is zero.
    pub fn with_capacity(cap: usize) -> Self {
        Self::try_with_capacity(cap).unwrap()
    }

    /// Creates an empty pre-allocated vector with a given storage capacity.
    ///
    /// Does not allocate memory if `cap` is zero.
    pub fn try_with_capacity(cap: usize) -> Result<Self, AllocError> {
        let inner: HeaderBuffer<raw::Header, T> = HeaderBuffer::try_with_capacity(cap)?;
        let cap = inner.capacity();
        let data = NonNull::new(inner.data_ptr()).unwrap();

        mem::forget(inner);

        Ok(UniqueVector { data, len: 0, cap })
    }

    pub fn from_slice(data: &[T]) -> Self
    where
        T: Clone,
    {
        let mut v = Self::new();
        v.push_slice(data);

        v
    }

    /// Creates a vector with `n` copies of `elem`.
    pub fn from_elem(elem: T, n: usize) -> Self
    where
        T: Clone,
    {
        if n == 0 {
            return Self::new();
        }

        let mut v = Self::with_capacity(n);
        for _ in 0..(n - 1) {
            v.push(elem.clone())
        }

        v.push(elem);

        v
    }

    #[inline]
    /// Returns `true` if the vector contains no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    /// Returns the number of elements in the vector, also referred to as its ‘length’.
    pub fn len(&self) -> usize {
        self.len as usize
    }

    #[inline]
    /// Returns the total number of elements the vector can hold without reallocating.
    pub fn capacity(&self) -> usize {
        self.cap as usize
    }

    /// Returns number of elements that can be added without reallocating.
    #[inline]
    pub fn remaining_capacity(&self) -> usize {
        (self.cap - self.len) as usize
    }

    #[inline]
    fn data_ptr(&self) -> *mut T {
        self.data.as_ptr()
    }

    unsafe fn write_header(&self) {
        debug_assert!(self.cap != 0);
        unsafe {
            let header = raw::header_from_data_ptr(self.data);
            ptr::write(header.as_ptr(), raw::Header {
                vec: VecHeader { len: self.len, cap: self.cap },
                ref_count: 1,
                _pad: 0,
            });
        }
    }

    /// Make this vector immutable.
    ///
    /// This operation is cheap, the underlying storage does not not need
    /// to be reallocated.
    #[inline]
    pub fn into_shared(self) -> SharedVector<T> {
        if self.cap == 0 {
            return SharedVector::new();
        }
        unsafe {
            self.write_header();
            let header = raw::header_from_data_ptr(self.data);
            let inner: HeaderBuffer<raw::Header, T> = HeaderBuffer::from_raw(header);
            mem::forget(self);
            SharedVector { inner }
        }
    }

    /// Make this vector immutable.
    ///
    /// This operation is cheap, the underlying storage does not not need
    /// to be reallocated.
    #[inline]
    pub fn into_shared_atomic(self) -> AtomicSharedVector<T> {
        if self.cap == 0 {
            return AtomicSharedVector::new();
        }
        unsafe {
            self.write_header();
            let header = raw::header_from_data_ptr(self.data);
            let inner: HeaderBuffer<raw::AtomicHeader, T> = HeaderBuffer::from_raw(header);
            mem::forget(self);
            AtomicSharedVector { inner }
        }
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.data_ptr(), self.len()) }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.data_ptr(), self.len()) }
    }

    #[inline]
    pub fn push(&mut self, val: T) {
        let len = self.len;
        let cap = self.cap;
        if cap == len {
            self.try_realloc_additional(1).unwrap();
        }

        unsafe {
            self.len += 1;
            let dst = self.data_ptr().add(len as usize);
            ptr::write(dst, val);
        }
    }

    /// Appends an element if there is sufficient spare capacity, otherwise an error is returned
    /// with the element.
    ///
    /// Unlike push this method will not reallocate when there’s insufficient capacity.
    /// The caller should use reserve or try_reserve to ensure that there is enough capacity.
    #[inline]
    pub fn push_within_capacity(&mut self, val: T) -> Result<(), T> {
        if self.len == self.cap {
            return Err(val);
        }

        unsafe {
            let dst = self.data_ptr().add(self.len as usize);
            self.len += 1;
            ptr::write(dst, val);
        }

        Ok(())
    }

    #[inline]
    /// Removes the last element from the vector and returns it, or `None` if it is empty.
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }

        self.len -= 1;

        unsafe {
            Some(ptr::read(self.data_ptr().add(self.len as usize)))
        }
    }

    /// Removes an element from the vector and returns it.
    ///
    /// The removed element is replaced by the last element of the vector.
    ///
    /// # Panics
    ///
    /// Panics if index is out of bounds.
    #[inline]
    pub fn swap_remove(&mut self, idx: usize) -> T {
        let len = self.len();
        assert!(idx < len);

        unsafe {
            let ptr = self.data_ptr().add(idx);
            let item = ptr::read(ptr);

            let last_idx = len - 1;
            if idx != last_idx {
                let last_ptr = self.data_ptr().add(last_idx);
                ptr::write(ptr, ptr::read(last_ptr));    
            }

            self.len -= 1;

            item
        }
    }

    pub fn push_slice(&mut self, data: &[T])
    where
        T: Clone,
    {
        self.extend(data.iter().cloned())
    }

    /// Moves all the elements of `other` into `self`, leaving `other` empty.
    pub fn append(&mut self, other: &mut Self) where T: Clone {
        if other.is_empty() {
            return;
        }

        self.reserve(other.len());

        unsafe {
            let src = other.data_ptr();
            let dst = self.data_ptr().add(self.len());
            ptr::copy_nonoverlapping(src, dst, other.len());
            self.len += other.len;
            other.len = 0
        }
    }

    pub fn extend(&mut self, data: impl IntoIterator<Item = T>) {
        let mut iter = data.into_iter();
        let (min, max) = iter.size_hint();
        self.reserve(max.unwrap_or(min));
        unsafe {
            self.extend_until_capacity(&mut iter);

            for item in iter {
                self.push(item);
            }
        }
    }

    unsafe fn extend_until_capacity(&mut self, iter: &mut impl Iterator<Item = T>) {
        let n = self.remaining_capacity() as BufferSize;

        let mut ptr = self.data_ptr().add(self.len());
        let mut count = 0;

        unsafe {
            for item in iter {
                if count == n {
                    break;
                }
                ptr::write(ptr, item);
                ptr = ptr.add(1);
                count += 1;
            }
            self.len += count;
        }
    }

    /// Clears the vector, removing all values.
    pub fn clear(&mut self) {
        unsafe {
            raw::drop_items(self.data_ptr(), self.len);
            self.len = 0;
        }
    }

    /// Allocate a clone of this buffer.
    pub fn clone_buffer(&self) -> Self
    where
        T: Clone,
    {
        self.clone_buffer_with_capacity(self.len())
    }

    /// Allocate a clone of this buffer with a different capacity
    ///
    /// The capacity must be at least as large as the buffer's length.
    pub fn clone_buffer_with_capacity(&self, cap: usize) -> Self
    where
        T: Clone,
    {
        let mut clone = Self::with_capacity(cap.max(self.len()));
        let len = self.len;

        unsafe {
            let mut src = self.data_ptr();
            let mut dst = clone.data_ptr();
            for _ in 0..len {
                ptr::write(dst, (*src).clone());
                src = src.add(1);
                dst = dst.add(1);
            }
        
            clone.len = len;
        }

        clone
    }

    // Note: Marking this #[inline(never)] is a pretty large regression in the push benchmark.
    #[cold]
    fn try_realloc_additional(&mut self, additional: usize) -> Result<(), AllocError> {
        let new_cap = grow_amortized(self.len(), additional);
        if new_cap < self.len() {
            return Err(AllocError::CapacityOverflow);
        }

        self.try_realloc_with_capacity(new_cap)
    }

    // Note: Marking this #[inline(never)] is a pretty large regression in the push benchmark.
    #[cold]
    fn try_realloc_with_capacity(&mut self, new_cap: usize) -> Result<(), AllocError> {
        let mut dst_buffer = Self::try_with_capacity(new_cap)?;

        if self.len > 0 {
            unsafe {
                let src = self.data_ptr();
                let dst = dst_buffer.data_ptr();
                let len = self.len;
    
                self.len = 0;
                dst_buffer.len = len;
    
                ptr::copy_nonoverlapping(src, dst, len as usize);
            }    
        }

        *self = dst_buffer;

        Ok(())
    }


    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        if self.remaining_capacity() < additional {
            self.try_realloc_additional(additional).unwrap();
        }
    }

    #[inline]
    pub fn try_reserve(&mut self, additional: usize) {
        if self.remaining_capacity() < additional {
            self.try_realloc_additional(additional).unwrap();
        }
    }

    /// Reserves the minimum capacity for at least `additional` elements to be inserted in the given vector.
    ///
    /// Unlike `reserve`, this will not deliberately over-allocate to speculatively avoid frequent allocations.
    /// After calling `try_reserve_exact`, capacity will be greater than or equal to `self.len() + additional` if
    /// it returns `Ok(())`.
    /// This will also allocate if the vector is not unique.
    /// Does nothing if the capacity is already sufficient and the vector is unique.
    ///
    /// Note that the allocator may give the collection more space than it requests. Therefore, capacity can not
    /// be relied upon to be precisely minimal. Prefer `try_reserve` if future insertions are expected.
    pub fn reserve_exact(&mut self, additional: usize)
    where
        T: Clone,
    {
        self.try_reserve_exact(additional).unwrap();
    }

    /// Tries to reserve the minimum capacity for at least `additional` elements to be inserted in the given vector.
    ///
    /// Unlike `try_reserve`, this will not deliberately over-allocate to speculatively avoid frequent allocations.
    /// After calling `reserve_exact`, capacity will be greater than or equal to `self.len() + additional`.
    /// This will also allocate if the vector is not unique.
    /// Does nothing if the capacity is already sufficient and the vector is unique.
    ///
    /// Note that the allocator may give the collection more space than it requests. Therefore, capacity can not
    /// be relied upon to be precisely minimal. Prefer `try_reserve` if future insertions are expected.
    pub fn try_reserve_exact(&mut self, additional: usize) -> Result<(), AllocError> {
        if self.remaining_capacity() >= additional {
            return Ok(())
        }

        self.try_realloc_with_capacity(self.len() + additional)
    }

    /// Shrinks the capacity of the vector with a lower bound.
    ///
    /// The capacity will remain at least as large as both the length and the supplied value.
    /// If the current capacity is less than the lower limit, this is a no-op.
    pub fn shrink_to(&mut self, min_capacity: usize) where T: Clone {
        let min_capacity = min_capacity.max(self.len());
        if self.capacity() <= min_capacity {
            return;
        }

        self.try_realloc_with_capacity(min_capacity).unwrap();
    }

    /// Shrinks the capacity of the vector as much as possible.
    pub fn shrink_to_fit(&mut self) where T: Clone {
        self.shrink_to(self.len())
    }

    pub fn take(&mut self) -> Self {
        mem::take(self)
    }
}

impl<T> Drop for UniqueVector<T> {
    fn drop(&mut self) {
        if self.cap == 0 {
            return;
        }

        self.clear();

        unsafe {
            let ptr = raw::header_from_data_ptr::<raw::Header, T>(self.data);
            raw::dealloc::<raw::Header, T>(ptr, self.cap);
        }
    }
}

impl<T: Clone> Clone for UniqueVector<T> {
    fn clone(&self) -> Self {
        self.clone_buffer()
    }
}

impl<T: PartialEq<T>> PartialEq<UniqueVector<T>> for UniqueVector<T> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: PartialEq<T>> PartialEq<&[T]> for UniqueVector<T> {
    fn eq(&self, other: &&[T]) -> bool {
        self.as_slice() == *other
    }
}

impl<T> AsRef<[T]> for UniqueVector<T> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> AsMut<[T]> for UniqueVector<T> {
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T> Default for UniqueVector<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, T> IntoIterator for &'a UniqueVector<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;
    fn into_iter(self) -> std::slice::Iter<'a, T> {
        self.as_slice().iter()
    }
}

impl<'a, T> IntoIterator for &'a mut UniqueVector<T> {
    type Item = &'a mut T;
    type IntoIter = std::slice::IterMut<'a, T>;
    fn into_iter(self) -> std::slice::IterMut<'a, T> {
        self.as_mut_slice().iter_mut()
    }
}

impl<T, I> Index<I> for UniqueVector<T>
where
    I: std::slice::SliceIndex<[T]>,
{
    type Output = <I as std::slice::SliceIndex<[T]>>::Output;
    fn index(&self, index: I) -> &Self::Output {
        self.as_slice().index(index)
    }
}

impl<T, I> IndexMut<I> for UniqueVector<T>
where
    I: std::slice::SliceIndex<[T]>,
{
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        self.as_mut_slice().index_mut(index)
    }
}

impl<T> Deref for UniqueVector<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> DerefMut for UniqueVector<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T: Debug> Debug for UniqueVector<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        self.as_slice().fmt(f)
    }
}

#[test]
fn ensure_unique_empty() {
    let mut v: SharedVector<u32> = SharedVector::new();
    v.ensure_unique();
}
