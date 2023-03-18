use core::fmt::Debug;
use core::ops::{Deref, DerefMut, Index, IndexMut};
use core::ptr::NonNull;
use core::{mem, ptr};

use crate::alloc::{AllocError, Allocator, Global};
use crate::raw::{
    self, buffer_layout, AtomicRefCount, BufferSize, Header, HeaderBuffer, RefCount, VecHeader,
};
use crate::shared::{AtomicSharedVector, SharedVector};
use crate::{grow_amortized, DefaultRefCount};

/// A heap allocated, mutable contiguous buffer containing elements of type `T`.
///
/// <svg width="280" height="120" viewBox="0 0 74.08 31.75" xmlns:xlink="http://www.w3.org/1999/xlink" xmlns="http://www.w3.org/2000/svg"><defs><linearGradient id="a"><stop offset="0" stop-color="#491c9c"/><stop offset="1" stop-color="#d54b27"/></linearGradient><linearGradient xlink:href="#a" id="b" gradientUnits="userSpaceOnUse" x1="6.27" y1="34.86" x2="87.72" y2="13.24" gradientTransform="translate(-2.64 -18.48)"/></defs><rect width="10.57" height="10.66" x="7.94" y="18.45" ry="1.37" fill="#3dbdaa"/><circle cx="12.7" cy="18.48" r=".79" fill="#666"/><path d="M12.7 18.48c0-3.93 7.14-1.28 7.14-5.25" fill="none" stroke="#999" stroke-width=".86" stroke-linecap="round"/><rect width="68.79" height="10.58" x="2.65" y="2.68" ry="1.37" fill="url(#b)"/><rect width="15.35" height="9.51" x="3.18" y="3.21" ry=".9" fill="#78a2d4"/><rect width="9.26" height="9.51" x="19.85" y="3.2" ry=".9" fill="#eaa577"/><rect width="9.26" height="9.51" x="29.64" y="3.22" ry=".9" fill="#eaa577"/><rect width="9.26" height="9.51" x="39.43" y="3.22" ry=".9" fill="#eaa577"/><rect width="9.26" height="9.51" x="49.22" y="3.21" ry=".9" fill="#eaa577"/><circle cx="62.84" cy="7.97" r=".66" fill="#eaa577"/><circle cx="64.7" cy="7.97" r=".66" fill="#eaa577"/><circle cx="66.55" cy="7.97" r=".66" fill="#eaa577"/></svg>
///
/// Similar in principle to a `Vec<T>`.
/// It can be converted for free into a reference counted `SharedVector<T>` or `AtomicSharedVector<T>`.
///
/// Unique and shared vectors expose similar functionality. `Vector` takes advantage of
/// the guaranteed uniqueness at the type level to provide overall faster operations than its
/// shared counterparts, while its memory layout makes it very cheap to convert to a shared vector
/// (involving no allocation or copy).
///
/// # Internal representation
///
/// `Vector` stores its length and capacity inline and points to the first element of the
/// allocated buffer. Room for a header is left uninitialized before the elements so that the
/// vector can be converted into a `SharedVector` or `AtomicSharedVector` without reallocating
/// the storage.
pub struct Vector<T, A: Allocator = Global> {
    pub(crate) data: NonNull<T>,
    pub(crate) len: BufferSize,
    pub(crate) cap: BufferSize,
    pub(crate) allocator: A,
}

impl<T> Vector<T, Global> {
    /// Creates an empty vector.
    ///
    /// This does not allocate memory.
    pub fn new() -> Vector<T, Global> {
        Vector {
            data: NonNull::dangling(),
            len: 0,
            cap: 0,
            allocator: Global,
        }
    }

    /// Creates an empty pre-allocated vector with a given storage capacity.
    ///
    /// Does not allocate memory if `cap` is zero.
    pub fn with_capacity(cap: usize) -> Vector<T, Global> {
        Self::try_with_capacity(cap).unwrap()
    }

    /// Creates an empty pre-allocated vector with a given storage capacity.
    ///
    /// Does not allocate memory if `cap` is zero.
    pub fn try_with_capacity(cap: usize) -> Result<Vector<T, Global>, AllocError> {
        Vector::try_with_capacity_in(cap, Global)
    }

    pub fn from_slice(data: &[T]) -> Self
    where
        T: Clone,
    {
        let mut v = Self::with_capacity(data.len());
        v.push_slice(data);

        v
    }

    /// Creates a vector with `n` copies of `elem`.
    pub fn from_elem(elem: T, n: usize) -> Vector<T, Global>
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
}

impl<T, A: Allocator> Vector<T, A> {
    /// Creates an empty vector without allocating memory.
    pub fn new_in(allocator: A) -> Self {
        Self::try_with_capacity_in(0, allocator).unwrap()
    }

    /// Creates an empty pre-allocated vector with a given storage capacity.
    ///
    /// Does not allocate memory if `cap` is zero.
    pub fn with_capacity_in(cap: usize, allocator: A) -> Self {
        Self::try_with_capacity_in(cap, allocator).unwrap()
    }

    /// Creates an empty pre-allocated vector with a given storage capacity.
    ///
    /// Does not allocate memory if `cap` is zero.
    pub fn try_with_capacity_in(cap: usize, allocator: A) -> Result<Vector<T, A>, AllocError> {
        if cap == 0 {
            return Ok(Vector {
                data: NonNull::dangling(),
                len: 0,
                cap: 0,
                allocator,
            });
        }

        unsafe {
            let (base_ptr, cap) = raw::allocate_header_buffer::<T, A>(cap, &allocator)?;
            let data = NonNull::new_unchecked(raw::data_ptr::<raw::Header<DefaultRefCount, A>, T>(
                base_ptr.cast(),
            ));
            Ok(Vector {
                data,
                len: 0,
                cap: cap as BufferSize,
                allocator,
            })
        }
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

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        unsafe { core::slice::from_raw_parts(self.data_ptr(), self.len()) }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { core::slice::from_raw_parts_mut(self.data_ptr(), self.len()) }
    }

    /// Clears the vector, removing all values.
    pub fn clear(&mut self) {
        unsafe {
            raw::drop_items(self.data_ptr(), self.len);
            self.len = 0;
        }
    }

    unsafe fn base_ptr(&self) -> NonNull<u8> {
        debug_assert!(self.cap > 0);
        raw::header_from_data_ptr::<Header<DefaultRefCount, A>, T>(self.data).cast()
    }

    unsafe fn into_header_buffer<R>(mut self) -> HeaderBuffer<T, R, A>
    where
        R: RefCount,
    {
        debug_assert!(self.cap != 0);
        unsafe {
            let mut header = raw::header_from_data_ptr(self.data);

            *header.as_mut() = raw::Header {
                vec: VecHeader {
                    len: self.len,
                    cap: self.cap,
                },
                ref_count: R::new(1),
                allocator: ptr::read(&mut self.allocator),
            };

            mem::forget(self);

            HeaderBuffer::from_raw(header)
        }
    }

    /// Make this vector immutable.
    ///
    /// This operation is cheap, the underlying storage does not not need
    /// to be reallocated.
    #[inline]
    pub fn into_shared(self) -> SharedVector<T, A>
    where
        A: Allocator + Clone,
    {
        if self.cap == 0 {
            return SharedVector::try_with_capacity_in(0, self.allocator.clone()).unwrap();
        }
        unsafe {
            let inner = self.into_header_buffer::<DefaultRefCount>();
            SharedVector { inner }
        }
    }

    /// Make this vector immutable.
    ///
    /// This operation is cheap, the underlying storage does not not need
    /// to be reallocated.
    #[inline]
    pub fn into_shared_atomic(self) -> AtomicSharedVector<T, A>
    where
        A: Allocator + Clone,
    {
        if self.cap == 0 {
            return AtomicSharedVector::try_with_capacity_in(0, self.allocator.clone()).unwrap();
        }
        unsafe {
            let inner = self.into_header_buffer::<AtomicRefCount>();
            AtomicSharedVector { inner }
        }
    }

    /// Appends an element to the back of a collection.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity exceeds `u32::MAX` bytes.
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

    /// Removes the last element from the vector and returns it, or `None` if it is empty.
    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }

        self.len -= 1;

        unsafe { Some(ptr::read(self.data_ptr().add(self.len as usize))) }
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

    /// Clones and appends the contents of the slice to the back of a collection.
    pub fn push_slice(&mut self, data: &[T])
    where
        T: Clone,
    {
        self.extend(data.iter().cloned())
    }

    /// Moves all the elements of `other` into `self`, leaving `other` empty.
    pub fn append(&mut self, other: &mut Self)
    where
        T: Clone,
    {
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

    /// Appends the contents of an iterator to the back of a collection.
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

    /// Allocate a clone of this buffer.
    pub fn clone_buffer(&self) -> Self
    where
        T: Clone,
        A: Clone,
    {
        self.clone_buffer_with_capacity(self.len())
    }

    /// Allocate a clone of this buffer with a different capacity
    ///
    /// The capacity must be at least as large as the buffer's length.
    pub fn clone_buffer_with_capacity(&self, cap: usize) -> Self
    where
        T: Clone,
        A: Clone,
    {
        let mut clone =
            Self::try_with_capacity_in(cap.max(self.len()), self.allocator.clone()).unwrap();
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
            return Err(AllocError);
        }

        self.try_realloc_with_capacity(new_cap)
    }

    #[cold]
    fn try_realloc_with_capacity(&mut self, new_cap: usize) -> Result<(), AllocError> {
        type R = DefaultRefCount;

        unsafe {
            if new_cap == 0 {
                self.deallocate();
            }

            let new_layout = buffer_layout::<Header<R, A>, T>(new_cap).unwrap();

            let new_alloc = if self.cap == 0 {
                self.allocator.allocate(new_layout)?
            } else {
                let old_cap = self.capacity();
                let old_ptr = self.base_ptr();
                let old_layout = buffer_layout::<Header<R, A>, T>(old_cap).unwrap();
                let new_layout = buffer_layout::<Header<R, A>, T>(new_cap).unwrap();

                if new_layout.size() >= old_layout.size() {
                    self.allocator.grow(old_ptr, old_layout, new_layout)
                } else {
                    self.allocator.shrink(old_ptr, old_layout, new_layout)
                }?
            };

            let new_data_ptr = crate::raw::data_ptr::<Header<R, A>, T>(new_alloc.cast());
            self.data = NonNull::new_unchecked(new_data_ptr);
            self.cap = new_cap as u32;
        }

        Ok(())
    }

    // Deallocates the memory, does not drop the vector's content.
    unsafe fn deallocate(&mut self) {
        let layout = buffer_layout::<Header<DefaultRefCount, A>, T>(self.capacity()).unwrap();
        let ptr = self.base_ptr();

        self.allocator.deallocate(ptr, layout);

        self.cap = 0;
        self.len = 0;
        self.data = NonNull::dangling();
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
            return Ok(());
        }

        self.try_realloc_with_capacity(self.len() + additional)
    }

    /// Shrinks the capacity of the vector with a lower bound.
    ///
    /// The capacity will remain at least as large as both the length and the supplied value.
    /// If the current capacity is less than the lower limit, this is a no-op.
    pub fn shrink_to(&mut self, min_capacity: usize)
    where
        T: Clone,
    {
        let min_capacity = min_capacity.max(self.len());
        if self.capacity() <= min_capacity {
            return;
        }

        self.try_realloc_with_capacity(min_capacity).unwrap();
    }

    /// Shrinks the capacity of the vector as much as possible.
    pub fn shrink_to_fit(&mut self)
    where
        T: Clone,
    {
        self.shrink_to(self.len())
    }

    pub fn take(&mut self) -> Self
    where
        A: Clone,
    {
        let other = Vector {
            data: self.data,
            len: self.len,
            cap: self.cap,
            allocator: self.allocator.clone(),
        };

        self.data = NonNull::dangling();
        self.len = 0;
        self.cap = 0;

        other
    }
}

impl<T, A: Allocator> Drop for Vector<T, A> {
    fn drop(&mut self) {
        if self.cap == 0 {
            return;
        }

        self.clear();

        unsafe {
            self.deallocate();
        }
    }
}

impl<T: Clone, A: Allocator + Clone> Clone for Vector<T, A> {
    fn clone(&self) -> Self {
        self.clone_buffer()
    }
}

impl<T: PartialEq<T>, A: Allocator + Clone> PartialEq<Vector<T, A>> for Vector<T, A> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: PartialEq<T>, A: Allocator + Clone> PartialEq<&[T]> for Vector<T, A> {
    fn eq(&self, other: &&[T]) -> bool {
        self.as_slice() == *other
    }
}

impl<T, A: Allocator + Clone> AsRef<[T]> for Vector<T, A> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T, A: Allocator + Clone> AsMut<[T]> for Vector<T, A> {
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T> Default for Vector<T, Global> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, T, A: Allocator + Clone> IntoIterator for &'a Vector<T, A> {
    type Item = &'a T;
    type IntoIter = core::slice::Iter<'a, T>;
    fn into_iter(self) -> core::slice::Iter<'a, T> {
        self.as_slice().iter()
    }
}

impl<'a, T, A: Allocator + Clone> IntoIterator for &'a mut Vector<T, A> {
    type Item = &'a mut T;
    type IntoIter = core::slice::IterMut<'a, T>;
    fn into_iter(self) -> core::slice::IterMut<'a, T> {
        self.as_mut_slice().iter_mut()
    }
}

impl<T, A: Allocator + Clone, I> Index<I> for Vector<T, A>
where
    I: core::slice::SliceIndex<[T]>,
{
    type Output = <I as core::slice::SliceIndex<[T]>>::Output;
    fn index(&self, index: I) -> &Self::Output {
        self.as_slice().index(index)
    }
}

impl<T, A: Allocator + Clone, I> IndexMut<I> for Vector<T, A>
where
    I: core::slice::SliceIndex<[T]>,
{
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        self.as_mut_slice().index_mut(index)
    }
}

impl<T, A: Allocator + Clone> Deref for Vector<T, A> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T, A: Allocator + Clone> DerefMut for Vector<T, A> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T: Debug, A: Allocator + Clone> Debug for Vector<T, A> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> Result<(), core::fmt::Error> {
        self.as_slice().fmt(f)
    }
}

impl<T: Clone, A: Allocator + Clone> From<SharedVector<T, A>> for Vector<T, A> {
    fn from(shared: SharedVector<T, A>) -> Self {
        shared.into_unique()
    }
}

impl<T: Clone, A: Allocator + Clone> From<AtomicSharedVector<T, A>> for Vector<T, A> {
    fn from(shared: AtomicSharedVector<T, A>) -> Self {
        shared.into_unique()
    }
}

impl<T: core::hash::Hash, A: Allocator> core::hash::Hash for Vector<T, A> {
    fn hash<H>(&self, state: &mut H) where H: core::hash::Hasher {
        self.as_slice().hash(state)
    }
}

#[test]
fn bump_alloc() {
    use blink_alloc::BlinkAlloc;

    let allocator = BlinkAlloc::new();

    {
        let mut v1: Vector<u32, &BlinkAlloc> = Vector::try_with_capacity_in(4, &allocator).unwrap();
        v1.push(0);
        v1.push(1);
        v1.push(2);
        assert_eq!(v1.capacity(), 4);
        assert_eq!(v1.as_slice(), &[0, 1, 2]);

        // let mut v2 = crate::vector!(@ &allocator [10, 11]);
        let mut v2 = crate::vector!([10, 11] in &allocator);
        assert_eq!(v2.capacity(), 2);

        assert_eq!(v2.as_slice(), &[10, 11]);

        v1.push(3);
        v1.push(4);

        assert_eq!(v1.as_slice(), &[0, 1, 2, 3, 4]);

        assert!(v1.capacity() > 4);

        v2.push(12);
        v2.push(13);
        v2.push(14);

        let v2 = v2.into_shared();

        assert_eq!(v1.as_slice(), &[0, 1, 2, 3, 4]);
        assert_eq!(v2.as_slice(), &[10, 11, 12, 13, 14]);
    }
}

#[test]
fn basic_unique() {
    fn num(val: u32) -> Box<u32> {
        Box::new(val)
    }

    let mut a = Vector::with_capacity(256);

    a.push(num(0));
    a.push(num(1));
    a.push(num(2));

    let a = a.into_shared();

    assert_eq!(a.len(), 3);

    assert_eq!(a.as_slice(), &[num(0), num(1), num(2)]);

    assert!(a.is_unique());

    let b = Vector::from_slice(&[num(0), num(1), num(2), num(3), num(4)]);

    assert_eq!(b.as_slice(), &[num(0), num(1), num(2), num(3), num(4)]);

    let c = a.clone_buffer();
    assert!(!c.ptr_eq(&a));

    let a2 = a.new_ref();
    assert!(a2.ptr_eq(&a));
    assert!(!a.is_unique());
    assert!(!a2.is_unique());

    mem::drop(a2);

    assert!(a.is_unique());

    let _ = c.clone_buffer();
    let _ = b.clone_buffer();

    let mut d = Vector::with_capacity(64);
    d.push_slice(&[num(0), num(1), num(2)]);
    d.push_slice(&[]);
    d.push_slice(&[num(3), num(4)]);

    assert_eq!(d.as_slice(), &[num(0), num(1), num(2), num(3), num(4)]);
}

#[test]
fn shrink() {
    let mut v: Vector<u32> = Vector::with_capacity(32);
    v.shrink_to(8);
}

#[test]
fn zst() {
    let mut v = Vector::new();
    v.push(());
    v.push(());
    v.push(());
    v.push(());

    assert_eq!(v.len(), 4);
}
