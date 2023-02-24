use std::{ptr, mem};
use std::ops::{Index, IndexMut, Deref, DerefMut};
use std::ptr::NonNull;
use std::fmt::Debug;

use crate::unique::UniqueVector;
use crate::raw::{BufferSize, HeaderBuffer};
use crate::alloc::{AllocError, GlobalAllocator, Allocator};
use crate::{RefCount, AtomicRefCount, DefaultRefCount, grow_amortized};

/// A heap allocated, atomically reference counted, immutable contiguous buffer containing elements of type `T`.
///
/// <svg width="280" height="120" viewBox="0 0 74.08 31.75" xmlns:xlink="http://www.w3.org/1999/xlink" xmlns="http://www.w3.org/2000/svg"><defs><linearGradient id="a"><stop offset="0" stop-color="#491c9c"/><stop offset="1" stop-color="#d54b27"/></linearGradient><linearGradient xlink:href="#a" id="b" gradientUnits="userSpaceOnUse" x1="6.27" y1="34.86" x2="87.72" y2="13.24" gradientTransform="translate(-2.64 -18.48)"/></defs><rect width="10.57" height="10.66" x="2.66" y="18.48" ry="1.37" fill="#3dbdaa"/><rect width="10.57" height="10.66" x="15.88" y="18.52" ry="1.37" fill="#3dbdaa"/><rect width="10.57" height="10.66" x="29.11" y="18.52" ry="1.37" fill="#3dbdaa"/><circle cx="33.87" cy="18.56" r=".79" fill="#666"/><circle cx="7.41" cy="18.56" r=".79" fill="#666"/><circle cx="20.64" cy="18.56" r=".79" fill="#666"/><path d="M7.38 18.54c.03-2.63-3.41-2.66-3.41-5.31" fill="none" stroke="#999" stroke-width=".86" stroke-linecap="round"/><path d="M20.64 18.56c0-2.91-15.35-1.36-15.35-5.33" fill="none" stroke="#999" stroke-width=".86" stroke-linecap="round"/><path d="M33.87 18.56c0-3.97-27.26-2.68-27.26-5.33" fill="none" stroke="#999" stroke-width=".86" stroke-linecap="round"/><rect width="68.79" height="10.58" x="2.65" y="2.68" ry="1.37" fill="url(#b)"/><rect width="15.35" height="9.51" x="3.18" y="3.21" ry=".9" fill="#78a2d4"/><rect width="9.26" height="9.51" x="19.85" y="3.2" ry=".9" fill="#eaa577"/><rect width="9.26" height="9.51" x="29.64" y="3.22" ry=".9" fill="#eaa577"/><rect width="9.26" height="9.51" x="39.43" y="3.22" ry=".9" fill="#eaa577"/><rect width="9.26" height="9.51" x="49.22" y="3.21" ry=".9" fill="#eaa577"/><circle cx="62.84" cy="7.97" r=".66" fill="#eaa577"/><circle cx="64.7" cy="7.97" r=".66" fill="#eaa577"/><circle cx="66.55" cy="7.97" r=".66" fill="#eaa577"/></svg>
///
/// See [RefCountedVector].
pub type AtomicSharedVector<T, A = GlobalAllocator> = RefCountedVector<T, AtomicRefCount, A>;

/// A heap allocated, reference counted, immutable contiguous buffer containing elements of type `T`.
///
/// <svg width="280" height="120" viewBox="0 0 74.08 31.75" xmlns:xlink="http://www.w3.org/1999/xlink" xmlns="http://www.w3.org/2000/svg"><defs><linearGradient id="a"><stop offset="0" stop-color="#491c9c"/><stop offset="1" stop-color="#d54b27"/></linearGradient><linearGradient xlink:href="#a" id="b" gradientUnits="userSpaceOnUse" x1="6.27" y1="34.86" x2="87.72" y2="13.24" gradientTransform="translate(-2.64 -18.48)"/></defs><rect width="10.57" height="10.66" x="2.66" y="18.48" ry="1.37" fill="#3dbdaa"/><rect width="10.57" height="10.66" x="15.88" y="18.52" ry="1.37" fill="#3dbdaa"/><rect width="10.57" height="10.66" x="29.11" y="18.52" ry="1.37" fill="#3dbdaa"/><circle cx="33.87" cy="18.56" r=".79" fill="#666"/><circle cx="7.41" cy="18.56" r=".79" fill="#666"/><circle cx="20.64" cy="18.56" r=".79" fill="#666"/><path d="M7.38 18.54c.03-2.63-3.41-2.66-3.41-5.31" fill="none" stroke="#999" stroke-width=".86" stroke-linecap="round"/><path d="M20.64 18.56c0-2.91-15.35-1.36-15.35-5.33" fill="none" stroke="#999" stroke-width=".86" stroke-linecap="round"/><path d="M33.87 18.56c0-3.97-27.26-2.68-27.26-5.33" fill="none" stroke="#999" stroke-width=".86" stroke-linecap="round"/><rect width="68.79" height="10.58" x="2.65" y="2.68" ry="1.37" fill="url(#b)"/><rect width="15.35" height="9.51" x="3.18" y="3.21" ry=".9" fill="#78a2d4"/><rect width="9.26" height="9.51" x="19.85" y="3.2" ry=".9" fill="#eaa577"/><rect width="9.26" height="9.51" x="29.64" y="3.22" ry=".9" fill="#eaa577"/><rect width="9.26" height="9.51" x="39.43" y="3.22" ry=".9" fill="#eaa577"/><rect width="9.26" height="9.51" x="49.22" y="3.21" ry=".9" fill="#eaa577"/><circle cx="62.84" cy="7.97" r=".66" fill="#eaa577"/><circle cx="64.7" cy="7.97" r=".66" fill="#eaa577"/><circle cx="66.55" cy="7.97" r=".66" fill="#eaa577"/></svg>
///
/// See [RefCountedVector].
pub type SharedVector<T, A = GlobalAllocator> = RefCountedVector<T, DefaultRefCount, A>;

/// A heap allocated, reference counted, immutable contiguous buffer containing elements of type `T`.
///
/// <svg width="280" height="120" viewBox="0 0 74.08 31.75" xmlns:xlink="http://www.w3.org/1999/xlink" xmlns="http://www.w3.org/2000/svg"><defs><linearGradient id="a"><stop offset="0" stop-color="#491c9c"/><stop offset="1" stop-color="#d54b27"/></linearGradient><linearGradient xlink:href="#a" id="b" gradientUnits="userSpaceOnUse" x1="6.27" y1="34.86" x2="87.72" y2="13.24" gradientTransform="translate(-2.64 -18.48)"/></defs><rect width="10.57" height="10.66" x="2.66" y="18.48" ry="1.37" fill="#3dbdaa"/><rect width="10.57" height="10.66" x="15.88" y="18.52" ry="1.37" fill="#3dbdaa"/><rect width="10.57" height="10.66" x="29.11" y="18.52" ry="1.37" fill="#3dbdaa"/><circle cx="33.87" cy="18.56" r=".79" fill="#666"/><circle cx="7.41" cy="18.56" r=".79" fill="#666"/><circle cx="20.64" cy="18.56" r=".79" fill="#666"/><path d="M7.38 18.54c.03-2.63-3.41-2.66-3.41-5.31" fill="none" stroke="#999" stroke-width=".86" stroke-linecap="round"/><path d="M20.64 18.56c0-2.91-15.35-1.36-15.35-5.33" fill="none" stroke="#999" stroke-width=".86" stroke-linecap="round"/><path d="M33.87 18.56c0-3.97-27.26-2.68-27.26-5.33" fill="none" stroke="#999" stroke-width=".86" stroke-linecap="round"/><rect width="68.79" height="10.58" x="2.65" y="2.68" ry="1.37" fill="url(#b)"/><rect width="15.35" height="9.51" x="3.18" y="3.21" ry=".9" fill="#78a2d4"/><rect width="9.26" height="9.51" x="19.85" y="3.2" ry=".9" fill="#eaa577"/><rect width="9.26" height="9.51" x="29.64" y="3.22" ry=".9" fill="#eaa577"/><rect width="9.26" height="9.51" x="39.43" y="3.22" ry=".9" fill="#eaa577"/><rect width="9.26" height="9.51" x="49.22" y="3.21" ry=".9" fill="#eaa577"/><circle cx="62.84" cy="7.97" r=".66" fill="#eaa577"/><circle cx="64.7" cy="7.97" r=".66" fill="#eaa577"/><circle cx="66.55" cy="7.97" r=".66" fill="#eaa577"/></svg>
///
/// Similar in principle to `Arc<[T]>`. It can be converted into a `UniqueVector<T>` for
/// free if there is only a single reference to the RefCountedVector alive.
///
/// # Copy-on-write "Immutable" vectors
///
/// This type contains mutable methods like `push` and `pop`. These internally allocate a new buffer
/// if the buffer is not unique (there are more than one reference to it). When there is a single reference,
/// these mutable operation simply update the existing buffer.
///
/// In other words, this type behaves like an [immutable (or persistent) data structure](https://en.wikipedia.org/wiki/Persistent_data_structure)
/// Actual mutability only happens under the hood as an optimization when a single reference exists.
#[repr(transparent)]
pub struct RefCountedVector<T, R: RefCount, A: Allocator = GlobalAllocator> {
    pub(crate) inner: HeaderBuffer<T, R, A>,
}

impl<T, R: RefCount> RefCountedVector<T, R, GlobalAllocator> {
    /// Creates an empty shared buffer without allocating memory.
    #[inline]
    pub fn new() -> RefCountedVector<T, R, GlobalAllocator> {
        RefCountedVector {
            inner: HeaderBuffer::try_with_capacity(0, GlobalAllocator).unwrap(),
        }
    }

    /// Constructs a new, empty vector with at least the specified capacity.
    #[inline]
    pub fn with_capacity(cap: usize) -> RefCountedVector<T, R, GlobalAllocator> {
        RefCountedVector {
            inner: HeaderBuffer::try_with_capacity(cap, GlobalAllocator).unwrap(),
        }
    }

    /// Clones the contents of a slice into a new vector.
    #[inline]
    pub fn from_slice(data: &[T]) -> RefCountedVector<T, R, GlobalAllocator>
    where
        T: Clone,
    {
        RefCountedVector {
            inner: HeaderBuffer::try_from_slice(data, None, GlobalAllocator).unwrap(),
        }
    }
}

impl<T, R: RefCount, A: Allocator> RefCountedVector<T, R, A> {
    /// Tries to construct a new, empty vector with at least the specified capacity.
    #[inline]
    pub fn try_with_allocator(cap: usize, allocator: A) -> Result<Self, AllocError> {
        Ok(RefCountedVector {
            inner: HeaderBuffer::try_with_capacity(cap, allocator)?,
        })
    }

    /// Returns `true` if the vector contains no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns the number of elements in the vector, also referred to as its ‘length’.
    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len() as usize
    }

    /// Returns the total number of elements the vector can hold without reallocating.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.inner.capacity() as usize
    }

    /// Returns number of elements that can be added without reallocating.
    #[inline]
    pub fn remaining_capacity(&self) -> usize {
        self.inner.remaining_capacity() as usize
    }

    /// Creates a new reference without allocating.
    ///
    /// Equivalent to `Clone::clone`.
    #[inline]
    pub fn new_ref(&self) -> Self {
        RefCountedVector {
            inner: self.inner.new_ref(),
        }
    }

    /// Extracts a slice containing the entire vector.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.inner.as_slice()
    }

    /// Extracts a mutable slice containing the entire vector.
    ///
    /// Like other mutable methods, this will clone the vector's storage
    /// if it is not unique to ensure safe mutations.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut[T] where T: Clone {
        self.ensure_unique();
        self.inner.as_mut_slice()
    }

    /// Allocates a duplicate of this buffer (infallible).
    pub fn clone_buffer(&self) -> Self
    where
        T: Clone,
    {
        RefCountedVector {
            inner: self.inner.try_clone_buffer(None).unwrap(),
        }
    }

    /// Allocates a duplicate of this buffer (infallible).
    pub fn copy_buffer(&self) -> Self where T: Copy {
        RefCountedVector {
            inner: self.inner.try_copy_buffer(None).unwrap(),
        }
    }

    /// Tries to allocate a duplicate of this buffer.
    pub fn try_copy_buffer(&self) -> Result<Self, AllocError> where T: Copy {
        Ok(RefCountedVector {
            inner: self.inner.try_copy_buffer(None)?,
        })
    }

    /// Returns true if this is the only existing handle to the buffer.
    ///
    /// When this function returns true, mutable methods and converting to a `UniqueVector`
    /// is very fast (does not involve additional memory allocations or copies).
    #[inline]
    pub fn is_unique(&self) -> bool {
        self.inner.is_unique()
    }

    /// Converts this RefCountedVector into an immutable one, allocating a new copy if there are other references.
    #[inline]
    pub fn into_unique(mut self) -> UniqueVector<T, A>
    where
        T: Clone,
    {
        self.ensure_unique();

        unsafe {
            let data = NonNull::new_unchecked(self.inner.data_ptr());
            let len = self.len() as BufferSize;
            let cap = self.capacity() as BufferSize;
            let allocator = self.inner.header.as_ref().allocator.clone();

            mem::forget(self);

            UniqueVector { data, len, cap, allocator }
        }
    }

    pub fn push(&mut self, val: T)
    where
        T: Clone,
    {
        self.reserve(1);
        unsafe {
            self.inner.push(val);
        }
    }

    /// Removes the last element from the vector and returns it, or `None` if it is empty.
    pub fn pop(&mut self) -> Option<T>
    where
        T: Clone,
    {
        self.ensure_unique();
        unsafe { self.inner.pop() }
    }

    /// Removes an element from the vector and returns it.
    ///
    /// The removed element is replaced by the last element of the vector.
    ///
    /// # Panics
    ///
    /// Panics if index is out of bounds.
    #[inline]
    pub fn swap_remove(&mut self, idx: usize) -> T where T: Clone {
        self.ensure_unique();

        let len = self.len();
        assert!(idx < len);

        unsafe {
            let ptr = self.inner.data_ptr().add(idx);
            let item = ptr::read(ptr);

            let last_idx = len - 1;
            if idx != last_idx {
                let last_ptr = self.inner.data_ptr().add(last_idx);
                ptr::write(ptr, ptr::read(last_ptr));    
            }

            self.inner.set_len(last_idx as BufferSize);

            item
        }
    }

    /// Appends an element if there is sufficient spare capacity, otherwise an error is returned
    /// with the element.
    ///
    /// Like other mutable operations, this method may reallocate if the vector is not unique.
    /// Hopwever it will not reallocate when there’s insufficient capacity.
    /// The caller should use reserve or try_reserve to ensure that there is enough capacity.
    pub fn push_within_capacity(&mut self, val: T) -> Result<(), T> where T: Clone {
        if self.remaining_capacity() == 0 {
            return Err(val);
        }

        self.ensure_unique();
        unsafe {
            self.inner.push(val);
        }

        Ok(())
    }

    pub fn push_slice(&mut self, data: &[T])
    where
        T: Clone,
    {
        self.reserve(data.len());
        unsafe {
            self.inner.try_push_slice(data).unwrap();
        }
    }

    pub fn extend(&mut self, data: impl IntoIterator<Item = T>)
    where
        T: Clone,
    {
        let mut iter = data.into_iter();
        let (min, max) = iter.size_hint();
        self.reserve(max.unwrap_or(min));
        unsafe {
            self.inner.try_extend(&mut iter).unwrap();
        }
    }

    /// Clears the vector, removing all values.
    pub fn clear(&mut self) {
        if self.is_unique() {
            unsafe { self.inner.clear(); }
            return;
        }

        *self = Self::try_with_allocator(self.capacity(), self.inner.clone_allocator()).unwrap();
    }

    /// Returns true if the two vectors share the same underlying storage.
    pub fn ptr_eq(&self, other: &Self) -> bool {
        self.inner.ptr_eq(&other.inner)
    }

    /// Ensures this shared vector uniquely owns its storage, allocating a new copy
    /// If there are other references to it.
    ///
    /// In principle this is mostly useful internally to provide safe mutable methods
    /// as it does not observaly affect most of the shared vector behavior, however
    /// it has a few niche use cases, for example to provoke copies earlier for more
    /// predictable performance or in some unsafe endeavors.
    #[inline]
    pub fn ensure_unique(&mut self)
    where
        T: Clone,
    {
        if !self.is_unique() {
            self.inner = self.inner.try_clone_buffer(None).unwrap();
        }
    }

    /// Ensures the vector can be safely mutated and has enough extra capacity to
    /// add `additional` more items.
    ///
    /// This will allocate new storage for the vector if the vector is not unique or if
    /// the capacity is not sufficient to accomodate `self.len() + additional` items.
    /// The vector may reserve more space to speculatively avoid frequent reallocations.
    #[inline]
    pub fn reserve(&mut self, additional: usize)
    where
        T: Clone,
    {
        let is_unique = self.is_unique();
        let enough_capacity = self.remaining_capacity() >= additional;

        if !is_unique || !enough_capacity {
            // Hopefully the least common case.
            self.try_realloc_additional(is_unique, enough_capacity, additional).unwrap();
        }
    }

    /// Tries to reserve at least `additional` extra elements to be inserted in the given vector.
    ///
    /// The vector may reserve more space to speculatively avoid frequent reallocations.
    /// After calling try_reserve, capacity will be greater than or equal to `self.len() + additional`
    /// if it returns `Ok(())`.
    /// Does nothing if capacity is already sufficient. This method preserves the contents even if an
    /// error occurs.
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), AllocError>
    where
        T: Clone,
    {
        let is_unique = self.is_unique();
        let enough_capacity = self.remaining_capacity() >= additional;

        if !is_unique || !enough_capacity {
            // Hopefully the least common case.
            self.try_realloc_additional(is_unique, enough_capacity, additional)?;
        }

        Ok(())
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
    pub fn try_reserve_exact(&mut self, additional: usize) -> Result<(), AllocError>
    where
        T: Clone,
    {
        let is_unique = self.is_unique();
        let enough_capacity = self.remaining_capacity() >= additional;

        if !is_unique || !enough_capacity {
            // Hopefully the least common case.
            self.try_realloc_with_capacity(is_unique, additional)?;
        }

        Ok(())
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

        let is_unique = self.is_unique();
        self.try_realloc_with_capacity(is_unique, min_capacity).unwrap();
    }

    /// Shrinks the capacity of the vector as much as possible.
    pub fn shrink_to_fit(&mut self) where T: Clone {
        self.shrink_to(self.len())
    }

    #[cold]
    fn try_realloc_additional(&mut self, is_unique: bool, enough_capacity: bool, additional: usize) -> Result<(), AllocError>
    where
        T: Clone,
    {
        let new_cap = if enough_capacity {
            self.capacity()
        } else {
            grow_amortized(self.len(), additional)
        };

        self.try_realloc_with_capacity(is_unique, new_cap)
    }

    #[cold]
    fn try_realloc_with_capacity(&mut self, is_unique: bool, new_cap: usize) -> Result<(), AllocError>
    where
        T: Clone,
    {
        let allocator = self.inner.clone_allocator();
        if is_unique {
            // The buffer is not large enough, we'll have to create a new one, however we
            // know that we have the only reference to it so we'll move the data with
            // a simple memcpy instead of cloning it.
            unsafe {
                let mut dst = Self::try_with_allocator(new_cap, allocator)?;
                let len = self.len();
                if len > 0 {
                    ptr::copy_nonoverlapping(
                        self.inner.data_ptr(),
                        dst.inner.data_ptr(),
                        len,
                    );
                    dst.inner.set_len(len as BufferSize);
                    self.inner.set_len(0);
                }

                self.inner = dst.inner;
                return Ok(());
            }
        }

        // The slowest path, we pay for both the new allocation and the need to clone
        // each item one by one.
        self.inner = HeaderBuffer::try_from_slice(self.as_slice(), Some(new_cap), allocator)?;

        Ok(())
    }

    // TODO: remove this one?
    /// Returns the concatenation of two vectors.
    pub fn concatenate(mut self, mut other: Self) -> Self
    where
        T: Clone,
    {
        self.append(&mut other);

        self
    }

    /// Moves all the elements of `other` into `self`, leaving `other` empty.
    ///
    /// If `other is not unique, the elements are cloned instead of moved.
    pub fn append(&mut self, other: &mut Self) where T: Clone {
        self.reserve(other.len());

        unsafe {
            if other.is_unique() {
                // Fast path: memcpy
                other.inner.move_data(&mut self.inner);
            } else {
                // Slow path, clone each item.
                self.inner.try_push_slice(other.as_slice()).unwrap();
                *other = Self::try_with_allocator(other.capacity(), self.inner.clone_allocator()).unwrap();
            }
        }
    }

    #[allow(unused)]
    pub(crate) fn addr(&self) -> *const u8 {
        self.inner.header.as_ptr() as *const u8
    }
}

unsafe impl<T: Sync, A: Allocator + Send> Send for AtomicSharedVector<T, A> {}

impl<T, R: RefCount, A: Allocator> Clone for RefCountedVector<T, R, A> {
    fn clone(&self) -> Self {
        self.new_ref()
    }
}

impl<T: PartialEq<T>, R: RefCount, A: Allocator> PartialEq<RefCountedVector<T, R, A>> for RefCountedVector<T, R, A> {
    fn eq(&self, other: &Self) -> bool {
        self.ptr_eq(other) || self.as_slice() == other.as_slice()
    }
}

impl<T: PartialEq<T>, R: RefCount, A: Allocator> PartialEq<&[T]> for RefCountedVector<T, R, A> {
    fn eq(&self, other: &&[T]) -> bool {
        self.as_slice() == *other
    }
}

impl<T, R: RefCount, A: Allocator> AsRef<[T]> for RefCountedVector<T, R, A> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T, R: RefCount> Default for RefCountedVector<T, R, GlobalAllocator> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, T, R: RefCount, A: Allocator> IntoIterator for &'a RefCountedVector<T, R, A> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;
    fn into_iter(self) -> std::slice::Iter<'a, T> {
        self.as_slice().iter()
    }
}

impl<'a, T: Clone, R: RefCount, A: Allocator> IntoIterator for &'a mut RefCountedVector<T, R, A> {
    type Item = &'a mut T;
    type IntoIter = std::slice::IterMut<'a, T>;
    fn into_iter(self) -> std::slice::IterMut<'a, T> {
        self.as_mut_slice().iter_mut()
    }
}

impl<T, R: RefCount, A: Allocator, I> Index<I> for RefCountedVector<T, R, A>
where
    I: std::slice::SliceIndex<[T]>,
{
    type Output = <I as std::slice::SliceIndex<[T]>>::Output;
    fn index(&self, index: I) -> &Self::Output {
        self.as_slice().index(index)
    }
}

impl<T, R: RefCount, A: Allocator, I> IndexMut<I> for RefCountedVector<T, R, A>
where
    T: Clone,
    I: std::slice::SliceIndex<[T]>,
{
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        self.as_mut_slice().index_mut(index)
    }
}

impl<T, R: RefCount, A: Allocator> Deref for RefCountedVector<T, R, A> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T: Clone, R: RefCount, A: Allocator> DerefMut for RefCountedVector<T, R, A> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T: Debug, R: RefCount, A: Allocator> Debug for RefCountedVector<T, R, A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        self.as_slice().fmt(f)
    }
}


// In order to give us a chance to catch leaks and double-frees, test with values that implement drop.
#[cfg(test)]
fn num(val: u32) -> Box<u32> {
    Box::new(val)
}

#[test]
fn basic_shared() {
    basic_shared_impl::<DefaultRefCount>();
    basic_shared_impl::<AtomicRefCount>();

    fn basic_shared_impl<R: RefCount>() {
        let mut a: RefCountedVector<Box<u32>, R> = RefCountedVector::with_capacity(64);
        a.push(num(1));
        a.push(num(2));

        let mut b = a.new_ref();
        b.push(num(4));

        a.push(num(3));

        assert_eq!(a.as_slice(), &[num(1), num(2), num(3)]);
        assert_eq!(b.as_slice(), &[num(1), num(2), num(4)]);

        let popped = a.pop();
        assert_eq!(a.as_slice(), &[num(1), num(2)]);
        assert_eq!(popped, Some(num(3)));

        let mut b2 = b.new_ref();
        let popped = b2.pop();
        assert_eq!(b2.as_slice(), &[num(1), num(2)]);
        assert_eq!(popped, Some(num(4)));

        let c = a.concatenate(b2);
        assert_eq!(c.as_slice(), &[num(1), num(2), num(1), num(2)]);
    }
}

#[test]
fn empty_buffer() {
    let _: AtomicSharedVector<u32> = AtomicSharedVector::new();
    let _: AtomicSharedVector<u32> = AtomicSharedVector::new();

    let _: SharedVector<()> = SharedVector::new();
    let _: SharedVector<()> = SharedVector::new();

    let _: AtomicSharedVector<()> = AtomicSharedVector::new();
    let _: AtomicSharedVector<()> = AtomicSharedVector::new();

    let _: UniqueVector<()> = UniqueVector::new();
}

#[test]
#[rustfmt::skip]
fn grow() {
    let mut a = UniqueVector::with_capacity(0);

    a.push(num(1));
    a.push(num(2));
    a.push(num(3));

    a.push_slice(&[num(4), num(5), num(6), num(7), num(8), num(9), num(10), num(12), num(12), num(13), num(14), num(15), num(16), num(17), num(18)]);

    assert_eq!(
        a.as_slice(),
        &[num(1), num(2), num(3), num(4), num(5), num(6), num(7), num(8), num(9), num(10), num(12), num(12), num(13), num(14), num(15), num(16), num(17), num(18)]
    );

    let mut b = SharedVector::new();
    b.push(num(1));
    b.push(num(2));
    b.push(num(3));

    assert_eq!(b.as_slice(), &[num(1), num(2), num(3)]);

    let mut b = AtomicSharedVector::new();
    b.push(num(1));
    b.push(num(2));
    b.push(num(3));

    assert_eq!(b.as_slice(), &[num(1), num(2), num(3)]);
}

#[test]
fn ensure_unique_empty() {
    let mut v: SharedVector<u32> = SharedVector::new();
    v.ensure_unique();
}
