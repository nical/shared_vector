use core::fmt::Debug;
use core::ops::{Deref, DerefMut, Index, IndexMut};
use core::ptr::NonNull;
use core::{mem, ptr};
use core::sync::atomic::Ordering;

use crate::raw;
use crate::alloc::{AllocError, Allocator, Global};
use crate::raw::{BufferSize, HeaderBuffer};
use crate::vector::{Vector, RawVector};
use crate::{grow_amortized, AtomicRefCount, DefaultRefCount, RefCount};

/// A heap allocated, atomically reference counted, immutable contiguous buffer containing elements of type `T`.
///
/// <svg width="280" height="120" viewBox="0 0 74.08 31.75" xmlns:xlink="http://www.w3.org/1999/xlink" xmlns="http://www.w3.org/2000/svg"><defs><linearGradient id="a"><stop offset="0" stop-color="#491c9c"/><stop offset="1" stop-color="#d54b27"/></linearGradient><linearGradient xlink:href="#a" id="b" gradientUnits="userSpaceOnUse" x1="6.27" y1="34.86" x2="87.72" y2="13.24" gradientTransform="translate(-2.64 -18.48)"/></defs><rect width="10.57" height="10.66" x="2.66" y="18.48" ry="1.37" fill="#3dbdaa"/><rect width="10.57" height="10.66" x="15.88" y="18.52" ry="1.37" fill="#3dbdaa"/><rect width="10.57" height="10.66" x="29.11" y="18.52" ry="1.37" fill="#3dbdaa"/><circle cx="33.87" cy="18.56" r=".79" fill="#666"/><circle cx="7.41" cy="18.56" r=".79" fill="#666"/><circle cx="20.64" cy="18.56" r=".79" fill="#666"/><path d="M7.38 18.54c.03-2.63-3.41-2.66-3.41-5.31" fill="none" stroke="#999" stroke-width=".86" stroke-linecap="round"/><path d="M20.64 18.56c0-2.91-15.35-1.36-15.35-5.33" fill="none" stroke="#999" stroke-width=".86" stroke-linecap="round"/><path d="M33.87 18.56c0-3.97-27.26-2.68-27.26-5.33" fill="none" stroke="#999" stroke-width=".86" stroke-linecap="round"/><rect width="68.79" height="10.58" x="2.65" y="2.68" ry="1.37" fill="url(#b)"/><rect width="15.35" height="9.51" x="3.18" y="3.21" ry=".9" fill="#78a2d4"/><rect width="9.26" height="9.51" x="19.85" y="3.2" ry=".9" fill="#eaa577"/><rect width="9.26" height="9.51" x="29.64" y="3.22" ry=".9" fill="#eaa577"/><rect width="9.26" height="9.51" x="39.43" y="3.22" ry=".9" fill="#eaa577"/><rect width="9.26" height="9.51" x="49.22" y="3.21" ry=".9" fill="#eaa577"/><circle cx="62.84" cy="7.97" r=".66" fill="#eaa577"/><circle cx="64.7" cy="7.97" r=".66" fill="#eaa577"/><circle cx="66.55" cy="7.97" r=".66" fill="#eaa577"/></svg>
///
/// See [RefCountedVector].
pub type AtomicSharedVector<T, A = Global> = RefCountedVector<T, AtomicRefCount, A>;

/// A heap allocated, reference counted, immutable contiguous buffer containing elements of type `T`.
///
/// <svg width="280" height="120" viewBox="0 0 74.08 31.75" xmlns:xlink="http://www.w3.org/1999/xlink" xmlns="http://www.w3.org/2000/svg"><defs><linearGradient id="a"><stop offset="0" stop-color="#491c9c"/><stop offset="1" stop-color="#d54b27"/></linearGradient><linearGradient xlink:href="#a" id="b" gradientUnits="userSpaceOnUse" x1="6.27" y1="34.86" x2="87.72" y2="13.24" gradientTransform="translate(-2.64 -18.48)"/></defs><rect width="10.57" height="10.66" x="2.66" y="18.48" ry="1.37" fill="#3dbdaa"/><rect width="10.57" height="10.66" x="15.88" y="18.52" ry="1.37" fill="#3dbdaa"/><rect width="10.57" height="10.66" x="29.11" y="18.52" ry="1.37" fill="#3dbdaa"/><circle cx="33.87" cy="18.56" r=".79" fill="#666"/><circle cx="7.41" cy="18.56" r=".79" fill="#666"/><circle cx="20.64" cy="18.56" r=".79" fill="#666"/><path d="M7.38 18.54c.03-2.63-3.41-2.66-3.41-5.31" fill="none" stroke="#999" stroke-width=".86" stroke-linecap="round"/><path d="M20.64 18.56c0-2.91-15.35-1.36-15.35-5.33" fill="none" stroke="#999" stroke-width=".86" stroke-linecap="round"/><path d="M33.87 18.56c0-3.97-27.26-2.68-27.26-5.33" fill="none" stroke="#999" stroke-width=".86" stroke-linecap="round"/><rect width="68.79" height="10.58" x="2.65" y="2.68" ry="1.37" fill="url(#b)"/><rect width="15.35" height="9.51" x="3.18" y="3.21" ry=".9" fill="#78a2d4"/><rect width="9.26" height="9.51" x="19.85" y="3.2" ry=".9" fill="#eaa577"/><rect width="9.26" height="9.51" x="29.64" y="3.22" ry=".9" fill="#eaa577"/><rect width="9.26" height="9.51" x="39.43" y="3.22" ry=".9" fill="#eaa577"/><rect width="9.26" height="9.51" x="49.22" y="3.21" ry=".9" fill="#eaa577"/><circle cx="62.84" cy="7.97" r=".66" fill="#eaa577"/><circle cx="64.7" cy="7.97" r=".66" fill="#eaa577"/><circle cx="66.55" cy="7.97" r=".66" fill="#eaa577"/></svg>
///
/// See [RefCountedVector].
pub type SharedVector<T, A = Global> = RefCountedVector<T, DefaultRefCount, A>;

/// A heap allocated, reference counted, immutable contiguous buffer containing elements of type `T`.
///
/// <svg width="280" height="120" viewBox="0 0 74.08 31.75" xmlns:xlink="http://www.w3.org/1999/xlink" xmlns="http://www.w3.org/2000/svg"><defs><linearGradient id="a"><stop offset="0" stop-color="#491c9c"/><stop offset="1" stop-color="#d54b27"/></linearGradient><linearGradient xlink:href="#a" id="b" gradientUnits="userSpaceOnUse" x1="6.27" y1="34.86" x2="87.72" y2="13.24" gradientTransform="translate(-2.64 -18.48)"/></defs><rect width="10.57" height="10.66" x="2.66" y="18.48" ry="1.37" fill="#3dbdaa"/><rect width="10.57" height="10.66" x="15.88" y="18.52" ry="1.37" fill="#3dbdaa"/><rect width="10.57" height="10.66" x="29.11" y="18.52" ry="1.37" fill="#3dbdaa"/><circle cx="33.87" cy="18.56" r=".79" fill="#666"/><circle cx="7.41" cy="18.56" r=".79" fill="#666"/><circle cx="20.64" cy="18.56" r=".79" fill="#666"/><path d="M7.38 18.54c.03-2.63-3.41-2.66-3.41-5.31" fill="none" stroke="#999" stroke-width=".86" stroke-linecap="round"/><path d="M20.64 18.56c0-2.91-15.35-1.36-15.35-5.33" fill="none" stroke="#999" stroke-width=".86" stroke-linecap="round"/><path d="M33.87 18.56c0-3.97-27.26-2.68-27.26-5.33" fill="none" stroke="#999" stroke-width=".86" stroke-linecap="round"/><rect width="68.79" height="10.58" x="2.65" y="2.68" ry="1.37" fill="url(#b)"/><rect width="15.35" height="9.51" x="3.18" y="3.21" ry=".9" fill="#78a2d4"/><rect width="9.26" height="9.51" x="19.85" y="3.2" ry=".9" fill="#eaa577"/><rect width="9.26" height="9.51" x="29.64" y="3.22" ry=".9" fill="#eaa577"/><rect width="9.26" height="9.51" x="39.43" y="3.22" ry=".9" fill="#eaa577"/><rect width="9.26" height="9.51" x="49.22" y="3.21" ry=".9" fill="#eaa577"/><circle cx="62.84" cy="7.97" r=".66" fill="#eaa577"/><circle cx="64.7" cy="7.97" r=".66" fill="#eaa577"/><circle cx="66.55" cy="7.97" r=".66" fill="#eaa577"/></svg>
///
/// Similar in principle to `Arc<[T]>`. It can be converted into a `Vector<T>` for
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
pub struct RefCountedVector<T, R: RefCount, A: Allocator = Global> {
    pub(crate) inner: HeaderBuffer<T, R, A>,
}

impl<T, R: RefCount> RefCountedVector<T, R, Global> {
    /// Creates an empty shared buffer without allocating memory.
    #[inline]
    pub fn new() -> RefCountedVector<T, R, Global> {
        Self::try_with_capacity_in(0, Global).unwrap()
    }

    /// Constructs a new, empty vector with at least the specified capacity.
    #[inline]
    pub fn with_capacity(cap: usize) -> RefCountedVector<T, R, Global> {
        Self::try_with_capacity_in(cap, Global).unwrap()
    }

    /// Clones the contents of a slice into a new vector.
    #[inline]
    pub fn from_slice(slice: &[T]) -> RefCountedVector<T, R, Global>
    where
        T: Clone,
    {
        Self::try_from_slice_in(slice, Global).unwrap()
    }
}

impl<T, R: RefCount, A: Allocator> RefCountedVector<T, R, A> {
    /// Creates an empty vector without allocating memory.
    pub fn new_in(allocator: A) -> Self {
        Self::try_with_capacity_in(0, allocator).unwrap()
    }

    /// Creates an empty pre-allocated vector with a given storage capacity.
    pub fn with_capacity_in(cap: usize, allocator: A) -> Self {
        Self::try_with_capacity_in(cap, allocator).unwrap()
    }

    /// Tries to construct a new, empty vector with at least the specified capacity.
    #[inline]
    pub fn try_with_capacity_in(cap: usize, allocator: A) -> Result<Self, AllocError> {
        raw::assert_ref_count_layout::<R>();
        unsafe {
            let (ptr, cap) = raw::allocate_header_buffer::<T, A>(cap, &allocator)?;

            ptr::write(
                ptr.cast().as_ptr(),
                raw::Header {
                    vec: raw::VecHeader {
                        cap: cap as BufferSize,
                        len: 0,
                    },
                    ref_count: R::new(1),
                    allocator,
                },
            );

            Ok(RefCountedVector {
                inner: HeaderBuffer::from_raw(ptr.cast()),
            })
            }
    }

    pub fn try_from_slice_in(slice: &[T], allocator: A) -> Result<Self, AllocError> where T: Clone {
        let mut v = Self::try_with_capacity_in(slice.len(), allocator)?;

        unsafe {
            raw::extend_from_slice_assuming_capacity(v.data_ptr(), v.vec_header_mut(), slice);
        }

        Ok(v)
    }

    /// Returns `true` if the vector contains no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.vec_header().len == 0
    }

    /// Returns the number of elements in the vector, also referred to as its ‘length’.
    #[inline]
    pub fn len(&self) -> usize {
        self.vec_header().len as usize
    }

    /// Returns the total number of elements the vector can hold without reallocating.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.vec_header().cap as usize
    }

    /// Returns number of elements that can be added without reallocating.
    #[inline]
    pub fn remaining_capacity(&self) -> usize {
        let h = self.vec_header();
        (h.cap - h.len) as usize
    }

    /// Returns a reference to the underlying allocator.
    pub fn allocator(&self) -> &A {
        self.inner.allocator()
    }

    /// Creates a new reference without allocating.
    ///
    /// Equivalent to `Clone::clone`.
    #[inline]
    pub fn new_ref(&self) -> Self {
        unsafe {
            self.inner.as_ref().ref_count.add_ref();
            RefCountedVector {
                inner: HeaderBuffer::from_raw(self.inner.header)
            }
        }
    }

    /// Extracts a slice containing the entire vector.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        unsafe {
            core::slice::from_raw_parts(self.data_ptr(), self.len())
        }
    }

    /// Returns true if this is the only existing handle to the buffer.
    ///
    /// When this function returns true, mutable methods and converting to a `Vector`
    /// is very fast (does not involve additional memory allocations or copies).
    #[inline]
    pub fn is_unique(&self) -> bool {
        unsafe { self.inner.as_ref().ref_count.get() == 1 }
    }

    /// Clears the vector, removing all values.
    pub fn clear(&mut self)
    where
        A: Clone,
    {
        if self.is_unique() {
            unsafe {
                raw::clear(self.data_ptr(), self.vec_header_mut());
            }
            return;
        }

        *self =
            Self::try_with_capacity_in(self.capacity(), self.inner.allocator().clone()).unwrap();
    }

    /// Returns true if the two vectors share the same underlying storage.
    pub fn ptr_eq(&self, other: &Self) -> bool {
        self.inner.header == other.inner.header
    }

    /// Allocates a duplicate of this buffer (infallible).
    pub fn copy_buffer(&self) -> Self
    where
        T: Copy,
        A: Clone,
    {
        self.try_copy_buffer().unwrap()
    }

    /// Tries to allocate a duplicate of this buffer.
    pub fn try_copy_buffer(&self) -> Result<Self, AllocError>
    where
        T: Copy,
        A: Clone,
    {
        unsafe {
            let header = self.inner.as_ref();
            let len = header.vec.len;
            let cap = header.vec.cap;

            if len > cap {
                return Err(AllocError);
            }

            let allocator = header.allocator.clone();
            let mut clone = Self::try_with_capacity_in(cap as usize, allocator)?;

            if len > 0 {
                core::ptr::copy_nonoverlapping(self.data_ptr(), clone.data_ptr(), len as usize);
                clone.vec_header_mut().len = len;
            }

            Ok(clone)
        }
    }

    #[inline]
    pub fn data_ptr(&self) -> *mut T {
        unsafe { (self.inner.as_ptr() as *mut u8).add(raw::header_size::<raw::Header<R, A>, T>()) as *mut T }
    }

    // SAFETY: call this only if the vector is unique.
    pub(crate) unsafe fn vec_header_mut(&mut self) -> &mut raw::VecHeader {
        &mut self.inner.as_mut().vec
    } 

    pub(crate) fn vec_header(&self) -> &raw::VecHeader {
        unsafe { &self.inner.as_ref().vec }
    }
}

/// Mutable methods that can cause the vector to be cloned and therefore require both the items and
/// the allocator to be cloneable.
impl<T: Clone, R: RefCount, A: Allocator + Clone> RefCountedVector<T, R, A> {
    /// Converts this RefCountedVector into an immutable one, allocating a new copy if there are other references.
    #[inline]
    pub fn into_unique(mut self) -> Vector<T, A> {
        self.ensure_unique();

        unsafe {
            let data = NonNull::new_unchecked(self.data_ptr());
            let header = self.vec_header().clone();
            let allocator = self.inner.as_ref().allocator.clone();

            mem::forget(self);

            Vector {
                raw: RawVector {
                    data,
                    header,
                },
                allocator,
            }
        }
    }

    /// Appends an element to the back of a collection.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity exceeds `u32::MAX` bytes.
    pub fn push(&mut self, val: T) {
        self.reserve(1);
        unsafe {
            raw::push_assuming_capacity(self.data_ptr(), &mut self.vec_header_mut(), val);
        }
    }

    /// Removes the last element from the vector and returns it, or `None` if it is empty.
    pub fn pop(&mut self) -> Option<T> {
        self.ensure_unique();

        unsafe {
            raw::pop(self.data_ptr(), &mut self.vec_header_mut())
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
        self.ensure_unique();

        let len = self.len();
        assert!(idx < len);

        unsafe {
            let data_ptr = self.data_ptr();
            let ptr = data_ptr.add(idx);
            let item = ptr::read(ptr);

            let last_idx = len - 1;
            if idx != last_idx {
                let last_ptr = data_ptr.add(last_idx);
                ptr::write(ptr, ptr::read(last_ptr));
            }

            self.vec_header_mut().len = last_idx as BufferSize;

            item
        }
    }

    /// Appends an element if there is sufficient spare capacity, otherwise an error is returned
    /// with the element.
    ///
    /// Like other mutable operations, this method may reallocate if the vector is not unique.
    /// However it will not reallocate when there’s insufficient capacity.
    /// The caller should use reserve or try_reserve to ensure that there is enough capacity.
    pub fn push_within_capacity(&mut self, val: T) -> Result<(), T> {
        if self.remaining_capacity() == 0 {
            return Err(val);
        }

        self.ensure_unique();
        unsafe {
            raw::push_assuming_capacity(self.data_ptr(), &mut self.vec_header_mut(), val);
        }

        Ok(())
    }

    /// Clones and appends the contents of the slice to the back of a collection.
    pub fn extend_from_slice(&mut self, slice: &[T]) {
        self.reserve(slice.len());
        unsafe {
            raw::extend_from_slice_assuming_capacity(self.data_ptr(), self.vec_header_mut(), slice);
        }
    }

    /// Appends the contents of an iterator to the back of a collection.
    pub fn extend(&mut self, data: impl IntoIterator<Item = T>) {
        let mut iter = data.into_iter();

        let (min, max) = iter.size_hint();
        self.reserve(max.unwrap_or(min));

        unsafe {
            if raw::extend_within_capacity(self.data_ptr(), self.vec_header_mut(), &mut iter) {
                return;
            }
        }

        for item in iter {
            self.push(item);
        }
    }

    /// Ensures this shared vector uniquely owns its storage, allocating a new copy
    /// If there are other references to it.
    ///
    /// In principle this is mostly useful internally to provide safe mutable methods
    /// as it does not observaly affect most of the shared vector behavior, however
    /// it has a few niche use cases, for example to provoke copies earlier for more
    /// predictable performance or in some unsafe endeavors.
    #[inline]
    pub fn ensure_unique(&mut self) {
        if !self.is_unique() {
            *self = self.try_clone_buffer(None).unwrap();
        }
    }

    /// Extracts a mutable slice containing the entire vector.
    ///
    /// Like other mutable methods, this will clone the vector's storage
    /// if it is not unique to ensure safe mutations.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T]
    where
        T: Clone,
        A: Clone,
    {
        self.ensure_unique();
        unsafe {
            core::slice::from_raw_parts_mut(self.data_ptr(), self.len())
        }
    }

    /// Allocates a duplicate of this buffer (infallible).
    pub fn clone_buffer(&self) -> Self
    where
        T: Clone,
        A: Clone,
    {
        self.try_clone_buffer(None).unwrap()
    }

    fn try_clone_buffer(&self, new_cap: Option<BufferSize>) -> Result<Self, AllocError>
    where
        T: Clone,
        A: Clone,
    {
        unsafe {
            let header = self.inner.as_ref();
            let len = header.vec.len;
            let cap = if let Some(cap) = new_cap {
                cap
            } else {
                header.vec.cap
            };
            let allocator = header.allocator.clone();

            if len > cap {
                return Err(AllocError);
            }

            let mut clone = Self::try_with_capacity_in(cap as usize, allocator)?;

            raw::extend_from_slice_assuming_capacity(
                clone.data_ptr(),
                clone.vec_header_mut(),
                self.as_slice()
            );

            Ok(clone)
        }
    }

    /// Ensures the vector can be safely mutated and has enough extra capacity to
    /// add `additional` more items.
    ///
    /// This will allocate new storage for the vector if the vector is not unique or if
    /// the capacity is not sufficient to accomodate `self.len() + additional` items.
    /// The vector may reserve more space to speculatively avoid frequent reallocations.
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        let is_unique = self.is_unique();
        let enough_capacity = self.remaining_capacity() >= additional;

        if !is_unique || !enough_capacity {
            // Hopefully the least common case.
            self.try_realloc_additional(is_unique, enough_capacity, additional)
                .unwrap();
        }
    }

    /// Tries to reserve at least `additional` extra elements to be inserted in the given vector.
    ///
    /// The vector may reserve more space to speculatively avoid frequent reallocations.
    /// After calling try_reserve, capacity will be greater than or equal to `self.len() + additional`
    /// if it returns `Ok(())`.
    /// Does nothing if capacity is already sufficient. This method preserves the contents even if an
    /// error occurs.
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), AllocError> {
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
    pub fn reserve_exact(&mut self, additional: usize) {
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
    pub fn shrink_to(&mut self, min_capacity: usize) {
        let min_capacity = min_capacity.max(self.len());
        if self.capacity() <= min_capacity {
            return;
        }

        let is_unique = self.is_unique();
        self.try_realloc_with_capacity(is_unique, min_capacity)
            .unwrap();
    }

    /// Shrinks the capacity of the vector as much as possible.
    pub fn shrink_to_fit(&mut self) {
        self.shrink_to(self.len())
    }

    /// Moves all the elements of `other` into `self`, leaving `other` empty.
    ///
    /// If `other is not unique, the elements are cloned instead of moved.
    pub fn append(&mut self, other: &mut Self) {
        self.reserve(other.len());

        unsafe {
            if other.is_unique() {
                // Fast path: memcpy
                raw::move_data(
                     other.data_ptr(), &mut other.inner.header.as_mut().vec,
                     self.data_ptr(), &mut self.inner.as_mut().vec,
                )
            } else {
                // Slow path, clone each item.
                raw::extend_from_slice_assuming_capacity(self.data_ptr(), self.vec_header_mut(), other.as_slice());

                *other =
                    Self::try_with_capacity_in(other.capacity(), self.inner.allocator().clone())
                        .unwrap();
            }
        }
    }

    #[cold]
    fn try_realloc_additional(
        &mut self,
        is_unique: bool,
        enough_capacity: bool,
        additional: usize,
    ) -> Result<(), AllocError> {
        let new_cap = if enough_capacity {
            self.capacity()
        } else {
            grow_amortized(self.len(), additional)
        };

        self.try_realloc_with_capacity(is_unique, new_cap)
    }

    #[cold]
    fn try_realloc_with_capacity(
        &mut self,
        is_unique: bool,
        new_cap: usize,
    ) -> Result<(), AllocError> {
        let allocator = self.inner.allocator().clone();
        if is_unique && self.capacity() > 0 {
            // The buffer is not large enough, we'll have to create a new one, however we
            // know that we have the only reference to it so we'll move the data with
            // a simple memcpy instead of cloning it.

            unsafe {
                use crate::raw::{buffer_layout, Header};
                let old_cap = self.capacity();
                let old_header = self.inner.header;
                let old_layout = buffer_layout::<Header<R, A>, T>(old_cap).unwrap();
                let new_layout = buffer_layout::<Header<R, A>, T>(new_cap).unwrap();

                let new_alloc = if new_layout.size() >= old_layout.size() {
                    allocator.grow(old_header.cast(), old_layout, new_layout)
                } else {
                    allocator.shrink(old_header.cast(), old_layout, new_layout)
                }?;

                self.inner.header = new_alloc.cast();
                self.inner.as_mut().vec.cap = new_cap as BufferSize;

                return Ok(());
            }
        }

        // The slowest path, we pay for both the new allocation and the need to clone
        // each item one by one.
        let mut new_vec = Self::try_with_capacity_in(new_cap, allocator)?;
        new_vec.extend_from_slice(self.as_slice());

        mem::swap(self, &mut new_vec);

        Ok(())
    }


    // TODO: remove this one?
    /// Returns the concatenation of two vectors.
    pub fn concatenate(mut self, mut other: Self) -> Self
    where
        T: Clone,
        A: Clone,
    {
        self.append(&mut other);

        self
    }
}

impl<T, R: RefCount, A: Allocator> Drop for RefCountedVector<T, R, A> {
    fn drop(&mut self) {
        unsafe {
            if self.inner.as_ref().ref_count.release_ref() {
                let header = self.vec_header().clone();
                // See the implementation of std Arc for the need to use this fence. Note that
                // we only need it for the atomic reference counted version but I don't expect
                // this to make a measurable difference.
                core::sync::atomic::fence(Ordering::Acquire);
                
                raw::drop_items(self.data_ptr(), header.len);
                raw::dealloc::<T, R, A>(self.inner.header, header.cap);
            }
        }
    }
}


unsafe impl<T: Sync, A: Allocator + Send> Send for AtomicSharedVector<T, A> {}

impl<T, R: RefCount, A: Allocator> Clone for RefCountedVector<T, R, A> {
    fn clone(&self) -> Self {
        self.new_ref()
    }
}

impl<T: PartialEq<T>, R: RefCount, A: Allocator> PartialEq<RefCountedVector<T, R, A>>
    for RefCountedVector<T, R, A>
{
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

impl<T, R: RefCount> Default for RefCountedVector<T, R, Global> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, T, R: RefCount, A: Allocator> IntoIterator for &'a RefCountedVector<T, R, A> {
    type Item = &'a T;
    type IntoIter = core::slice::Iter<'a, T>;
    fn into_iter(self) -> core::slice::Iter<'a, T> {
        self.as_slice().iter()
    }
}

impl<'a, T: Clone, R: RefCount, A: Allocator + Clone> IntoIterator
    for &'a mut RefCountedVector<T, R, A>
{
    type Item = &'a mut T;
    type IntoIter = core::slice::IterMut<'a, T>;
    fn into_iter(self) -> core::slice::IterMut<'a, T> {
        self.as_mut_slice().iter_mut()
    }
}

impl<T, R, A, I> Index<I> for RefCountedVector<T, R, A>
where
    R: RefCount,
    A: Allocator,
    I: core::slice::SliceIndex<[T]>,
{
    type Output = <I as core::slice::SliceIndex<[T]>>::Output;
    fn index(&self, index: I) -> &Self::Output {
        self.as_slice().index(index)
    }
}

impl<T, R, A, I> IndexMut<I> for RefCountedVector<T, R, A>
where
    T: Clone,
    R: RefCount,
    A: Allocator + Clone,
    I: core::slice::SliceIndex<[T]>,
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

impl<T: Clone, R: RefCount, A: Allocator + Clone> DerefMut for RefCountedVector<T, R, A> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T: Debug, R: RefCount, A: Allocator> Debug for RefCountedVector<T, R, A> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> Result<(), core::fmt::Error> {
        self.as_slice().fmt(f)
    }
}

impl<T: Clone, A: Allocator + Clone> From<Vector<T, A>> for SharedVector<T, A> {
    fn from(vector: Vector<T, A>) -> Self {
        vector.into_shared()
    }
}

impl<T: Clone, A: Allocator + Clone> From<Vector<T, A>> for AtomicSharedVector<T, A> {
    fn from(vector: Vector<T, A>) -> Self {
        vector.into_shared_atomic()
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

        println!("concatenate");
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

    let _: Vector<()> = Vector::new();
}

#[test]
#[rustfmt::skip]
fn grow() {
    let mut a = Vector::with_capacity(0);

    a.push(num(1));
    a.push(num(2));
    a.push(num(3));

    a.extend_from_slice(&[num(4), num(5), num(6), num(7), num(8), num(9), num(10), num(12), num(12), num(13), num(14), num(15), num(16), num(17), num(18)]);

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


#[test]
fn shrink_to_zero() {
    let mut v: SharedVector<u32> = SharedVector::new();
    v.shrink_to(0);
}
