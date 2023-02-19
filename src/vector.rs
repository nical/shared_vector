use std::{ptr, mem};
use std::ops::{Index, IndexMut};
use std::ptr::NonNull;

use crate::raw::{self, AllocError, BufferSize, HeaderBuffer, BufferHeader, VecHeader};

pub trait ReferenceCount {
    type Header: BufferHeader;
}

pub struct DefaultRefCount;
pub struct AtomicRefCount;

impl ReferenceCount for DefaultRefCount { type Header = raw::Header; }
impl ReferenceCount for AtomicRefCount { type Header = raw::AtomicHeader; }

pub type AtomicSharedVector<T> = RefCountedVector<T, AtomicRefCount>;
pub type SharedVector<T> = RefCountedVector<T, DefaultRefCount>;

/// A heap allocated, reference counted, immutable contiguous buffer containing elements of type `T`.
///
/// Similar in principle to `Arc<[T]>`. It can be converted into a `UniqueVector<T>` for
/// free if there is only a single reference to the RefCountedVector alive.
///
/// # "Immutable" vectors
///
/// This type contains mutable methods like `push` and `pop`. These internally allocate a new buffer
/// if the buffer is shared (there are more than one reference to it). When there is a single reference,
/// these mutable operation simply update the existing buffer.
///
/// In other words, this type behaves like an [immutable (or persistent) data structure](https://en.wikipedia.org/wiki/Persistent_data_structure)
/// Actual mutability only happens under the hood as an optimization when a single reference exists.
#[repr(transparent)]
pub struct RefCountedVector<T, R: ReferenceCount = DefaultRefCount> {
    inner: HeaderBuffer<R::Header, T>,
}

impl<T, R: ReferenceCount> RefCountedVector<T, R> {
    /// Creates an empty shared buffer without allocating memory.
    #[inline]
    pub fn new() -> Self {
        RefCountedVector {
            inner: HeaderBuffer::new_empty().unwrap(),
        }
    }

    #[inline]
    pub fn with_capacity(cap: usize) -> Self {
        RefCountedVector {
            inner: HeaderBuffer::try_with_capacity(cap).unwrap(),
        }
    }

    #[inline]
    pub fn from_slice(data: &[T]) -> Self
    where
        T: Clone,
    {
        RefCountedVector {
            inner: HeaderBuffer::try_from_slice(data, None).unwrap(),
        }
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

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.inner.as_slice()
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut[T] where T: Clone {
        self.ensure_unique();
        self.inner.as_mut_slice()
    }


    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.as_slice().iter()
    }

    #[inline]
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> where T: Clone {
        self.as_mut_slice().iter_mut()
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
    pub fn copy_buffer(&self) -> Self
    where
        T: Copy,
    {
        RefCountedVector {
            inner: self.inner.try_copy_buffer(None).unwrap(),
        }
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
    pub fn into_unique(mut self) -> UniqueVector<T>
    where
        T: Clone,
    {
        self.ensure_unique();

        unsafe {
            let data = NonNull::new_unchecked(self.inner.data_ptr());
            let len = self.len() as BufferSize;
            let cap = self.capacity() as BufferSize;

            mem::forget(self);

            UniqueVector { data, len, cap }
        }
    }

    #[inline]
    pub fn first(&self) -> Option<&T> {
        unsafe { self.inner.first().map(|ptr| &*ptr) }
    }

    #[inline]
    pub fn last(&self) -> Option<&T> {
        unsafe { self.inner.last().map(|ptr| &*ptr) }
    }

    #[inline]
    pub fn first_mut(&mut self) -> Option<&mut T> where T: Clone {
        self.ensure_unique();
        unsafe { self.inner.first().map(|ptr| &mut *ptr) }
    }

    #[inline]
    pub fn last_mut(&mut self) -> Option<&mut T>  where T: Clone {
        self.ensure_unique();
        unsafe { self.inner.last().map(|ptr| &mut *ptr) }
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

    pub fn pop(&mut self) -> Option<T>
    where
        T: Clone,
    {
        self.ensure_unique();
        unsafe { self.inner.pop() }
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

    pub fn clear(&mut self) {
        if self.is_unique() {
            unsafe { self.inner.clear(); }
            return;
        }

        *self = Self::with_capacity(self.capacity());
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

    /// Returns a buffer that can be safely mutated and has enough extra capacity to
    /// add `additional` more items.
    #[inline]
    pub fn reserve(&mut self, additional: usize)
    where
        T: Clone,
    {
        let is_unique = self.is_unique();
        let enough_capacity = self.remaining_capacity() >= additional;

        if !is_unique || !enough_capacity {
            // Hopefully the least common case.
            self.realloc(is_unique, enough_capacity, additional);
        }
    }

    #[cold]
    fn realloc(&mut self, is_unique: bool, enough_capacity: bool, additional: usize)
    where
        T: Clone,
    {
        let new_cap = if enough_capacity {
            self.capacity()
        } else {
            grow_amortized(self.len(), additional)
        };

        if is_unique {
            // The buffer is not large enough, we'll have to create a new one, however we
            // know that we have the only reference to it so we'll move the data with
            // a simple memcpy instead of cloning it.
            unsafe {
                let mut dst = Self::with_capacity(new_cap);
                let len = self.len();
                if len > 0 {
                    ptr::copy_nonoverlapping(
                        self.inner.data_ptr(),
                        dst.inner.data_ptr(),
                        len as usize,
                    );
                    dst.inner.set_len(len as BufferSize);
                    self.inner.set_len(0);
                }

                self.inner = dst.inner;
                return;
            }
        }

        // The slowest path, we pay for both the new allocation and the need to clone
        // each item one by one.
        self.inner = HeaderBuffer::try_from_slice(self.as_slice(), Some(new_cap)).unwrap();
    }

    /// Returns the concatenation of two vectors.
    pub fn concatenate(mut self, other: Self) -> Self
    where
        T: Clone,
    {
        self.push_vector(other);

        self
    }

    pub fn push_vector(&mut self, mut other: Self) where T: Clone {
        self.reserve(other.len());

        unsafe {
            if other.is_unique() {
                // Fast path: memcpy
                other.inner.move_data(&mut self.inner);
            } else {
                // Slow path, clone each item.
                self.inner.try_push_slice(other.as_slice()).unwrap();
            }
        }
    }

    pub fn ref_count(&self) -> i32 {
        self.inner.ref_count()
    }

    #[allow(unused)]
    pub(crate) fn addr(&self) -> *const u8 {
        self.inner.header.as_ptr() as *const u8
    }
}

unsafe impl<T: Sync> Send for AtomicSharedVector<T> {}

impl<T, R: ReferenceCount> Clone for RefCountedVector<T, R> {
    fn clone(&self) -> Self {
        self.new_ref()
    }
}

impl<T: PartialEq<T>, R: ReferenceCount> PartialEq<RefCountedVector<T, R>> for RefCountedVector<T, R> {
    fn eq(&self, other: &Self) -> bool {
        self.ptr_eq(other) || self.as_slice() == other.as_slice()
    }
}

impl<T: PartialEq<T>, R: ReferenceCount> PartialEq<&[T]> for RefCountedVector<T, R> {
    fn eq(&self, other: &&[T]) -> bool {
        self.as_slice() == *other
    }
}

impl<T, R: ReferenceCount> AsRef<[T]> for RefCountedVector<T, R> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T, R: ReferenceCount> Default for RefCountedVector<T, R> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, T, R: ReferenceCount> IntoIterator for &'a RefCountedVector<T, R> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;
    fn into_iter(self) -> std::slice::Iter<'a, T> {
        self.as_slice().iter()
    }
}

impl<'a, T: Clone, R: ReferenceCount> IntoIterator for &'a mut RefCountedVector<T, R> {
    type Item = &'a mut T;
    type IntoIter = std::slice::IterMut<'a, T>;
    fn into_iter(self) -> std::slice::IterMut<'a, T> {
        self.as_mut_slice().iter_mut()
    }
}

impl<T, R: ReferenceCount, I> Index<I> for RefCountedVector<T, R>
where
    I: std::slice::SliceIndex<[T]>,
{
    type Output = <I as std::slice::SliceIndex<[T]>>::Output;
    fn index(&self, index: I) -> &Self::Output {
        self.as_slice().index(index)
    }
}



/// A heap allocated, mutable contiguous buffer containing elements of type `T`.
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
    data: NonNull<T>,
    len: BufferSize,
    cap: BufferSize,
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
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.as_slice().iter()
    }

    #[inline]
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.as_mut_slice().iter_mut()
    }

    #[inline]
    pub fn first(&self) -> Option<&T> {
        if self.len == 0 {
            return None;
        }

        unsafe { Some(self.data.as_ref()) }
    }

    #[inline]
    pub fn last(&self) -> Option<&T> {
        if self.len == 0 {
            return None;
        }

        let idx = self.len() - 1;
        unsafe { self.data_ptr().add(idx).as_ref() }
    }

    #[inline]
    pub fn first_mut(&mut self) -> Option<&mut T> {
        if self.len == 0 {
            return None;
        }

        unsafe { Some(self.data.as_mut()) }
    }

    #[inline]
    pub fn last_mut(&mut self) -> Option<&mut T> {
        if self.len == 0 {
            return None;
        }

        let idx = self.len() - 1;
        unsafe { self.data_ptr().add(idx).as_mut() }
    }

    #[inline]
    pub fn push(&mut self, val: T) {
        let len = self.len;
        let cap = self.cap;
        if cap == len {
            self.realloc(1);
        }

        unsafe {
            self.len += 1;
            let dst = self.data_ptr().add(len as usize);
            ptr::write(dst, val);
        }
    }

    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }

        self.len -= 1;

        unsafe {
            Some(ptr::read(self.data_ptr().add(self.len as usize)))
        }
    }

    pub fn push_slice(&mut self, data: &[T])
    where
        T: Clone,
    {
        self.extend(data.iter().cloned())
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

    #[inline]
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
        let mut clone = Self::with_capacity(cap);
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

    fn try_realloc(&mut self, additional: usize) -> Result<(), AllocError> {
        if additional < self.remaining_capacity() {
            return  Ok(());
        }

        let new_cap = grow_amortized(self.len(), additional);
        if new_cap < self.len() {
            return Err(AllocError::CapacityOverflow);
        }

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

    // Note: Marking this #[inline(never)] is a pretty large regression in the push benchmark.
    #[cold]
    fn realloc(&mut self, additional: usize) {
        self.try_realloc(additional).unwrap();
    }

    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        if self.remaining_capacity() < additional {
            self.realloc(additional);
        }
    }

    pub fn push_vector(&mut self, mut other: Self) where T: Clone {
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

fn grow_amortized(len: usize, additional: usize) -> usize {
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

// In order to give us a chance to catch leaks and double-frees, test with values that implement drop.
#[cfg(test)]
fn num(val: u32) -> Box<u32> {
    Box::new(val)
}

#[test]
fn basic_unique() {
    let mut a = UniqueVector::with_capacity(256);

    a.push(num(0));
    a.push(num(1));
    a.push(num(2));

    let a = a.into_shared();

    assert_eq!(a.len(), 3);

    assert_eq!(a.as_slice(), &[num(0), num(1), num(2)]);

    assert!(a.is_unique());

    let b = UniqueVector::from_slice(&[num(0), num(1), num(2), num(3), num(4)]);

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

    let mut d = UniqueVector::with_capacity(64);
    d.push_slice(&[num(0), num(1), num(2)]);
    d.push_slice(&[]);
    d.push_slice(&[num(3), num(4)]);

    assert_eq!(d.as_slice(), &[num(0), num(1), num(2), num(3), num(4)]);
}

#[test]
fn basic_shared() {
    basic_shared_impl::<DefaultRefCount>();
    basic_shared_impl::<AtomicRefCount>();

    fn basic_shared_impl<R: ReferenceCount>() {
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
    // TODO: The behavior is different for SharedVector because it does not use a global header.
    let a: AtomicSharedVector<u32> = AtomicSharedVector::new();
    assert!(!a.is_unique());
    {
        let b: AtomicSharedVector<u32> = AtomicSharedVector::new();
        assert!(!b.is_unique());
        assert!(a.ptr_eq(&b));
    }

    assert!(!a.is_unique());

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

#[test]
fn ensure_unique_empty() {
    let mut v: SharedVector<u32> = SharedVector::new();
    v.ensure_unique();
}
