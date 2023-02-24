use std::alloc::Layout;
use std::marker::PhantomData;
use std::mem;
use std::ptr::{self, NonNull};
use std::sync::atomic::{AtomicI32, Ordering::{Acquire, Release, Relaxed}};
use std::cell::UnsafeCell;

use crate::alloc::*;

pub type BufferSize = u32;

pub trait RefCount {
    unsafe fn add_ref(&self);
    unsafe fn release_ref(&self) -> bool;
    fn new(count: i32) -> Self;
    fn get(&self) -> i32;
}

pub struct DefaultRefCount(UnsafeCell<i32>);
pub struct AtomicRefCount(AtomicI32);

#[repr(C)]
pub struct VecHeader {
    pub cap: BufferSize,
    pub len: BufferSize,
}

#[repr(C)]
pub struct Header<R, A> {
    pub(crate) vec: VecHeader,
    pub(crate) ref_count: R,
    pub(crate) allocator: A,
}

impl RefCount for AtomicRefCount {
    #[inline]
    unsafe fn add_ref(&self) {
        // Relaxed ordering is OK since the presence of the existing reference
        // prevents threads from deleting the buffer.
        self.0.fetch_add(1, Relaxed);
    }

    #[inline]
    unsafe fn release_ref(&self) -> bool {
        self.0.fetch_sub(1, Release) == 1
    }

    #[inline]
    fn new(val: i32) -> Self {
        AtomicRefCount(AtomicI32::new(val))
    }

    #[inline]
    fn get(&self) -> i32 {
        self.0.load(Relaxed)
    }
}

impl RefCount for DefaultRefCount {
    #[inline]
    unsafe fn add_ref(&self) {
        *self.0.get() += 1;
    }

    #[inline]
    unsafe fn release_ref(&self) -> bool {
        let count = self.0.get();
        *count -= 1;
        *count == 0
    }

    #[inline]
    fn new(val: i32) -> Self {
        DefaultRefCount(UnsafeCell::new(val))
    }

    #[inline]
    fn get(&self) -> i32 {
        unsafe { *self.0.get() }
    }
}

#[inline]
pub unsafe fn data_ptr<Header, T>(header: NonNull<Header>) -> *mut T {
    (header.as_ptr() as *mut u8).add(header_size::<Header, T>()) as *mut T
}

const fn header_size<Header, T>() -> usize {
    let a = mem::align_of::<T>();
    let s = mem::size_of::<Header>();
    let size = if a > s { a } else { s };

    // Favor L1 cache line alignment for large structs.
    let min = if mem::size_of::<T>() < 64 { 16 } else { 64 };
    if size < min { min } else { size }
}

pub fn buffer_layout<Header, T>(n: usize) -> Result<Layout, AllocError> {
    let size = mem::size_of::<T>()
        .checked_mul(n)
        .ok_or(AllocError::CapacityOverflow)?;
    let align = mem::align_of::<Header>().max(mem::align_of::<T>());
    let align = if mem::size_of::<T>() < 64 { align } else { align.max(64) };
    let header_size = header_size::<Header, T>();

    Layout::from_size_align(header_size + size, align).map_err(|_| AllocError::CapacityOverflow)
}

pub unsafe fn drop_items<T>(mut ptr: *mut T, count: u32) {
    for _ in 0..count {
        std::ptr::drop_in_place(ptr);
        ptr = ptr.add(1);
    }
}

pub unsafe fn dealloc<T, R, A: Allocator>(mut ptr: NonNull<Header<R, A>>, cap: BufferSize) {
    let layout = buffer_layout::<Header<R, A>, T>(cap as usize).unwrap();
    let allocator = ptr::read(&ptr.as_mut().allocator);
    allocator.dealloc(ptr.cast::<u8>(), layout);
}

#[cold]
fn capacity_error() -> AllocError {
    AllocError::CapacityOverflow
}


#[repr(transparent)]
pub struct HeaderBuffer<T, R: RefCount, A: Allocator> {
    pub header: NonNull<Header<R, A>>,
    _marker: PhantomData<T>,
}

impl<T, R: RefCount, A: Allocator> HeaderBuffer<T, R, A> {
    pub unsafe fn from_raw(ptr: NonNull<Header<R, A>>) -> Self {
        HeaderBuffer { header: ptr, _marker: PhantomData }
    }

    #[inline(never)]
    pub fn try_with_capacity(mut cap: usize, allocator: A) -> Result<Self, AllocError>
    where
        A: Allocator,
        R: RefCount,
    {
        if cap == 0 {
            cap = 16;  
        }

        unsafe {
            if cap > BufferSize::MAX as usize {
                return Err(capacity_error());
            }

            let layout = buffer_layout::<Header<R, A>, T>(cap)?;
            let allocation = allocator.alloc(layout)?;
            // TODO: allocation could provide more capcity than what was requrested.
            let alloc: NonNull<Header<R, A>> = allocation.ptr.cast();

            let header = Header {
                vec: VecHeader { cap: cap as BufferSize, len: 0 },
                ref_count: R::new(1),
                allocator,
            };

            ptr::write(
                alloc.as_ptr(),
                header,
            );

            Ok(HeaderBuffer {
                header: alloc,
                _marker: PhantomData,
            })
        }
    }

    pub fn try_from_slice(data: &[T], cap: Option<usize>, allocator: A) -> Result<Self, AllocError>
    where
        T: Clone,
        R: RefCount,
        A: Allocator,
    {
        let len = data.len();
        let cap = cap.map(|cap| cap.max(len)).unwrap_or(len);

        if cap > BufferSize::MAX as usize {
            return Err(capacity_error());
        }

        let mut buffer = Self::try_with_capacity(cap, allocator)?;

        unsafe {
            buffer.header.as_mut().vec.len = len as BufferSize;

            let mut ptr = buffer.data_ptr();

            for item in data {
                ptr::write(ptr, item.clone());
                ptr = ptr.add(1)
            }
        }

        Ok(buffer)
    }

    #[inline]
    pub fn len(&self) -> BufferSize {
        unsafe { self.header.as_ref().vec.len }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
       self.len() == 0
    }

    #[inline]
    pub fn capacity(&self) -> BufferSize {
        unsafe { self.header.as_ref().vec.cap }
    }

    #[inline]
    pub fn remaining_capacity(&self) -> BufferSize {
        let h = unsafe { self.header.as_ref() };
        h.vec.cap - h.vec.len
    }

    /// Allocates a duplicate of this SharedBuffer (fallible).
    pub fn try_clone_buffer(&self, new_cap: Option<BufferSize>) -> Result<Self, AllocError>
    where
        T: Clone,
        R: RefCount,
        A: Allocator,
    {
        unsafe {
            let header = self.header.as_ref();
            let len = header.vec.len;
            let cap = if let Some(cap) = new_cap {
                cap
            } else {
                header.vec.cap
            };
            let allocator = header.allocator.clone();

            if len > cap {
                return Err(capacity_error());
            }

            let mut clone = HeaderBuffer::try_with_capacity(cap as usize, allocator)?;

            if len > 0 {
                let mut src = self.data_ptr();
                let mut dst = clone.data_ptr();
                for _ in 0..len {
                    ptr::write(dst, (*src).clone());
                    src = src.add(1);
                    dst = dst.add(1);
                }
            
                clone.set_len(len);    
            }
        
            Ok(clone)    
        }
    }

    /// Allocates a duplicate of this SharedBuffer (fallible).
    pub fn try_copy_buffer(&self, new_cap: Option<BufferSize>) -> Result<Self, AllocError>
    where
        T: Copy,
        R: RefCount,
        A: Allocator,
    {
        unsafe {
            let header = self.header.as_ref();
            let len = header.vec.len;
            let cap = if let Some(cap) = new_cap {
                cap
            } else {
                header.vec.cap
            };

            if len > cap {
                return Err(capacity_error());
            }
        
            let allocator = header.allocator.clone();
            let mut clone = HeaderBuffer::try_with_capacity(cap as usize, allocator)?;
        
            if len > 0 {
                std::ptr::copy_nonoverlapping(self.data_ptr(), clone.data_ptr(), len as usize);
                clone.set_len(len);
            }
        
            Ok(clone)    
        }
    }

    #[inline]
    pub fn is_unique(&self) -> bool where R: RefCount {
        unsafe { self.header.as_ref().ref_count.get() == 1 }
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.data_ptr(), self.header.as_ref().vec.len as usize) }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.data_ptr(), self.header.as_ref().vec.len as usize) }
    }

    #[inline]
    pub fn new_ref(&self) -> Self where R: RefCount {
        unsafe {
            self.header.as_ref().ref_count.add_ref();
        }
        HeaderBuffer {
            header: self.header,
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn data_ptr(&self) -> *mut T {
        unsafe { (self.header.as_ptr() as *mut u8).add(header_size::<Header<R, A>, T>()) as *mut T }
    }

    #[inline]
    pub fn clone_allocator(&self) -> A {
        unsafe { self.header.as_ref().allocator.clone() }
    }

    #[inline]
    pub fn ptr_eq(&self, other: &Self) -> bool {
        self.header == other.header
    }
}

// SAFETY: All of the following methods require the buffer to be safely mutable. In other
// words, there is a single reference to the buffer (is_unique() returned true).
impl<T, R: RefCount, A: Allocator> HeaderBuffer<T, R, A> {
    #[inline]
    pub unsafe fn set_len(&mut self, new_len: BufferSize) {
        debug_assert!(self.is_unique());
        self.header.as_mut().vec.len = new_len;
    }

    pub unsafe fn try_push(&mut self, val: T) -> Result<(), AllocError> {
        debug_assert!(self.is_unique());
        let header = self.header.as_mut();
        let len = header.vec.len;
        if len >= header.vec.cap {
            return Err(capacity_error());
        }

        let address = self.data_ptr().add(len as usize);
        header.vec.len += 1;

        ptr::write(address, val);

        Ok(())
    }

    // SAFETY: The capacity MUST be ensured beforehand.
    // The inline annotation really helps here.
    #[inline]
    pub unsafe fn push(&mut self, val: T) {
        debug_assert!(self.is_unique());
        let header = self.header.as_mut();
        let len = header.vec.len;
        header.vec.len += 1;

        let address = self.data_ptr().add(len as usize);
        ptr::write(address, val);
    }

    #[inline]
    pub unsafe fn pop(&mut self) -> Option<T> {
        debug_assert!(self.is_unique());
        let header = self.header.as_mut();
        let len = header.vec.len;
        if len == 0 {
            return None;
        }

        let new_len = len - 1;
        header.vec.len = new_len;

        let popped = ptr::read(self.data_ptr().add(new_len as usize));

        Some(popped)
    }

    pub unsafe fn try_push_slice(&mut self, data: &[T]) -> Result<(), AllocError>
    where
        T: Clone,
        R: RefCount,
    {
        debug_assert!(self.is_unique());
        if data.len() > self.remaining_capacity() as usize {
            return Err(capacity_error());
        }

        let header = self.header.as_mut();
        let inital_len = header.vec.len;
        header.vec.len = inital_len + data.len() as BufferSize;

        let mut ptr = self.data_ptr().add(inital_len as usize);

        for item in data {
            ptr::write(ptr, item.clone());
            ptr = ptr.add(1)
        }

        Ok(())
    }

    pub unsafe fn try_extend(
        &mut self,
        iter: &mut impl Iterator<Item = T>,
    ) -> Result<(), AllocError> {
        debug_assert!(self.is_unique());
        let (min_len, _upper_bound) = iter.size_hint();
        if min_len > self.remaining_capacity() as usize {
            return Err(capacity_error());
        }

        if min_len > 0 {
            self.extend_n(iter, min_len as BufferSize);
        }

        for item in iter {
            self.try_push(item)?;
        }

        Ok(())
    }

    pub unsafe fn extend_n(&mut self, iter: &mut impl Iterator<Item = T>, n: BufferSize) {
        debug_assert!(self.is_unique());
        let header = self.header.as_mut();
        let initial_len = header.vec.len;

        let mut ptr = self.data_ptr().add(initial_len as usize);
        let mut count = 0;
        for item in iter {
            if count == n {
                break;
            }
            ptr::write(ptr, item);
            ptr = ptr.add(1);
            count += 1;
        }

        header.vec.len = initial_len + count;
    }

    pub unsafe fn clear(&mut self) {
        debug_assert!(self.is_unique());
        unsafe {
            let len = self.header.as_ref().vec.len;
            drop_items(data_ptr::<Header<R, A>, T>(self.header), len);
            self.header.as_mut().vec.len = 0;
        }
    }

    pub unsafe fn move_data(&mut self, dst_buffer: &mut Self) {
        debug_assert!(self.is_unique());
        debug_assert!(dst_buffer.remaining_capacity() >= self.len());
        let src_header = self.header.as_mut();
        let dst_header = dst_buffer.header.as_mut();
        let len = src_header.vec.len;
        if len > 0 {
            unsafe {
                let src = self.data_ptr();
                let dst = dst_buffer.data_ptr().add(dst_header.vec.len as usize);

                let inital_dst_len = dst_header.vec.len;
                dst_header.vec.len = inital_dst_len + len;
                src_header.vec.len = 0;

                ptr::copy_nonoverlapping(src, dst, len as usize);
            }
        }
    }
}

pub unsafe fn header_from_data_ptr<H, T>(data_ptr: NonNull<T>) -> NonNull<H> {
    NonNull::new_unchecked((data_ptr.as_ptr() as *mut u8).sub(header_size::<H, T>()) as *mut H)
}

impl<T, R: RefCount, A: Allocator> Drop for HeaderBuffer<T, R, A> {
    fn drop(&mut self) {
        unsafe {
            if self.header.as_ref().ref_count.release_ref() {
                let cap = self.capacity();
                // See the implementation of std Arc for the need to use this fence. Note that
                // we only need it for the atomic reference counted version but I don't expect
                // this to make a measurable difference.
                std::sync::atomic::fence(Acquire);
                let len = self.header.as_ref().vec.len;
                drop_items(data_ptr::<Header<R, A>, T>(self.header), len);
                dealloc::<T, R, A>(self.header, cap);
            }
        }
    }
}

#[test]
fn buffer_layout_alignemnt() {
    type B = Box<u32>;
    let layout = buffer_layout::<Header<DefaultRefCount, GlobalAllocator>, B>(2).unwrap();
    assert_eq!(layout.align(), mem::size_of::<B>());

    let atomic_layout = buffer_layout::<Header<AtomicRefCount, GlobalAllocator>, B>(2).unwrap();

    assert_eq!(layout, atomic_layout);
}

