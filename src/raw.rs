use core::alloc::Layout;
use core::cell::UnsafeCell;
use core::marker::PhantomData;
use core::mem;
use core::ptr::{self, NonNull};
use core::sync::atomic::{
    AtomicI32,
    Ordering::{Relaxed, Release},
};

pub use crate::alloc::{AllocError, Allocator, Global};

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
#[derive(Clone)]
pub struct VecHeader {
    pub cap: BufferSize,
    pub len: BufferSize,
}

impl VecHeader {
    fn remaining_capacity(&self) -> u32 { self.cap - self.len }
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

pub(crate) const fn header_size<Header, T>() -> usize {
    let a = mem::align_of::<T>();
    let s = mem::size_of::<Header>();
    let size = if a > s { a } else { s };

    // Favor L1 cache line alignment for large structs.
    let min = if mem::size_of::<T>() < 64 { 16 } else { 64 };
    if size < min {
        min
    } else {
        size
    }
}

pub fn buffer_layout<Header, T>(n: usize) -> Result<Layout, AllocError> {
    let size = mem::size_of::<T>().checked_mul(n).ok_or(AllocError)?;
    let align = mem::align_of::<Header>().max(mem::align_of::<T>());
    let align = if mem::size_of::<T>() < 64 {
        align
    } else {
        align.max(64)
    };
    let header_size = header_size::<Header, T>();

    Layout::from_size_align(header_size + size, align).map_err(|_| AllocError)
}

pub unsafe fn drop_items<T>(mut ptr: *mut T, count: u32) {
    for _ in 0..count {
        core::ptr::drop_in_place(ptr);
        ptr = ptr.add(1);
    }
}

pub unsafe fn dealloc<T, R, A: Allocator>(mut ptr: NonNull<Header<R, A>>, cap: BufferSize) {
    let layout = buffer_layout::<Header<R, A>, T>(cap as usize).unwrap();
    let allocator = ptr::read(&ptr.as_mut().allocator);
    allocator.deallocate(ptr.cast::<u8>(), layout);
}

#[cold]
pub fn alloc_error_cold() -> AllocError {
    AllocError
}

#[repr(transparent)]
pub struct HeaderBuffer<T, R: RefCount, A: Allocator> {
    pub header: NonNull<Header<R, A>>,
    _marker: PhantomData<T>,
}

impl<T, R: RefCount, A: Allocator> HeaderBuffer<T, R, A> {
    pub unsafe fn from_raw(ptr: NonNull<Header<R, A>>) -> Self {
        HeaderBuffer {
            header: ptr,
            _marker: PhantomData,
        }
    }

    #[inline]
    pub unsafe fn as_mut(&mut self) -> &mut Header<R, A> {
        self.header.as_mut()
    }

    #[inline]
    pub unsafe fn as_ref(&self) -> &Header<R, A> {
        self.header.as_ref()
    }

    #[inline]
    pub unsafe fn as_ptr(&self) -> *mut Header<R, A> {
        self.header.as_ptr()
    }

    #[inline]
    pub fn allocator(&self) -> &A {
        unsafe { &self.header.as_ref().allocator }
    }
}

pub unsafe fn move_data<T>(src_data: *mut T, src_vec: &mut VecHeader, dst_data: *mut T, dst_vec: &mut VecHeader) {
    debug_assert!(dst_vec.cap - dst_vec.len  >= src_vec.len);
    let len = src_vec.len;
    if len > 0 {
        unsafe {
            let dst = dst_data.add(dst_vec.len as usize);

            let inital_dst_len = dst_vec.len;
            dst_vec.len = inital_dst_len + len;
            src_vec.len = 0;

            ptr::copy_nonoverlapping(src_data, dst, len as usize);
        }
    }
}

pub unsafe fn extend_from_slice_assuming_capacity<T: Clone>(data: *mut T, vec: &mut VecHeader, slice: &[T])
where
    T: Clone,
{
    let len = slice.len() as u32;
    debug_assert!(len <= vec.remaining_capacity());

    let inital_len = vec.len;

    let mut ptr = data.add(inital_len as usize);

    for item in slice {
        ptr::write(ptr, item.clone());
        ptr = ptr.add(1)
    }

    vec.len += len;
}

// Returns true if the iterator was emptied.
pub unsafe fn extend_within_capacity<T, I: Iterator<Item = T>>(data: *mut T, vec: &mut VecHeader, iter: &mut I) -> bool {
    let inital_len = vec.len;

    let mut ptr = data.add(inital_len as usize);

    let mut count = 0;
    let max = vec.remaining_capacity();
    let mut finished = false;
    loop {
        if count == max {
            break;
        }
        if let Some(item) = iter.next() {
            ptr::write(ptr, item);
            ptr = ptr.add(1);
            count += 1;    
        } else {
            finished = true;
            break;
        }
    }

    vec.len += count;
    return finished;
}

#[inline]
pub unsafe fn pop<T>(data: *mut T, vec: &mut VecHeader) -> Option<T> {
    if vec.len == 0 {
        return None;
    }

    vec.len -= 1;

    Some(ptr::read(data.add(vec.len as usize)))
}

#[inline(always)]
pub unsafe fn push_assuming_capacity<T>(data: *mut T, vec: &mut VecHeader, val: T) {
    let dst = data.add(vec.len as usize);
    ptr::write(dst, val);
    vec.len += 1;
}

pub unsafe fn clear<T>(data: *mut T, vec: &mut VecHeader) {
    drop_items(data, vec.len);
    vec.len = 0;
}

pub fn assert_ref_count_layout<R>() {
    assert_eq!(mem::size_of::<R>(), mem::size_of::<i32>());
    assert_eq!(mem::align_of::<R>(), mem::align_of::<i32>());
}

#[inline(never)]
pub fn allocate_header_buffer<T, A>(
    mut cap: usize,
    allocator: &A,
) -> Result<(NonNull<u8>, usize), AllocError>
where
    A: Allocator,
{
    if cap == 0 {
        cap = 16;
    }

    if cap > BufferSize::MAX as usize {
        return Err(alloc_error_cold());
    }

    let layout = buffer_layout::<Header<DefaultRefCount, A>, T>(cap)?;
    let allocation = allocator.allocate(layout)?;
    let items_size = allocation.len() - header_size::<Header<DefaultRefCount, A>, T>();
    let size_of = mem::size_of::<T>();
    let real_capacity = if size_of == 0 {
        cap
    } else {
        items_size / size_of
    };

    Ok((allocation.cast(), real_capacity))
}

pub unsafe fn header_from_data_ptr<H, T>(data_ptr: NonNull<T>) -> NonNull<H> {
    NonNull::new_unchecked((data_ptr.as_ptr() as *mut u8).sub(header_size::<H, T>()) as *mut H)
}

#[test]
fn buffer_layout_alignemnt() {
    type B = Box<u32>;
    let layout = buffer_layout::<Header<DefaultRefCount, Global>, B>(2).unwrap();
    assert_eq!(layout.align(), mem::size_of::<B>());

    let atomic_layout = buffer_layout::<Header<AtomicRefCount, Global>, B>(2).unwrap();

    assert_eq!(layout, atomic_layout);
}
