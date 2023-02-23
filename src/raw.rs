use std::alloc::{self, Layout};
use std::marker::PhantomData;
use std::mem;
use std::ptr::{self, NonNull};
use std::sync::atomic::{AtomicI32, Ordering::{Acquire, Release, Relaxed}};

pub type BufferSize = u32;

pub trait BufferHeader {
    fn with_capacity(cap: BufferSize) -> Self;
    unsafe fn add_ref(this: NonNull<Self>);
    unsafe fn release_ref(this:  NonNull<Self>) -> bool;
    fn len(&self) -> BufferSize;
    fn set_len(&mut self, val: BufferSize);
    fn capacity(&self) -> BufferSize;
    fn ref_count(&self) -> i32;
    unsafe fn global() -> Option<NonNull<Self>>  { None }
}

#[repr(C)]
pub struct VecHeader {
    pub cap: BufferSize,
    pub len: BufferSize,
}

pub struct HeaderInternal<R, A> {
    vec: VecHeader,
    ref_count: R,
    allocator: A,
}

// SAFETY: UniqueVector relies on AtomicHeader and Header having the same representation.
#[repr(C)]
pub struct AtomicHeader {
    pub vec: VecHeader,
    pub ref_count: AtomicI32,
    pub _pad: u32,
}

impl BufferHeader for AtomicHeader {
    fn with_capacity(cap: BufferSize) -> Self {
        AtomicHeader {
            vec: VecHeader { cap, len: 0 },
            ref_count: AtomicI32::new(1),
            _pad: 0,
        }
    }
    #[inline]
    unsafe fn add_ref(this: NonNull<Self>) {
        // Relaxed ordering is OK since the presence of the existing reference
        // prevents threads from deleting the buffer.
        this.as_ref().ref_count.fetch_add(1, Relaxed);
    }
    #[inline]
    unsafe fn release_ref(this:  NonNull<Self>) -> bool {
        this.as_ref().ref_count.fetch_sub(1, Release) == 1
    }
    #[inline] fn len(&self) -> BufferSize { self.vec.len }
    #[inline] fn set_len(&mut self, val: BufferSize) { self.vec.len = val; }
    #[inline] fn capacity(&self) -> BufferSize { self.vec.cap }
    #[inline] fn ref_count(&self) -> i32 {
        self.ref_count.load(Acquire)
    }
    unsafe fn global() -> Option<NonNull<Self>> {
        NonNull::new(&GLOBAL_EMPTY_BUFFER_ATOMIC as *const Self as *mut Self)
    }
}

#[repr(C)]
pub struct Header {
    pub vec: VecHeader,
    pub ref_count: i32,
    pub _pad: u32,
}

impl BufferHeader for Header {
    fn with_capacity(cap: BufferSize) -> Self {
        Header {
            vec: VecHeader { cap, len: 0 },
            ref_count: 1,
            _pad: 0,
        }
    }
    #[inline]
    unsafe fn add_ref(mut this: NonNull<Self>) {
        this.as_mut().ref_count += 1;
    }
    #[inline]
    unsafe fn release_ref(mut this:  NonNull<Self>) -> bool {
        let p = this.as_mut();
        p.ref_count -= 1;
        p.ref_count == 0
    }
    #[inline] fn len(&self) -> BufferSize { self.vec.len }
    #[inline] fn set_len(&mut self, val: BufferSize) { self.vec.len = val; }
    #[inline] fn capacity(&self) -> BufferSize { self.vec.cap }
    #[inline] fn ref_count(&self) -> i32 { self.ref_count }
    unsafe fn global() -> Option<NonNull<Self>> {
        None
        //NonNull::new(&GLOBAL_EMPTY_BUFFER as *const Self as *mut Self)
    }
}


// // A global empty header so that we can create empty shared buffers with allocating memory.
// static GLOBAL_EMPTY_BUFFER: Header = Header {
//     // The initial reference count is 1 so that it never gets to zero.
//     // this is important in order to ensure that the global empty buffer
//     // is never considered mutable (any live handle will contribute at least one reference
//     // meaning the ref_count should always be observably more than 1 if a RawBuffer points to it.)
//     ref_count: 1,
//     cap: 0,
//     len: 0,
//     _pad: 0,
// };

// A global empty header so that we can create empty shared buffers with allocating memory.
static GLOBAL_EMPTY_BUFFER_ATOMIC: AtomicHeader = AtomicHeader {
    vec: VecHeader { cap: 0, len: 0 },
    // The initial reference count is 1 so that it never gets to zero.
    // this is important in order to ensure that the global empty buffer
    // is never considered mutable (any live handle will contribute at least one reference
    // meaning the ref_count should always be observably more than 1 if a RawBuffer points to it.)
    ref_count: AtomicI32::new(1),
    _pad: 0,
};

/// Error type for APIs with fallible heap allocation
#[derive(Debug)]
pub enum AllocError {
    /// Overflow `usize::MAX` or other error during size computation
    CapacityOverflow,
    /// The allocator return an error
    Allocator {
        /// The layout that was passed to the allocator
        layout: Layout,
    },
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

pub unsafe fn dealloc<Header: BufferHeader, T>(ptr: NonNull<Header>, cap: BufferSize) {
    let layout = buffer_layout::<Header, T>(cap as usize).unwrap();

    alloc::dealloc(ptr.as_ptr() as *mut u8, layout);
}

#[cold]
fn capacity_error() -> AllocError {
    AllocError::CapacityOverflow
}


#[repr(transparent)]
pub struct HeaderBuffer<H: BufferHeader, T> {
    pub header: NonNull<H>,
    _marker: PhantomData<T>,
}

impl<H: BufferHeader, T> HeaderBuffer<H, T> {
    pub fn new_empty() -> Result<Self, AllocError> {
        Self::try_with_capacity(0)
    }

    pub unsafe fn from_raw(ptr: NonNull<H>) -> Self {
        HeaderBuffer { header: ptr, _marker: PhantomData }
    }

    #[inline(never)]
    pub fn try_with_capacity(mut cap: usize) -> Result<Self, AllocError> {
        if cap == 0 {
            unsafe {
                if let Some(header) = H::global() {
                    H::add_ref(header);
                    return Ok(HeaderBuffer {
                        header,
                        _marker: PhantomData,
                    })
                }    
            }

            cap = 16;  
        }

        unsafe {
            if cap > BufferSize::MAX as usize {
                return Err(capacity_error());
            }

            let layout = buffer_layout::<Header, T>(cap)?;
            let alloc: NonNull<H> = NonNull::new(alloc::alloc(layout))
                .ok_or(AllocError::Allocator { layout })?
                .cast();

            let header = H::with_capacity(cap as BufferSize);
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

    pub fn try_from_slice(data: &[T], cap: Option<usize>) -> Result<Self, AllocError>
    where
        T: Clone,
    {
        let len = data.len();
        let cap = cap.map(|cap| cap.max(len)).unwrap_or(len);

        if cap > BufferSize::MAX as usize {
            return Err(capacity_error());
        }

        let mut buffer = Self::try_with_capacity(cap)?;

        unsafe {
            buffer.header.as_mut().set_len(len as BufferSize);

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
        unsafe { self.header.as_ref().len() }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
       self.len() == 0
    }

    #[inline]
    pub fn capacity(&self) -> BufferSize {
        unsafe { self.header.as_ref().capacity() }
    }

    #[inline]
    pub fn remaining_capacity(&self) -> BufferSize {
        let h = unsafe { self.header.as_ref() };
        h.capacity() - h.len()
    }

    /// Allocates a duplicate of this SharedBuffer (fallible).
    pub fn try_clone_buffer(&self, new_cap: Option<BufferSize>) -> Result<Self, AllocError>
    where
        T: Clone,
    {
        unsafe {
            let header = self.header.as_ref();
            let len = header.len();
            let cap = if let Some(cap) = new_cap {
                cap
            } else {
                header.capacity()
            };

            if len > cap {
                return Err(capacity_error());
            }

            let mut clone = HeaderBuffer::try_with_capacity(cap as usize)?;

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
    {
        unsafe {
            let header = self.header.as_ref();
            let len = header.len();
            let cap = if let Some(cap) = new_cap {
                cap
            } else {
                header.capacity()
            };

            if len > cap {
                return Err(capacity_error());
            }
        
            let mut clone = HeaderBuffer::try_with_capacity(cap as usize)?;
        
            if len > 0 {
                std::ptr::copy_nonoverlapping(self.data_ptr(), clone.data_ptr(), len as usize);
                clone.set_len(len);
            }
        
            Ok(clone)    
        }
    }

    #[inline]
    pub fn is_unique(&self) -> bool {
        unsafe { self.header.as_ref().ref_count() == 1 }
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.data_ptr(), self.header.as_ref().len() as usize) }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.data_ptr(), self.header.as_ref().len() as usize) }
    }

    #[inline]
    pub fn new_ref(&self) -> Self {
        unsafe {
            H::add_ref(self.header);
        }
        HeaderBuffer {
            header: self.header,
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn data_ptr(&self) -> *mut T {
        unsafe { (self.header.as_ptr() as *mut u8).add(header_size::<H, T>()) as *mut T }
    }

    #[inline]
    pub fn ptr_eq(&self, other: &Self) -> bool {
        self.header == other.header
    }

    pub fn ref_count(&self) -> i32 {
        unsafe { self.header.as_ref().ref_count() }
    }
}

// SAFETY: All of the following methods require the buffer to be safely mutable. In other
// words, there is a single reference to the buffer (is_unique() returned true).
impl<H: BufferHeader, T> HeaderBuffer<H, T> {
    #[inline]
    pub unsafe fn set_len(&mut self, new_len: BufferSize) {
        debug_assert!(self.is_unique());
        self.header.as_mut().set_len(new_len);
    }

    pub unsafe fn try_push(&mut self, val: T) -> Result<(), AllocError> {
        debug_assert!(self.is_unique());
        let header = self.header.as_mut();
        let len = header.len();
        if len >= header.capacity() {
            return Err(capacity_error());
        }

        let address = self.data_ptr().add(len as usize);
        header.set_len(len + 1);

        ptr::write(address, val);

        Ok(())
    }

    // SAFETY: The capacity MUST be ensured beforehand.
    // The inline annotation really helps here.
    #[inline]
    pub unsafe fn push(&mut self, val: T) {
        debug_assert!(self.is_unique());
        let header = self.header.as_mut();
        let len = header.len();
        header.set_len(len + 1);

        let address = self.data_ptr().add(len as usize);
        ptr::write(address, val);
    }

    #[inline]
    pub unsafe fn pop(&mut self) -> Option<T> {
        debug_assert!(self.is_unique());
        let header = self.header.as_mut();
        let len = header.len();
        if len == 0 {
            return None;
        }

        let new_len = len - 1;
        header.set_len(new_len);

        let popped = ptr::read(self.data_ptr().add(new_len as usize));

        Some(popped)
    }

    pub unsafe fn try_push_slice(&mut self, data: &[T]) -> Result<(), AllocError>
    where
        T: Clone,
    {
        debug_assert!(self.is_unique());
        if data.len() > self.remaining_capacity() as usize {
            return Err(capacity_error());
        }

        let header = self.header.as_mut();
        let inital_len = header.len();
        header.set_len(inital_len + data.len() as BufferSize);

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
        let initial_len = header.len();

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

        header.set_len(initial_len + count);
    }

    pub unsafe fn clear(&mut self) {
        debug_assert!(self.is_unique());
        unsafe {
            let len = self.header.as_ref().len();
            drop_items(data_ptr::<H, T>(self.header), len);
            self.header.as_mut().set_len(0);
        }
    }

    pub unsafe fn move_data(&mut self, dst_buffer: &mut Self) {
        debug_assert!(self.is_unique());
        debug_assert!(dst_buffer.remaining_capacity() >= self.len());
        let src_header = self.header.as_mut();
        let dst_header = dst_buffer.header.as_mut();
        let len = src_header.len();
        if len > 0 {
            unsafe {
                let src = self.data_ptr();
                let dst = dst_buffer.data_ptr().add(dst_header.len() as usize);

                let inital_dst_len = dst_header.len();
                dst_header.set_len(inital_dst_len + len);
                src_header.set_len(0);

                ptr::copy_nonoverlapping(src, dst, len as usize);
            }
        }
    }
}

pub unsafe fn header_from_data_ptr<H, T>(data_ptr: NonNull<T>) -> NonNull<H> {
    NonNull::new_unchecked((data_ptr.as_ptr() as *mut u8).sub(header_size::<H, T>()) as *mut H)
}

impl<H: BufferHeader, T> Drop for HeaderBuffer<H, T> {
    fn drop(&mut self) {
        unsafe {
            if H::release_ref(self.header) {
                let cap = self.capacity();
                // See the implementation of std Arc for the need to use this fence. Note that
                // we only need it for the atomic reference counted version but I don't expect
                // this to make a measurable difference.
                std::sync::atomic::fence(Acquire);
                let len = self.header.as_ref().len();
                drop_items(data_ptr::<H, T>(self.header), len);
                dealloc::<H, T>(self.header, cap);
            }
        }
    }
}

#[test]
fn buffer_layout_alignemnt() {
    type B = Box<u32>;
    let layout = buffer_layout::<Header, B>(2).unwrap();
    assert_eq!(layout.align(), mem::size_of::<B>());

    let atomic_layout = buffer_layout::<AtomicHeader, B>(2).unwrap();

    assert_eq!(layout, atomic_layout);
}

pub struct Allocation {
    pub ptr: NonNull<u8>,
    pub size: usize,
}

pub trait Allocator {
    unsafe fn alloc(layout: Layout) -> Result<Allocation, AllocError>;
    unsafe fn dealloc(ptr: NonNull<u8>, layout: Layout);
}

pub struct GlobalAllocator;

impl Allocator for GlobalAllocator {
    unsafe fn alloc(layout: Layout) -> Result<Allocation, AllocError> {
        if let Some(ptr) = NonNull::new(std::alloc::alloc(layout)) {
            return Ok(Allocation {
                ptr,
                size: layout.size(),
            });
        }

        Err(AllocError::Allocator { layout })
    }

    unsafe fn dealloc(ptr: NonNull<u8>, layout: Layout) {
        std::alloc::dealloc(ptr.as_ptr(), layout)
    }
}
