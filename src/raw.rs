use core::alloc::Layout;
use core::cell::UnsafeCell;
use core::marker::PhantomData;
use core::mem;
use core::ptr::{self, NonNull};
use core::sync::atomic::{
    AtomicI32,
    Ordering::{Relaxed, Release},
};

pub use crate::alloc::{AllocError, Allocator};

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

/// Alignment of the shared empty header. Also the largest item and allocator
/// alignment it can accommodate.
const EMPTY_HEADER_ALIGN: usize = 64;

/// The reference count stored in the shared empty header.
///
/// It is never incremented nor decremented, it only needs to be different from
/// one so that the vectors pointing to the shared empty header are never
/// considered unique (mutating them must allocate a real buffer).
const EMPTY_HEADER_REF_COUNT: i32 = i32::MAX;

/// Storage for the header shared by all of the empty reference counted vectors.
///
/// This mirrors the layout of `Header<R, A>` for any reference counting scheme
/// `R` (they are all laid out like an `i32`, see `assert_ref_count_layout`) and
/// any zero-sized allocator `A`. The padding covers the allocator field as well
/// as the alignment requirements of the header itself.
#[repr(C, align(64))]
struct EmptyHeader {
    vec: VecHeader,
    ref_count: AtomicI32,
    padding: [u8; EMPTY_HEADER_ALIGN - mem::size_of::<VecHeader>() - mem::size_of::<i32>()],
}

/// A single header shared by all empty reference counted vectors, which lets
/// them exist without allocating.
///
/// The reference count is deliberately never touched (see `HeaderBuffer::add_ref`
/// and `HeaderBuffer::release_ref`), so this static is only ever read from. That
/// is what makes it safe to share it between threads and between reference
/// counting schemes: a data race needs at least one writer.
///
/// The type of the reference count field is atomic only so that the static is
/// placed in writable memory rather than read-only memory, out of caution: the
/// vectors observe it through `UnsafeCell` and `AtomicI32` references.
static EMPTY_HEADER: EmptyHeader = EmptyHeader {
    vec: VecHeader { cap: 0, len: 0 },
    ref_count: AtomicI32::new(EMPTY_HEADER_REF_COUNT),
    padding: [0; EMPTY_HEADER_ALIGN - mem::size_of::<VecHeader>() - mem::size_of::<i32>()],
};

/// Returns the shared empty header for a vector of `T` items with the `R` reference
/// counting scheme and the `A` allocator, or null if it can't be used with these
/// parameters.
///
/// There can only be a single static header, so it can only stand in for allocators
/// that have nothing to store, and its address has to satisfy the alignment
/// requirements of both the header and the items.
#[inline(always)]
pub fn shared_empty_header<T, R: RefCount, A: Allocator>() -> *mut Header<R, A> {
    if mem::size_of::<A>() != 0
        || mem::needs_drop::<A>()
        || mem::align_of::<A>() > EMPTY_HEADER_ALIGN
        || mem::align_of::<T>() > EMPTY_HEADER_ALIGN
    {
        return ptr::null_mut();
    }

    debug_assert!(mem::size_of::<Header<R, A>>() <= mem::size_of::<EmptyHeader>());
    debug_assert!(mem::align_of::<Header<R, A>>() <= EMPTY_HEADER_ALIGN);
    debug_assert_eq!(mem::size_of::<R>(), mem::size_of::<i32>());

    &EMPTY_HEADER as *const EmptyHeader as *mut Header<R, A>
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

    /// Returns true if this buffer is the header shared by all empty vectors.
    ///
    /// When `shared_empty_header` returns null (the parameters can't use the shared
    /// header), this folds into a comparison of a non-null pointer against null.
    #[inline]
    pub fn is_shared_empty(&self) -> bool {
        ptr::eq(
            self.header.as_ptr() as *const u8,
            shared_empty_header::<T, R, A>() as *const u8,
        )
    }

    /// Adds a reference to this buffer.
    #[inline]
    pub unsafe fn add_ref(&self) {
        // The shared empty header is immortal and shared between threads: its
        // reference count must not be written to.
        if self.is_shared_empty() {
            return;
        }

        self.as_ref().ref_count.add_ref();
    }

    /// Removes a reference from this buffer, returning true if it was the last one
    /// and the buffer must now be destroyed.
    #[inline]
    pub unsafe fn release_ref(&self) -> bool {
        if self.is_shared_empty() {
            return false;
        }

        self.as_ref().ref_count.release_ref()
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
        cap = 8;
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

// The shared empty header stands in for `Header<R, A>` values, so their layouts have
// to agree.
#[test]
fn empty_header_layout() {
    pub use crate::alloc::Global;

    type H = Header<AtomicRefCount, Global>;

    assert!(mem::size_of::<H>() <= mem::size_of::<EmptyHeader>());
    assert!(mem::align_of::<H>() <= mem::align_of::<EmptyHeader>());
    assert_eq!(mem::size_of::<EmptyHeader>(), EMPTY_HEADER_ALIGN);

    let header = H {
        vec: VecHeader { cap: 0, len: 0 },
        ref_count: AtomicRefCount::new(1),
        allocator: Global,
    };

    let offset = |field: *const u8, base: *const u8| field as usize - base as usize;
    let base = &header as *const H as *const u8;
    let empty_base = &EMPTY_HEADER as *const EmptyHeader as *const u8;

    assert_eq!(
        offset(&header.vec as *const _ as *const u8, base),
        offset(&EMPTY_HEADER.vec as *const _ as *const u8, empty_base),
    );
    assert_eq!(
        offset(&header.ref_count as *const _ as *const u8, base),
        offset(&EMPTY_HEADER.ref_count as *const _ as *const u8, empty_base),
    );

    // The header is only ever read from, its reference count is never touched.
    let header = shared_empty_header::<u32, DefaultRefCount, Global>();
    assert!(!header.is_null());
    unsafe {
        assert_eq!((*header).vec.cap, 0);
        assert_eq!((*header).vec.len, 0);
        assert_eq!((*header).ref_count.get(), EMPTY_HEADER_REF_COUNT);
    }

    // Items and allocators that are more aligned than the shared header can't use it.
    #[repr(align(128))]
    struct OverAligned;
    assert!(shared_empty_header::<OverAligned, DefaultRefCount, Global>().is_null());
}

#[test]
fn buffer_layout_alignemnt() {
    pub use crate::alloc::Global;

    type B = Box<u32>;
    let layout = buffer_layout::<Header<DefaultRefCount, Global>, B>(2).unwrap();
    assert_eq!(layout.align(), mem::size_of::<B>());

    let atomic_layout = buffer_layout::<Header<AtomicRefCount, Global>, B>(2).unwrap();

    assert_eq!(layout, atomic_layout);
}
