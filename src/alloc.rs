use std::{ptr::{self, NonNull}, sync::atomic::{AtomicUsize, AtomicI32, Ordering}, cell::UnsafeCell};
pub use std::alloc::Layout;

use crate::UniqueVector;

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

/// The pointer and size resulting from a memory allocation.
///
/// Note: to make `Allocator` more like the standard version it woud be nice to replace
/// this struct with `NonNull<[u8]>`. That probably requires `NonNull::slice_from_raw_parts`
/// which is nightly-only at the moment.
pub struct Allocation {
    pub ptr: NonNull<u8>,
    pub size: usize,
}

/// A very similar trait to `std::alloc::Alloc` until the latter is available
/// on stable rust.
pub trait Allocator: Clone {
    unsafe fn alloc(&self, layout: Layout) -> Result<Allocation, AllocError>;

    unsafe fn dealloc(&self, ptr: NonNull<u8>, layout: Layout);

    unsafe fn alloc_zeroed(&self, layout: Layout) -> Result<Allocation, AllocError> {
        let mut alloc = self.alloc(layout);
        if let Ok(alloc) = &mut alloc {
            unsafe { ptr::write_bytes(alloc.ptr.as_ptr(), 0, alloc.size) };
        }

        alloc
    }

    unsafe fn realloc(&self, ptr: NonNull<u8>, layout: Layout, new_size: usize) -> Result<Allocation, AllocError> {
        // SAFETY: the caller must ensure that the `new_size` does not overflow.
        // `layout.align()` comes from a `Layout` and is thus guaranteed to be valid.
        let new_layout = unsafe { Layout::from_size_align_unchecked(new_size, layout.align()) };
        // SAFETY: the caller must ensure that `new_layout` is greater than zero.
        let new_alloc = unsafe { self.alloc(new_layout) };
        if let Ok(new_alloc) = &new_alloc {
            // SAFETY: the previously allocated block cannot overlap the newly allocated block.
            // The safety contract for `dealloc` must be upheld by the caller.
            unsafe {
                let size = std::cmp::min(layout.size(), new_size);
                ptr::copy_nonoverlapping(ptr.as_ptr(), new_alloc.ptr.as_ptr(), size);
                self.dealloc(ptr, layout);
            }
        }

        new_alloc
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GlobalAllocator;

unsafe impl Sync for GlobalAllocator {}

impl Allocator for GlobalAllocator {
    unsafe fn alloc(&self, layout: Layout) -> Result<Allocation, AllocError> {
        if let Some(ptr) = NonNull::new(std::alloc::alloc(layout)) {
            return Ok(Allocation {
                ptr,
                size: layout.size(),
            });
        }

        Err(AllocError::Allocator { layout })
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> Result<Allocation, AllocError> {
        let mut alloc = self.alloc(layout);
        if let Ok(alloc) = &mut alloc {
            unsafe { ptr::write_bytes(alloc.ptr.as_ptr(), 0, alloc.size) };
        }

        alloc
    }

    unsafe fn dealloc(&self, ptr: NonNull<u8>, layout: Layout) {
        std::alloc::dealloc(ptr.as_ptr(), layout)
    }
}

pub trait AsMutBytes {
    unsafe fn as_mut_bytes(&self) -> *mut u8;
    fn size(&self) -> usize;
}

impl AsMutBytes for Box<[u8]> {
    unsafe fn as_mut_bytes(&self) -> *mut u8 {
        self.as_ptr() as *mut u8
    }
    fn size(&self) -> usize {
        self.len()
    }
}

impl<'l> AsMutBytes for &'l mut [u8] {
    unsafe fn as_mut_bytes(&self) -> *mut u8 {
        self.as_ptr() as *mut u8
    }

    fn size(&self) -> usize {
        self.len()
    }
}

/// A very simple thread-safe bump allocator using a single pre-allocatec buffer.
pub struct BoundedBumpAllocator<Buffer: AsMutBytes> {
    buffer: Buffer,
    head: AtomicUsize,
    live_allocations: AtomicI32,
}

impl BoundedBumpAllocator<Box<[u8]>> {
    pub fn with_capacity(cap: usize) -> Self {
        let buffer: Box<[u8]> = vec![0; cap].into_boxed_slice();
        BoundedBumpAllocator::with_buffer(buffer)
    }
}


impl<Buffer: AsMutBytes> BoundedBumpAllocator<Buffer> {
    /// Allocates a bump allocator with a buffer of `size` bytes.
    pub fn with_buffer(buffer: Buffer) -> Self {
        BoundedBumpAllocator { buffer, head: AtomicUsize::new(0), live_allocations: AtomicI32::new(0) }
    }

    /// Returns true if there is no live allocations from this allocator.
    pub fn can_reset(&self) -> bool {
        self.live_allocations.load(Ordering::SeqCst) == 0
    }

    /// Resets the bump allocator.
    ///
    /// Panics if there are live allocations from this allocator.
    pub fn reset(&self) {
        assert!(self.can_reset());
        self.head.store(0, Ordering::SeqCst);
    }

    unsafe fn allocate(&self, layout: Layout) -> Result<Allocation, AllocError> {
        let mut offset;
        loop {
            offset = self.head.load(Ordering::SeqCst);
            let mut size = layout.size();
            let rem = offset % layout.align();
            if rem != 0 {
                size += layout.align() - rem;
            }
            if offset + size > self.buffer.size() {
                return Err(AllocError::Allocator { layout });
            }
            if self.head.compare_exchange(offset, offset + size, Ordering::SeqCst, Ordering::Relaxed).is_ok() {
                break;
            }
        }

        let ptr = NonNull::new_unchecked(self.buffer.as_mut_bytes().add(offset));
        self.live_allocations.fetch_add(1, Ordering::SeqCst);

        Ok(Allocation { ptr, size: layout.size() })
    }

    unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: Layout) {
        self.live_allocations.fetch_sub(1, Ordering::SeqCst);
    }
}

impl<Buffer: AsMutBytes> Drop for BoundedBumpAllocator<Buffer> {
    /// Panics if there are live allocations from this allocator.
    fn drop(&mut self) {
        assert!(self.can_reset());
    }
}

impl<'l, Buffer: AsMutBytes> Allocator for &'l BoundedBumpAllocator<Buffer> {
    unsafe fn alloc(&self, layout: Layout) -> Result<Allocation, AllocError> {
        self.allocate(layout)
    }

    unsafe fn dealloc(&self, ptr: NonNull<u8>, layout: Layout) {
        self.deallocate(ptr, layout)
    }
}

impl<Buffer: AsMutBytes> Allocator for std::sync::Arc<BoundedBumpAllocator<Buffer>> {
    unsafe fn alloc(&self, layout: Layout) -> Result<Allocation, AllocError> {
        self.allocate(layout)
    }

    unsafe fn dealloc(&self, ptr: NonNull<u8>, layout: Layout) {
        self.deallocate(ptr, layout)
    }
}

impl<Buffer: AsMutBytes> Allocator for std::rc::Rc<BoundedBumpAllocator<Buffer>> {
    unsafe fn alloc(&self, layout: Layout) -> Result<Allocation, AllocError> {
        self.allocate(layout)
    }

    unsafe fn dealloc(&self, ptr: NonNull<u8>, layout: Layout) {
        self.deallocate(ptr, layout)
    }
}

pub struct SingleThreadedBumpAllocator<A: Allocator> {
    inner: UnsafeCell<StbaInner<A>>,
    allocator: A,
    buffer_layout: Layout,
}

struct StbaInner<A: Allocator> {
    current: StbaBuffer,
    others: UniqueVector<StbaBuffer, A>,
    live_allocations: i32,
    last_alloc: Option<NonNull<u8>>,
}

struct StbaBuffer {
    buffer: Allocation,
    head: usize,
}


impl<A: Allocator> SingleThreadedBumpAllocator<A> {
    pub fn with_allocator(allocator: A, size: usize) -> Result<Self, AllocError> {
        let layout = Layout::from_size_align(size, 64).unwrap(); // TODO: unwrap
        let buffer = unsafe { allocator.alloc(layout)? };

        Ok(SingleThreadedBumpAllocator {
            inner: UnsafeCell::new(StbaInner {
                current: StbaBuffer {
                    buffer,
                    head: 0,
                },
                others: UniqueVector::try_with_allocator(8, allocator.clone())?,
                live_allocations: 0,
                last_alloc: None,
            }),
            allocator,
            buffer_layout: layout,
        })
    }

    unsafe fn inner(&self) -> &mut StbaInner<A> {
        &mut *self.inner.get()
    }

    unsafe fn allocate_new_buffer(&self) -> Result<(), AllocError> {
        let buffer = unsafe { self.allocator.alloc(self.buffer_layout)? };

        let inner = self.inner();
        let current = std::mem::replace(&mut inner.current, StbaBuffer { buffer, head: 0 });
        inner.others.push(current);

        Ok(())
    }

    unsafe fn allocate(&self, layout: Layout) -> Result<Allocation, AllocError> {
        let status = self.allocate_in_current_buffer(layout);
        if status.is_ok() {
            return status;
        }

        self.allocate_new_buffer()?;

        self.allocate_in_current_buffer(layout)
    }

    unsafe fn allocate_in_current_buffer(&self, layout: Layout) -> Result<Allocation, AllocError> {
        let inner = self.inner();
        let current = &mut inner.current;
        let offset = current.head;
        let mut size = layout.size();
        let rem = offset % layout.align();
        if rem != 0 {
            size += layout.align() - rem;
        }
        if offset + size > current.buffer.size {
            return Err(AllocError::Allocator { layout });
        }

        current.head += size;

        let ptr = NonNull::new_unchecked(current.buffer.ptr.as_ptr().add(offset));
        inner.last_alloc = Some(ptr);
        inner.live_allocations += 1;

        Ok(Allocation { ptr, size: layout.size() })
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, _layout: Layout) {
        let inner = self.inner();

        if inner.last_alloc == Some(ptr) {
            let current = &mut inner.current;
            let diff = current.buffer.ptr.as_ptr().add(current.head) as usize - ptr.as_ptr() as usize;
            current.head -= diff;
        }

        inner.live_allocations -= 1;
        assert!(inner.live_allocations >= 0);
    }

    unsafe fn try_grow(&self, ptr: NonNull<u8>, new_layout: Layout, new_size: usize) -> Option<Allocation> {
        let inner = self.inner();
        let current = &mut inner.current;

        if inner.last_alloc == Some(ptr) && ptr.as_ptr() as usize % new_layout.align() == 0{
            if current.buffer.size - current.head <= new_size {
                let diff = current.buffer.ptr.as_ptr().add(current.head) as usize - ptr.as_ptr() as usize;
                current.head += diff;
                return Some(Allocation { ptr , size: new_size });
            }
        }

        None
    }

    unsafe fn reallocate(&self, ptr: NonNull<u8>, old_layout: Layout, new_size: usize) -> Result<Allocation, AllocError> {
        unsafe {
            let new_layout = Layout::from_size_align_unchecked(new_size, old_layout.align());

            // First see if we can simply grow the current allocation.
            if let Some(alloc) = self.try_grow(ptr, new_layout, new_size) {
                return Ok(alloc);
            }

            let new_alloc = self.allocate(new_layout);
            self.inner().live_allocations -= 1;

            if let Ok(new_alloc) = &new_alloc {
                let size = std::cmp::min(new_layout.size(), new_size);
                ptr::copy_nonoverlapping(ptr.as_ptr(), new_alloc.ptr.as_ptr(), size);
            }

            new_alloc
        }
    }
}

impl<A: Allocator> Drop for SingleThreadedBumpAllocator<A> {
    fn drop(&mut self) {
        unsafe {
            let inner = self.inner();
            self.allocator.dealloc(inner.current.buffer.ptr, self.buffer_layout);
            for other in &inner.others {
                self.allocator.dealloc(other.buffer.ptr, self.buffer_layout);
            }
        }
    }
}

impl<'l, A: Allocator> Allocator for &'l SingleThreadedBumpAllocator<A> {
    unsafe fn alloc(&self, layout: Layout) -> Result<Allocation, AllocError> {
        self.allocate(layout)
    }

    unsafe fn dealloc(&self, ptr: NonNull<u8>, layout: Layout) {
        self.deallocate(ptr, layout)
    }

    unsafe fn realloc(&self, ptr: NonNull<u8>, old_layout: Layout, new_size: usize) -> Result<Allocation, AllocError> {
        self.reallocate(ptr, old_layout, new_size)
    }
}

impl<A: Allocator> Allocator for std::sync::Arc<SingleThreadedBumpAllocator<A>> {
    unsafe fn alloc(&self, layout: Layout) -> Result<Allocation, AllocError> {
        self.allocate(layout)
    }

    unsafe fn dealloc(&self, ptr: NonNull<u8>, layout: Layout) {
        self.deallocate(ptr, layout)
    }
}

impl<A: Allocator> Allocator for std::rc::Rc<SingleThreadedBumpAllocator<A>> {
    unsafe fn alloc(&self, layout: Layout) -> Result<Allocation, AllocError> {
        self.allocate(layout)
    }

    unsafe fn dealloc(&self, ptr: NonNull<u8>, layout: Layout) {
        self.deallocate(ptr, layout)
    }
}
