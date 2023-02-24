use std::ptr::{self, NonNull};
pub use std::alloc::Layout;

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
