//! Most of this should/will be in separate crates.

use std::{ptr::{self, NonNull}, sync::atomic::{AtomicPtr, AtomicI32, Ordering}, cell::UnsafeCell};
pub use std::alloc::Layout;

pub use allocator_api2::alloc::{Allocator, Global as GlobalAllocator, AllocError};

use crate::UniqueVector;

// It would have been convenient to use Box<[u8]> but it is just very tedious to work around the aliasing restriction.
// Every time the box is moved, the head/end pointers we created from it are invalidated (from miri's point of view,
// even though the heap allocation itself does not move).
pub struct HeapAllocation {
    ptr: NonNull<[u8]>,
    size: usize,
}

impl HeapAllocation {
    pub fn new(size: usize) -> Self {
        let ptr = allocator_api2::alloc::Global.allocate(Layout::from_size_align(size, 16).unwrap()).unwrap();
        HeapAllocation { ptr, size }
    }
}

impl Drop for HeapAllocation {
    fn drop(&mut self) {
        unsafe {
            allocator_api2::alloc::Global.deallocate(self.ptr.cast(), Layout::from_size_align(self.size, 16).unwrap());
        }
    }
}

pub trait AsMutBytes {
    unsafe fn as_mut_bytes(&self) -> *mut u8;
    fn size(&self) -> usize;
}

impl AsMutBytes for HeapAllocation {
    unsafe fn as_mut_bytes(&self) -> *mut u8 {
        self.ptr.cast::<u8>().as_ptr()
    }
    fn size(&self) -> usize {
        self.size
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
    head: AtomicPtr<u8>,
    end: *mut u8,
    live_allocations: AtomicI32,
}

impl BoundedBumpAllocator<HeapAllocation> {
    pub fn with_capacity(cap: usize) -> Self {
        let buffer = HeapAllocation::new(cap);
        BoundedBumpAllocator::with_buffer(buffer)
    }
}


impl<Buffer: AsMutBytes> BoundedBumpAllocator<Buffer> {
    /// Allocates a bump allocator with a buffer of `size` bytes.
    pub fn with_buffer(buffer: Buffer) -> Self {
        let start = unsafe { buffer.as_mut_bytes() };
        let end = unsafe { start.add(buffer.size()) };
        BoundedBumpAllocator {
            buffer,
            head: AtomicPtr::new(start),
            end,
            live_allocations: AtomicI32::new(0),
        }
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
        unsafe {
            self.head.store(self.buffer.as_mut_bytes(), Ordering::SeqCst);
        }
    }

    fn alloc(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        unsafe {
            self.live_allocations.fetch_add(1, Ordering::SeqCst);

            let mut ptr;
            loop {
                let head = self.head.load(Ordering::SeqCst);
                let rem = (head as usize) % layout.align();
                // Extra bytes at the beginning for alignemnt.
                let adjust = if rem == 0 { 0 } else { layout.align() - rem };
                let new_head = head.add(adjust + layout.size());

                if new_head > self.end {
                    return Err(AllocError);
                }
                ptr = head.add(adjust);
                if self.head.compare_exchange(head, new_head, Ordering::SeqCst, Ordering::Relaxed).is_ok() {
                    break;
                }
            }

            assert_eq!(ptr as usize % layout.align(), 0);
            let slice = core::ptr::slice_from_raw_parts_mut(ptr, layout.size());

            Ok(NonNull::new_unchecked(slice as *mut _))
        }
    }

    unsafe fn dealloc(&self, _ptr: NonNull<u8>, _layout: Layout) {
        self.live_allocations.fetch_sub(1, Ordering::SeqCst);
    }
}

impl<Buffer: AsMutBytes> Drop for BoundedBumpAllocator<Buffer> {
    /// Panics if there are live allocations from this allocator.
    fn drop(&mut self) {
        assert!(self.can_reset());
    }
}

unsafe impl<'l, Buffer: AsMutBytes> Allocator for &'l BoundedBumpAllocator<Buffer> {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.alloc(layout)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        self.dealloc(ptr, layout)
    }
}

/*
impl<Buffer: AsMutBytes> Allocator for std::sync::Arc<BoundedBumpAllocator<Buffer>> {
    unsafe fn allocate(&self, layout: Layout) -> Result<Allocation, AllocError> {
        self.alloc(layout)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        self.dealloc(ptr, layout)
    }
}

impl<Buffer: AsMutBytes> Allocator for std::rc::Rc<BoundedBumpAllocator<Buffer>> {
    unsafe fn allocate(&self, layout: Layout) -> Result<Allocation, AllocError> {
        self.alloc(layout)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        self.dealloc(ptr, layout)
    }
}
*/

pub struct SingleThreadedBumpAllocator<A: Allocator + Clone> {
    inner: UnsafeCell<StbaInner<A>>,
    allocator: A,
    buffer_layout: Layout,
}

struct StbaInner<A: Allocator + Clone> {
    current: StbaBuffer,
    others: UniqueVector<StbaBuffer, A>,
    live_allocations: i32,
    last_alloc: Option<NonNull<u8>>,
}

struct StbaBuffer {
    head: *mut u8,
    end: *mut u8,
    buffer: NonNull<[u8]>,
}


impl<A: Allocator + Clone> SingleThreadedBumpAllocator<A> {
    pub fn with_allocator(allocator: A, size: usize) -> Result<Self, AllocError> {
        let layout = Layout::from_size_align(size, 64).unwrap(); // TODO: unwrap
        let buffer = allocator.allocate(layout)?;

        let head = buffer.cast::<u8>().as_ptr();
        let end = unsafe { head.add(buffer.as_ref().len()) };
        Ok(SingleThreadedBumpAllocator {
            inner: UnsafeCell::new(StbaInner {
                current: StbaBuffer {
                    head,
                    end,
                    buffer,
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
        let buffer = self.allocator.allocate(self.buffer_layout)?;
        let head = buffer.cast::<u8>().as_ptr();
        let end = unsafe { head.add(buffer.as_ref().len()) };

        let inner = self.inner();
        let current = std::mem::replace(&mut inner.current, StbaBuffer { buffer, head, end });
        inner.others.push(current);

        Ok(())
    }

    fn alloc(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        unsafe {
            let status = self.allocate_in_current_buffer(layout);
            if status.is_ok() {
                return status;
            }
    
            self.allocate_new_buffer()?;
    
            self.allocate_in_current_buffer(layout)    
        }
    }

    unsafe fn allocate_in_current_buffer(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let inner = self.inner();
        let current = &mut inner.current;

        let head = current.head;
        let rem = (head as usize) % layout.align();
        // Extra bytes at the beginning for alignemnt.
        let adjust = if rem == 0 { 0 } else { layout.align() - rem };
        let new_head = head.add(adjust + layout.size());

        if new_head > current.end {
            return Err(AllocError);
        }

        let ptr = NonNull::new_unchecked(head.add(adjust));
        current.head = new_head;

        inner.last_alloc = Some(ptr);
        inner.live_allocations += 1;

        assert_eq!(ptr.as_ptr() as usize % layout.align(), 0);

        Ok(raw_slice(ptr, layout.size()))
    }

    unsafe fn dealloc(&self, ptr: NonNull<u8>, _layout: Layout) {
        let inner = self.inner();

        if inner.last_alloc == Some(ptr) {
            let current = &mut inner.current;
            let diff = current.head as isize - ptr.as_ptr() as isize;
            current.head = current.head.offset(-diff);
        }

        inner.live_allocations -= 1;
        assert!(inner.live_allocations >= 0);
    }

    unsafe fn try_grow(&self, ptr: NonNull<u8>, new_layout: Layout, new_size: usize) -> Option<NonNull<[u8]>> {
        let inner = self.inner();
        let current = &mut inner.current;

        if inner.last_alloc == Some(ptr) && ptr.as_ptr() as usize % new_layout.align() == 0 {
            if (current.end as isize) - (current.head as isize) <= new_size as isize {
                let diff = current.head as isize - ptr.as_ptr() as isize;
                current.head = current.head.offset(-diff);
                return Some(raw_slice(ptr , new_size));
            }
        }

        None
    }

    unsafe fn grow_impl(&self, ptr: NonNull<u8>, old_layout: Layout, new_layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        unsafe {
            let new_layout = Layout::from_size_align_unchecked(new_layout.size(), old_layout.align());

            // First see if we can simply grow the current allocation.
            if let Some(alloc) = self.try_grow(ptr, new_layout, new_layout.size()) {
                return Ok(alloc);
            }

            let new_alloc = self.allocate(new_layout);
            self.inner().live_allocations -= 1;

            if let Ok(new_alloc) = &new_alloc {
                let size = std::cmp::min(new_layout.size(), new_layout.size());
                ptr::copy_nonoverlapping(ptr.as_ptr(), new_alloc.cast().as_ptr(), size);
            }

            new_alloc
        }
    }
}

unsafe fn raw_slice(ptr: NonNull<u8>, len: usize) -> NonNull<[u8]> {
    NonNull::new_unchecked(core::slice::from_raw_parts_mut(ptr.as_ptr(), len))
}

impl<A: Allocator + Clone> Drop for SingleThreadedBumpAllocator<A> {
    fn drop(&mut self) {
        unsafe {
            let inner = self.inner();
            self.allocator.deallocate(inner.current.buffer.cast(), self.buffer_layout);
            for other in &inner.others {
                self.allocator.deallocate(other.buffer.cast(), self.buffer_layout);
            }
        }
    }
}

unsafe impl<'l, A: Allocator + Clone> Allocator for &'l SingleThreadedBumpAllocator<A> {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.alloc(layout)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        self.dealloc(ptr, layout)
    }

    unsafe fn grow(&self, ptr: NonNull<u8>, old_layout: Layout, new_layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.grow_impl(ptr, old_layout, new_layout)
    }
}

pub struct RcAllocator<A>(std::rc::Rc<A>);
pub struct ArcAllocator<A>(std::sync::Arc<A>);

pub struct RefAllocator<'a, A>(&'a A);

unsafe impl<A: Allocator> Allocator for RcAllocator<A> {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.0.allocate(layout)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        self.0.deallocate(ptr, layout)
    }
}

unsafe impl<A: Allocator> Allocator for ArcAllocator<A> {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.0.allocate(layout)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        self.0.deallocate(ptr, layout)
    }
}

unsafe impl<'a, A: Allocator> Allocator for RefAllocator<'a, A> {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.0.allocate(layout)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        self.0.deallocate(ptr, layout)
    }
}

impl<A> Clone for RcAllocator<A> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<A> Clone for ArcAllocator<A> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<'a, A> Clone for RefAllocator<'a, A> {
    fn clone(&self) -> Self {
        Self(self.0)
    }
}
