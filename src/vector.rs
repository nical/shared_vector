use core::fmt::Debug;
use core::ops::{Deref, DerefMut, Index, IndexMut};
use core::ptr::NonNull;
use core::{mem, ptr};
use core::ops::RangeBounds;

use crate::alloc::{AllocError, Allocator, Global};
use crate::drain::Drain;
use crate::raw::{
    self, buffer_layout, AtomicRefCount, BufferSize, Header, HeaderBuffer, RefCount, VecHeader, move_data,
};
use crate::shared::{AtomicSharedVector, SharedVector};
use crate::splice::Splice;
use crate::{grow_amortized, DefaultRefCount};

/// A heap allocated, mutable contiguous buffer containing elements of type `T`, with manual deallocation.
///
/// <svg width="280" height="120" viewBox="0 0 74.08 31.75" xmlns:xlink="http://www.w3.org/1999/xlink" xmlns="http://www.w3.org/2000/svg"><defs><linearGradient id="a"><stop offset="0" stop-color="#491c9c"/><stop offset="1" stop-color="#d54b27"/></linearGradient><linearGradient xlink:href="#a" id="b" gradientUnits="userSpaceOnUse" x1="6.27" y1="34.86" x2="87.72" y2="13.24" gradientTransform="translate(-2.64 -18.48)"/></defs><rect width="10.57" height="10.66" x="7.94" y="18.45" ry="1.37" fill="#3dbdaa"/><circle cx="12.7" cy="18.48" r=".79" fill="#666"/><path d="M12.7 18.48c0-3.93 7.14-1.28 7.14-5.25" fill="none" stroke="#999" stroke-width=".86" stroke-linecap="round"/><rect width="68.79" height="10.58" x="2.65" y="2.68" ry="1.37" fill="url(#b)"/><rect width="15.35" height="9.51" x="3.18" y="3.21" ry=".9" fill="#78a2d4"/><rect width="9.26" height="9.51" x="19.85" y="3.2" ry=".9" fill="#eaa577"/><rect width="9.26" height="9.51" x="29.64" y="3.22" ry=".9" fill="#eaa577"/><rect width="9.26" height="9.51" x="39.43" y="3.22" ry=".9" fill="#eaa577"/><rect width="9.26" height="9.51" x="49.22" y="3.21" ry=".9" fill="#eaa577"/><circle cx="62.84" cy="7.97" r=".66" fill="#eaa577"/><circle cx="64.7" cy="7.97" r=".66" fill="#eaa577"/><circle cx="66.55" cy="7.97" r=".66" fill="#eaa577"/></svg>
///
/// See also `Vector<T, A>`.
///
/// This container is similar to this crate's `Vector` data structure with two key difference:
/// - It does store an allocator field. Instead, all methods that require interacting with an allocator are
///   marked unsafe and take the allocator as parameter.
/// - `RawVector`'s `Drop` implementation does not automatically deallocate the memory. Instead the memory
///   must be manually deallocated via the `deallocate` method. Dropping a raw vector without deallocating it
///   silently leaks the memory.
///
/// `Vector<T, A>` is implemented as a thin wrapper around this type.
///
/// # Use cases
///
/// In most cases, `Vector<T, A>` is more appropriate. However in some situations it can be beneficial to not
/// store the allocator in the container. In complex data structures that contain many vectors, for example,
/// it may be preferable to store the allocator once at the root of the data structure than multiple times
/// in each of the internally managed vectors.
pub struct RawVector<T> {
    pub(crate) data: NonNull<T>,
    pub(crate) header: VecHeader,
}

impl<T> RawVector<T> {
    /// Creates an empty, unallocated raw vector.
    pub fn new() -> Self {
        RawVector { data: NonNull::dangling(), header: VecHeader { len: 0, cap: 0 } }
    }

    /// Creates an empty pre-allocated vector with a given storage capacity.
    ///
    /// Does not allocate memory if `cap` is zero.
    pub fn try_with_capacity<A: Allocator>(allocator: &A, cap: usize) -> Result<RawVector<T>, AllocError> {
        if cap == 0 {
            return Ok(RawVector::new());
        }

        unsafe {
            let (base_ptr, cap) = raw::allocate_header_buffer::<T, A>(cap, allocator)?;
            let data = NonNull::new_unchecked(raw::data_ptr::<raw::Header<DefaultRefCount, A>, T>(
                base_ptr.cast(),
            ));
            Ok(RawVector {
                data,
                header: VecHeader { cap: cap as BufferSize, len: 0 },
            })
        }
    }

    pub fn try_from_slice<A: Allocator>(allocator: &A, data: &[T]) -> Result<Self, AllocError>
    where
        T: Clone,
    {
        let mut v = Self::try_with_capacity(allocator, data.len())?;
        unsafe {
            v.extend_from_slice(allocator, data);
        }

        Ok(v)
    }

    /// Creates a raw vector with `n` clones of `elem`.
    pub fn try_from_elem<A: Allocator>(allocator: &A, elem: T, n: usize) -> Result<Self, AllocError>
    where
        T: Clone,
    {
        if n == 0 {
            return Ok(Self::new());
        }

        let mut v = Self::try_with_capacity(allocator, n)?;
        unsafe {
            for _ in 0..(n - 1) {
                v.push(allocator, elem.clone())
            }
    
            v.push(allocator, elem);    
        }

        Ok(v)
    }

    /// Clears and deallocates this raw vector, leaving it in its unallocated state.
    ///
    /// It is safe (no-op) to call `deallocate` on a vector that is already in its unallocated state.
    ///
    /// # Safety
    ///
    /// The provided allocator must be the one this raw vector was created with.
    pub unsafe fn deallocate<A: Allocator>(&mut self, allocator: &A) {
        if self.header.cap == 0 {
            return;
        }

        self.clear();

        self.deallocate_buffer(allocator);

        self.data = NonNull::dangling();
        self.header.cap = 0;
        self.header.len = 0;
    }

    #[inline]
    /// Returns `true` if the vector contains no elements.
    pub fn is_empty(&self) -> bool {
        self.header.len == 0
    }

    #[inline]
    /// Returns the number of elements in the vector, also referred to as its ‘length’.
    pub fn len(&self) -> usize {
        self.header.len as usize
    }

    #[inline]
    /// Returns the total number of elements the vector can hold without reallocating.
    pub fn capacity(&self) -> usize {
        self.header.cap as usize
    }

    /// Returns number of elements that can be added without reallocating.
    #[inline]
    pub fn remaining_capacity(&self) -> usize {
        (self.header.cap - self.header.len) as usize
    }

    #[inline]
    fn data_ptr(&self) -> *mut T {
        self.data.as_ptr()
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        unsafe { core::slice::from_raw_parts(self.data_ptr(), self.len()) }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { core::slice::from_raw_parts_mut(self.data_ptr(), self.len()) }
    }

    /// Clears the vector, removing all values.
    pub fn clear(&mut self) {
        unsafe {
            raw::clear(self.data_ptr(), &mut self.header)
        }
    }

    unsafe fn base_ptr<A: Allocator>(&self, _allocator: &A) -> NonNull<u8> {
        debug_assert!(self.header.cap > 0);
        raw::header_from_data_ptr::<Header<DefaultRefCount, A>, T>(self.data).cast()
    }

    /// Appends an element to the back of a collection.
    ///
    /// # Safety
    ///
    /// The provided allocator must be the one this raw vector was created with.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity exceeds `u32::MAX` bytes.
    #[inline]
    pub unsafe fn push<A: Allocator>(&mut self, allocator: &A, val: T) {
        if self.header.len == self.header.cap {
            self.try_realloc_additional(allocator, 1).unwrap();
        }

        raw::push_assuming_capacity(self.data_ptr(), &mut self.header, val);
    }

    /// Appends an element if there is sufficient spare capacity, otherwise an error is returned
    /// with the element.
    ///
    /// Unlike push this method will not reallocate when there’s insufficient capacity.
    /// The caller should use reserve or try_reserve to ensure that there is enough capacity.
    #[inline]
    pub fn push_within_capacity(&mut self, val: T) -> Result<(), T> {
        if self.header.len == self.header.cap {
            return Err(val);
        }

        unsafe {
            let dst = self.data_ptr().add(self.header.len as usize);
            self.header.len += 1;
            ptr::write(dst, val);
        }

        Ok(())
    }

    /// Removes the last element from the vector and returns it, or `None` if it is empty.
    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        unsafe {
            raw::pop(self.data_ptr(), &mut self.header)
        }
    }

    /// Removes and returns the element at position `index` within the vector,
    /// shifting all elements after it to the left.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    ///
    pub fn remove(&mut self, index: usize) -> T {
        #[cold]
        #[inline(never)]
        #[track_caller]
        fn assert_failed(index: usize, len: usize) -> ! {
            panic!("removal index (is {index}) should be < len (is {len})");
        }

        let len = self.len();
        if index >= len {
            assert_failed(index, len);
        }
        unsafe {
            // infallible
            let ret;
            {
                // the place we are taking from.
                let ptr = self.as_mut_ptr().add(index);
                // copy it out, unsafely having a copy of the value on
                // the stack and in the vector at the same time.
                ret = ptr::read(ptr);

                // Shift everything down to fill in that spot.
                ptr::copy(ptr.add(1), ptr, len - index - 1);
            }
            self.header.len = len as u32 - 1;
            ret
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
        let len = self.len();
        assert!(idx < len);

        unsafe {
            let ptr = self.data_ptr().add(idx);
            let item = ptr::read(ptr);

            let last_idx = len - 1;
            if idx != last_idx {
                let last_ptr = self.data_ptr().add(last_idx);
                ptr::write(ptr, ptr::read(last_ptr));
            }

            self.header.len -= 1;

            item
        }
    }

    /// Inserts an element at position `index` within the vector, shifting all
    /// elements after it to the right.
    ///
    /// # Panics
    ///
    /// Panics if `index > len`.
    pub unsafe fn insert<A: Allocator>(&mut self, allocator: &A, index: usize, element: T) {
        #[cold]
        #[inline(never)]
        fn assert_failed(index: usize, len: usize) -> ! {
            panic!("insertion index (is {index}) should be <= len (is {len})");
        }

        unsafe {
            // space for the new element
            if self.header.len == self.header.cap {
                self.try_reserve(allocator, 1).unwrap();
            }

            let len = self.len();

            // infallible
            // The spot to put the new value
            {
                let p = self.as_mut_ptr().add(index);
                if index < len {
                    // Shift everything over to make space. (Duplicating the
                    // `index`th element into two consecutive places.)
                    ptr::copy(p, p.add(1), len - index);
                } else if index == len {
                    // No elements need shifting.
                } else {
                    assert_failed(index, len);
                }
                // Write it in, overwriting the first copy of the `index`th
                // element.
                ptr::write(p, element);
            }
            self.header.len += 1;
        }
    }

    /// Clones and appends the contents of the slice to the back of a collection.
    ///
    /// # Safety
    ///
    /// The provided allocator must be the one this raw vector was created with.
    pub unsafe fn extend_from_slice<A: Allocator>(&mut self, allocator: &A, slice: &[T])
    where
        T: Clone,
    {
        self.try_reserve(allocator, slice.len()).unwrap();
        unsafe {
            raw::extend_from_slice_assuming_capacity(self.data_ptr(), &mut self.header, slice);
        }
    }

    /// Moves all the elements of `other` into `self`, leaving `other` empty.
    ///
    /// # Safety
    ///
    /// The provided allocator must be the one this raw vector was created with.
    pub unsafe fn append<A: Allocator>(&mut self, allocator: &A, other: &mut Self)
    where
        T: Clone,
    {
        if other.is_empty() {
            return;
        }

        self.try_reserve(allocator, other.len()).unwrap();

        unsafe {
            move_data(other.data_ptr(), &mut other.header, self.data_ptr(), &mut self.header);
        }
    }

    /// Appends the contents of an iterator to the back of a collection.
    ///
    /// # Safety
    ///
    /// The provided allocator must be the one this raw vector was created with.
    pub unsafe fn extend<A: Allocator>(&mut self, allocator: &A, data: impl IntoIterator<Item = T>) {
        let mut iter = data.into_iter();
        let (min, max) = iter.size_hint();
        self.try_reserve(allocator, max.unwrap_or(min)).unwrap();
        unsafe {
            self.extend_within_capacity(&mut iter);

            for item in iter {
                self.push(allocator, item);
            }
        }
    }

    unsafe fn extend_within_capacity(&mut self, iter: &mut impl Iterator<Item = T>) {
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
            self.header.len += count;
        }
    }

    /// Allocate a clone of this buffer.
    ///
    /// The provided allocator does not need to be the one this raw vector was created with.
    /// The returned raw vector is considered to be created with the provided allocator.
    pub fn clone_buffer<A: Allocator>(&self, allocator: &A) -> Self
    where
        T: Clone,
    {
        self.clone_buffer_with_capacity(allocator, self.len())
    }

    /// Allocate a clone of this buffer with a different capacity
    ///
    /// The capacity must be at least as large as the buffer's length.
    pub fn clone_buffer_with_capacity<A: Allocator>(&self, allocator: &A, cap: usize) -> Self
    where
        T: Clone,
    {
        let mut clone =
            Self::try_with_capacity(allocator, cap.max(self.len())).unwrap();

        unsafe {
            raw::extend_from_slice_assuming_capacity(clone.data_ptr(), &mut clone.header, self.as_slice());
        }

        clone
    }

    // Note: Marking this #[inline(never)] is a pretty large regression in the push benchmark.
    #[cold]
    unsafe fn try_realloc_additional<A: Allocator>(&mut self, allocator: &A, additional: usize) -> Result<(), AllocError> {
        let new_cap = grow_amortized(self.len(), additional);
        if new_cap < self.len() {
            return Err(AllocError);
        }

        self.try_realloc_with_capacity(allocator, new_cap)
    }

    #[cold]
    unsafe fn try_realloc_with_capacity<A: Allocator>(&mut self, allocator: &A, new_cap: usize) -> Result<(), AllocError> {
        type R = DefaultRefCount;

        unsafe {
            if new_cap == 0 {
                self.deallocate_buffer(allocator);
            }

            let new_layout = buffer_layout::<Header<R, A>, T>(new_cap).unwrap();

            let new_alloc = if self.header.cap == 0 {
                allocator.allocate(new_layout)?
            } else {
                let old_cap = self.capacity();
                let old_ptr = self.base_ptr(allocator);
                let old_layout = buffer_layout::<Header<R, A>, T>(old_cap).unwrap();
                let new_layout = buffer_layout::<Header<R, A>, T>(new_cap).unwrap();

                if new_layout.size() >= old_layout.size() {
                    allocator.grow(old_ptr, old_layout, new_layout)
                } else {
                    allocator.shrink(old_ptr, old_layout, new_layout)
                }?
            };

            let new_data_ptr = crate::raw::data_ptr::<Header<R, A>, T>(new_alloc.cast());
            self.data = NonNull::new_unchecked(new_data_ptr);
            self.header.cap = new_cap as u32;
        }

        Ok(())
    }

    // Deallocates the memory, does not drop the vector's content.
    unsafe fn deallocate_buffer<A: Allocator>(&mut self, allocator: &A) {
        let layout = buffer_layout::<Header<DefaultRefCount, A>, T>(self.capacity()).unwrap();
        let ptr = self.base_ptr(allocator);

        allocator.deallocate(ptr, layout);

        self.header.cap = 0;
        self.header.len = 0;
        self.data = NonNull::dangling();
    }

    /// Tries to reserve at least enough space for `additional` extra items.
    ///
    /// # Safety
    ///
    /// The provided allocator must be the one this raw vector was created with.
    #[inline]
    pub unsafe fn try_reserve<A: Allocator>(&mut self, allocator: &A, additional: usize) -> Result<(), AllocError> {
        if self.remaining_capacity() < additional {
            self.try_realloc_additional(allocator, additional)?;
        }

        Ok(())
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
    ///
    /// # Safety
    ///
    /// The provided allocator must be the one this raw vector was created with.
    pub unsafe fn try_reserve_exact<A: Allocator>(&mut self, allocator: &A, additional: usize) -> Result<(), AllocError> {
        if self.remaining_capacity() >= additional {
            return Ok(());
        }

        self.try_realloc_with_capacity(allocator, self.len() + additional)
    }

    /// Shrinks the capacity of the vector with a lower bound.
    ///
    /// The capacity will remain at least as large as both the length and the supplied value.
    /// If the current capacity is less than the lower limit, this is a no-op.
    ///
    /// # Safety
    ///
    /// The provided allocator must be the one this raw vector was created with.
    pub unsafe fn shrink_to<A: Allocator>(&mut self, allocator: &A, min_capacity: usize)
    {
        let min_capacity = min_capacity.max(self.len());
        if self.capacity() <= min_capacity {
            return;
        }

        self.try_realloc_with_capacity(allocator, min_capacity).unwrap();
    }

    /// Shrinks the capacity of the vector as much as possible.
    ///
    /// # Safety
    ///
    /// The provided allocator must be the one this raw vector was created with.
    pub unsafe fn shrink_to_fit<A: Allocator>(&mut self, allocator: &A)
    {
        self.shrink_to(allocator, self.len())
    }

    /// Removes the specified range from the vector in bulk, returning all
    /// removed elements as an iterator. If the iterator is dropped before
    /// being fully consumed, it drops the remaining removed elements.
    ///
    /// The returned iterator keeps a mutable borrow on the vector to optimize
    /// its implementation.
    ///
    /// # Panics
    ///
    /// Panics if the starting point is greater than the end point or if
    /// the end point is greater than the length of the vector.
    ///
    /// # Leaking
    ///
    /// If the returned iterator goes out of scope without being dropped (due to
    /// [`mem::forget`], for example), the vector may have lost and leaked
    /// elements arbitrarily, including elements outside the range.
    ///
    pub fn drain<R>(&mut self, range: R) -> Drain<'_, T>
    where
        R: RangeBounds<usize>,
    {
        // Memory safety
        //
        // When the Drain is first created, it shortens the length of
        // the source vector to make sure no uninitialized or moved-from elements
        // are accessible at all if the Drain's destructor never gets to run.
        //
        // Drain will ptr::read out the values to remove.
        // When finished, remaining tail of the vec is copied back to cover
        // the hole, and the vector length is restored to the new length.
        //
        use core::ops::Bound::*;
        let len = self.len();
        let end = match range.end_bound() {
            Included(n) => *n + 1,
            Excluded(n) => *n,
            Unbounded => len
        };
        let start = match range.start_bound() {
            Included(n) => *n,
            Excluded(n) => *n+1,
            Unbounded => 0
        };
        assert!(end <= len);
        assert!(start <= end);

        unsafe {
            // Set self.vec length's to start, to be safe in case Drain is leaked
            self.header.len = start as u32;
            let range_slice = core::slice::from_raw_parts(self.as_ptr().add(start), end - start);
            Drain {
                tail_start: end,
                tail_len: len - end,
                iter: range_slice.iter(),
                vec: NonNull::from(self),
            }
        }        
    }

    /// Creates a splicing iterator that replaces the specified range in the vector
    /// with the given `replace_with` iterator and yields the removed items.
    /// `replace_with` does not need to be the same length as `range`.
    ///
    /// `range` is removed even if the iterator is not consumed until the end.
    ///
    /// It is unspecified how many elements are removed from the vector
    /// if the `Splice` value is leaked.
    ///
    /// The input iterator `replace_with` is only consumed when the `Splice` value is dropped.
    ///
    /// This is optimal if:
    ///
    /// * The tail (elements in the vector after `range`) is empty,
    /// * or `replace_with` yields fewer or equal elements than `range`’s length
    /// * or the lower bound of its `size_hint()` is exact.
    ///
    /// Otherwise, a temporary vector is allocated and the tail is moved twice.
    ///
    /// # Panics
    ///
    /// Panics if the starting point is greater than the end point or if
    /// the end point is greater than the length of the vector.
    ///
    pub unsafe fn splice<'l, A, R, I>(
        &'l mut self,
        allocator: &'l A,
        range: R,
        replace_with: I
    ) -> Splice<'l, <I as IntoIterator>::IntoIter, A>
    where
        A: Allocator,
        R: RangeBounds<usize>,
        I: IntoIterator<Item = T>,
    {
        Splice {
            drain: self.drain(range),
            replace_with: replace_with.into_iter(),
            allocator,
        }
    }

    /// Retains only the elements specified by the predicate.
    ///
    /// In other words, remove all elements `e` for which `f(&e)` returns `false`.
    /// This method operates in place, visiting each element exactly once in the
    /// original order, and preserves the order of the retained elements.
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&T) -> bool,
    {
        self.retain_mut(|elem| f(elem));
    }

    /// Retains only the elements specified by the predicate, passing a mutable reference to it.
    ///
    /// In other words, remove all elements `e` such that `f(&mut e)` returns `false`.
    /// This method operates in place, visiting each element exactly once in the
    /// original order, and preserves the order of the retained elements.
    pub fn retain_mut<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut T) -> bool,
    {
        let original_len = self.len();
        // Avoid double drop if the drop guard is not executed,
        // since we may make some holes during the process.
        self.header.len = 0;

        // Vec: [Kept, Kept, Hole, Hole, Hole, Hole, Unchecked, Unchecked]
        //      |<-              processed len   ->| ^- next to check
        //                  |<-  deleted cnt     ->|
        //      |<-              original_len                          ->|
        // Kept: Elements which predicate returns true on.
        // Hole: Moved or dropped element slot.
        // Unchecked: Unchecked valid elements.
        //
        // This drop guard will be invoked when predicate or `drop` of element panicked.
        // It shifts unchecked elements to cover holes and `set_len` to the correct length.
        // In cases when predicate and `drop` never panick, it will be optimized out.
        struct BackshiftOnDrop<'a, T> {
            v: &'a mut RawVector<T>,
            processed_len: usize,
            deleted_cnt: usize,
            original_len: usize,
        }

        impl<T> Drop for BackshiftOnDrop<'_, T> {
            fn drop(&mut self) {
                if self.deleted_cnt > 0 {
                    // SAFETY: Trailing unchecked items must be valid since we never touch them.
                    unsafe {
                        ptr::copy(
                            self.v.as_ptr().add(self.processed_len),
                            self.v.as_mut_ptr().add(self.processed_len - self.deleted_cnt),
                            self.original_len - self.processed_len,
                        );
                    }
                }
                // SAFETY: After filling holes, all items are in contiguous memory.
                self.v.header.len = (self.original_len - self.deleted_cnt) as u32;
            }
        }

        let mut g = BackshiftOnDrop { v: self, processed_len: 0, deleted_cnt: 0, original_len };

        fn process_loop<F, T, const DELETED: bool>(
            original_len: usize,
            f: &mut F,
            g: &mut BackshiftOnDrop<'_, T>,
        ) where
            F: FnMut(&mut T) -> bool,
        {
            while g.processed_len != original_len {
                // SAFETY: Unchecked element must be valid.
                let cur = unsafe { &mut *g.v.as_mut_ptr().add(g.processed_len) };
                if !f(cur) {
                    // Advance early to avoid double drop if `drop_in_place` panicked.
                    g.processed_len += 1;
                    g.deleted_cnt += 1;
                    // SAFETY: We never touch this element again after dropped.
                    unsafe { ptr::drop_in_place(cur) };
                    // We already advanced the counter.
                    if DELETED {
                        continue;
                    } else {
                        break;
                    }
                }
                if DELETED {
                    // SAFETY: `deleted_cnt` > 0, so the hole slot must not overlap with current element.
                    // We use copy for move, and never touch this element again.
                    unsafe {
                        let hole_slot = g.v.as_mut_ptr().add(g.processed_len - g.deleted_cnt);
                        ptr::copy_nonoverlapping(cur, hole_slot, 1);
                    }
                }
                g.processed_len += 1;
            }
        }

        // Stage 1: Nothing was deleted.
        process_loop::<F, T, false>(original_len, &mut f, &mut g);

        // Stage 2: Some elements were deleted.
        process_loop::<F, T, true>(original_len, &mut f, &mut g);

        // All item are processed. This can be optimized to `set_len` by LLVM.
        drop(g);
    }

    /// Transfers ownership of this raw vector's contents to the one that is returned, and leaves
    /// this one empty and unallocated.
    pub fn take(&mut self) -> Self {
        mem::replace(self, RawVector::new())
    }
}

impl<T: PartialEq<T>> PartialEq<RawVector<T>> for RawVector<T> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: PartialEq<T>> PartialEq<&[T]> for RawVector<T> {
    fn eq(&self, other: &&[T]) -> bool {
        self.as_slice() == *other
    }
}

impl<T> AsRef<[T]> for RawVector<T> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> AsMut<[T]> for RawVector<T> {
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T> Default for RawVector<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, T> IntoIterator for &'a RawVector<T> {
    type Item = &'a T;
    type IntoIter = core::slice::Iter<'a, T>;
    fn into_iter(self) -> core::slice::Iter<'a, T> {
        self.as_slice().iter()
    }
}

impl<'a, T> IntoIterator for &'a mut RawVector<T> {
    type Item = &'a mut T;
    type IntoIter = core::slice::IterMut<'a, T>;
    fn into_iter(self) -> core::slice::IterMut<'a, T> {
        self.as_mut_slice().iter_mut()
    }
}

impl<T, I> Index<I> for RawVector<T>
where
    I: core::slice::SliceIndex<[T]>,
{
    type Output = <I as core::slice::SliceIndex<[T]>>::Output;
    fn index(&self, index: I) -> &Self::Output {
        self.as_slice().index(index)
    }
}

impl<T, I> IndexMut<I> for RawVector<T>
where
    I: core::slice::SliceIndex<[T]>,
{
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        self.as_mut_slice().index_mut(index)
    }
}

impl<T> Deref for RawVector<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> DerefMut for RawVector<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T: Debug> Debug for RawVector<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> Result<(), core::fmt::Error> {
        self.as_slice().fmt(f)
    }
}

impl<T: core::hash::Hash> core::hash::Hash for RawVector<T> {
    fn hash<H>(&self, state: &mut H) where H: core::hash::Hasher {
        self.as_slice().hash(state)
    }
}


/// A heap allocated, mutable contiguous buffer containing elements of type `T`.
///
/// <svg width="280" height="120" viewBox="0 0 74.08 31.75" xmlns:xlink="http://www.w3.org/1999/xlink" xmlns="http://www.w3.org/2000/svg"><defs><linearGradient id="a"><stop offset="0" stop-color="#491c9c"/><stop offset="1" stop-color="#d54b27"/></linearGradient><linearGradient xlink:href="#a" id="b" gradientUnits="userSpaceOnUse" x1="6.27" y1="34.86" x2="87.72" y2="13.24" gradientTransform="translate(-2.64 -18.48)"/></defs><rect width="10.57" height="10.66" x="7.94" y="18.45" ry="1.37" fill="#3dbdaa"/><circle cx="12.7" cy="18.48" r=".79" fill="#666"/><path d="M12.7 18.48c0-3.93 7.14-1.28 7.14-5.25" fill="none" stroke="#999" stroke-width=".86" stroke-linecap="round"/><rect width="68.79" height="10.58" x="2.65" y="2.68" ry="1.37" fill="url(#b)"/><rect width="15.35" height="9.51" x="3.18" y="3.21" ry=".9" fill="#78a2d4"/><rect width="9.26" height="9.51" x="19.85" y="3.2" ry=".9" fill="#eaa577"/><rect width="9.26" height="9.51" x="29.64" y="3.22" ry=".9" fill="#eaa577"/><rect width="9.26" height="9.51" x="39.43" y="3.22" ry=".9" fill="#eaa577"/><rect width="9.26" height="9.51" x="49.22" y="3.21" ry=".9" fill="#eaa577"/><circle cx="62.84" cy="7.97" r=".66" fill="#eaa577"/><circle cx="64.7" cy="7.97" r=".66" fill="#eaa577"/><circle cx="66.55" cy="7.97" r=".66" fill="#eaa577"/></svg>
///
/// Similar in principle to a `Vec<T>`.
/// It can be converted for free into a reference counted `SharedVector<T>` or `AtomicSharedVector<T>`.
///
/// Unique and shared vectors expose similar functionality. `Vector` takes advantage of
/// the guaranteed uniqueness at the type level to provide overall faster operations than its
/// shared counterparts, while its memory layout makes it very cheap to convert to a shared vector
/// (involving no allocation or copy).
///
/// # Internal representation
///
/// `Vector` stores its length and capacity inline and points to the first element of the
/// allocated buffer. Room for a header is left uninitialized before the elements so that the
/// vector can be converted into a `SharedVector` or `AtomicSharedVector` without reallocating
/// the storage.
///
/// Internally, `Vector` is built on top of `RawVector`.
pub struct Vector<T, A: Allocator = Global> {
    pub(crate) raw: RawVector<T>,
    pub(crate) allocator: A,
}

impl<T> Vector<T, Global> {
    /// Creates an empty vector.
    ///
    /// This does not allocate memory.
    pub fn new() -> Vector<T, Global> {
        Vector {
            raw: RawVector::new(),
            allocator: Global,
        }
    }



    /// Creates an empty pre-allocated vector with a given storage capacity.
    ///
    /// Does not allocate memory if `cap` is zero.
    pub fn with_capacity(cap: usize) -> Vector<T, Global> {
        Self::try_with_capacity(cap).unwrap()
    }

    /// Creates an empty pre-allocated vector with a given storage capacity.
    ///
    /// Does not allocate memory if `cap` is zero.
    pub fn try_with_capacity(cap: usize) -> Result<Vector<T, Global>, AllocError> {
        Vector::try_with_capacity_in(cap, Global)
    }

    pub fn from_slice(data: &[T]) -> Self
    where
        T: Clone,
    {
        Vector { raw: RawVector::try_from_slice(&Global, data).unwrap(), allocator: Global }
    }

    /// Creates a vector with `n` clones of `elem`.
    pub fn from_elem(elem: T, n: usize) -> Vector<T, Global>
    where
        T: Clone,
    {
        Vector { raw: RawVector::try_from_elem(&Global, elem, n).unwrap(), allocator: Global }
    }
}

impl<T, A: Allocator> Vector<T, A> {
    /// Creates an empty vector without allocating memory.
    pub fn new_in(allocator: A) -> Self {
        Self::try_with_capacity_in(0, allocator).unwrap()
    }

    /// Creates an empty pre-allocated vector with a given storage capacity.
    ///
    /// Does not allocate memory if `cap` is zero.
    pub fn with_capacity_in(cap: usize, allocator: A) -> Self {
        Self::try_with_capacity_in(cap, allocator).unwrap()
    }

    /// Creates an empty pre-allocated vector with a given storage capacity.
    ///
    /// Does not allocate memory if `cap` is zero.
    pub fn try_with_capacity_in(cap: usize, allocator: A) -> Result<Vector<T, A>, AllocError> {
        let raw = RawVector::try_with_capacity(&allocator, cap)?;

        Ok(Vector { raw, allocator })
    }


    #[inline(always)]
    /// Returns `true` if the vector contains no elements.
    pub fn is_empty(&self) -> bool {
        self.raw.is_empty()
    }

    #[inline(always)]
    /// Returns the number of elements in the vector, also referred to as its ‘length’.
    pub fn len(&self) -> usize {
        self.raw.len()
    }

    #[inline(always)]
    /// Returns the total number of elements the vector can hold without reallocating.
    pub fn capacity(&self) -> usize {
        self.raw.capacity()
    }

    /// Returns number of elements that can be added without reallocating.
    #[inline(always)]
    pub fn remaining_capacity(&self) -> usize {
        self.raw.remaining_capacity()
    }

    /// Returns a reference to the underlying allocator.
    #[inline(always)]
    pub fn allocator(&self) -> &A {
        &self.allocator
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        self.raw.as_slice()
    }

    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.raw.as_mut_slice()
    }

    /// Clears the vector, removing all values.
    pub fn clear(&mut self) {
        unsafe {
            raw::clear(self.raw.data_ptr(), &mut self.raw.header)
        }
    }

    unsafe fn into_header_buffer<R>(mut self) -> HeaderBuffer<T, R, A>
    where
        R: RefCount,
    {
        debug_assert!(self.raw.header.cap != 0);
        unsafe {
            let mut header = raw::header_from_data_ptr(self.raw.data);

            *header.as_mut() = raw::Header {
                vec: VecHeader {
                    len: self.raw.header.len,
                    cap: self.raw.header.cap,
                },
                ref_count: R::new(1),
                allocator: ptr::read(&mut self.allocator),
            };

            mem::forget(self);

            HeaderBuffer::from_raw(header)
        }
    }

    /// Make this vector immutable.
    ///
    /// This operation is cheap, the underlying storage does not not need
    /// to be reallocated.
    #[inline]
    pub fn into_shared(self) -> SharedVector<T, A>
    where
        A: Allocator + Clone,
    {
        if self.raw.header.cap == 0 {
            return SharedVector::try_with_capacity_in(0, self.allocator.clone()).unwrap();
        }
        unsafe {
            let inner = self.into_header_buffer::<DefaultRefCount>();
            SharedVector { inner }
        }
    }

    /// Make this vector immutable.
    ///
    /// This operation is cheap, the underlying storage does not not need
    /// to be reallocated.
    #[inline]
    pub fn into_shared_atomic(self) -> AtomicSharedVector<T, A>
    where
        A: Allocator + Clone,
    {
        if self.raw.header.cap == 0 {
            return AtomicSharedVector::try_with_capacity_in(0, self.allocator.clone()).unwrap();
        }
        unsafe {
            let inner = self.into_header_buffer::<AtomicRefCount>();
            AtomicSharedVector { inner }
        }
    }

    /// Appends an element to the back of a collection.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity exceeds `u32::MAX` bytes.
    #[inline(always)]
    pub fn push(&mut self, val: T) {
        unsafe {
            self.raw.push(&self.allocator, val);
        }
    }

    /// Appends an element if there is sufficient spare capacity, otherwise an error is returned
    /// with the element.
    ///
    /// Unlike push this method will not reallocate when there’s insufficient capacity.
    /// The caller should use reserve or try_reserve to ensure that there is enough capacity.
    #[inline(always)]
    pub fn push_within_capacity(&mut self, val: T) -> Result<(), T> {
        self.raw.push_within_capacity(val)
    }

    /// Removes the last element from the vector and returns it, or `None` if it is empty.
    #[inline(always)]
    pub fn pop(&mut self) -> Option<T> {
        self.raw.pop()
    }

    /// Removes and returns the element at position `index` within the vector,
    /// shifting all elements after it to the left.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    ///
    #[inline(always)]
    pub fn remove(&mut self, index: usize) -> T {
        self.raw.remove(index)
    }

    /// Removes an element from the vector and returns it.
    ///
    /// The removed element is replaced by the last element of the vector.
    ///
    /// # Panics
    ///
    /// Panics if index is out of bounds.
    #[inline(always)]
    pub fn swap_remove(&mut self, idx: usize) -> T {
        self.raw.swap_remove(idx)
    }

    /// Inserts an element at position `index` within the vector, shifting all
    /// elements after it to the right.
    ///
    /// # Panics
    ///
    /// Panics if `index > len`.
    #[inline(always)]
    pub fn insert(&mut self, index: usize, element: T) {
       unsafe { self.raw.insert(&self.allocator, index, element) }
    }

    /// Clones and appends the contents of the slice to the back of a collection.
    #[inline(always)]
    pub fn extend_from_slice(&mut self, data: &[T])
    where
        T: Clone,
    {
        unsafe {
            self.raw.extend_from_slice(&self.allocator, data)
        }
    }

    /// Moves all the elements of `other` into `self`, leaving `other` empty.
    #[inline(always)]
    pub fn append(&mut self, other: &mut Self)
    where
        T: Clone,
    {
        unsafe {
            self.raw.append(&self.allocator, &mut other.raw)
        }
    }

    /// Appends the contents of an iterator to the back of a collection.
    #[inline(always)]
    pub fn extend(&mut self, data: impl IntoIterator<Item = T>) {
        unsafe {
            self.raw.extend(&self.allocator, data)
        }
    }

    /// Allocates a clone of this buffer.
    #[inline(always)]
    pub fn clone_buffer(&self) -> Self
    where
        T: Clone,
        A: Clone,
    {
        Vector {
            raw: self.raw.clone_buffer(&self.allocator),
            allocator: self.allocator.clone(),
        }
    }

    /// Allocate a clone of this buffer with a different capacity
    ///
    /// The capacity must be at least as large as the buffer's length.
    #[inline(always)]
    pub fn clone_buffer_with_capacity(&self, cap: usize) -> Self
    where
        T: Clone,
        A: Clone,
    {
        Vector {
            raw: self.raw.clone_buffer_with_capacity(&self.allocator, cap),
            allocator: self.allocator.clone(),
        }
    }

    #[inline(always)]
    pub fn reserve(&mut self, additional: usize) {
        unsafe {
            self.raw.try_reserve(&self.allocator, additional).unwrap()
        }
    }

    #[inline(always)]
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), AllocError> {
        unsafe {
            self.raw.try_reserve(&self.allocator, additional)
        }
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
    pub fn try_reserve_exact(&mut self, additional: usize) -> Result<(), AllocError> {
        unsafe {
            self.raw.try_reserve_exact(&self.allocator, additional)
        }
    }

    /// Shrinks the capacity of the vector with a lower bound.
    ///
    /// The capacity will remain at least as large as both the length and the supplied value.
    /// If the current capacity is less than the lower limit, this is a no-op.
    #[inline(always)]
    pub fn shrink_to(&mut self, min_capacity: usize)
    where
        T: Clone,
    {
        unsafe {
            self.raw.shrink_to(&self.allocator, min_capacity)
        }
    }

    /// Shrinks the capacity of the vector as much as possible.
    #[inline(always)]
    pub fn shrink_to_fit(&mut self)
    where
        T: Clone,
    {
        unsafe {
            self.raw.shrink_to_fit(&self.allocator)
        }
    }

    /// Removes the specified range from the vector in bulk, returning all
    /// removed elements as an iterator. If the iterator is dropped before
    /// being fully consumed, it drops the remaining removed elements.
    ///
    /// The returned iterator keeps a mutable borrow on the vector to optimize
    /// its implementation.
    ///
    /// # Panics
    ///
    /// Panics if the starting point is greater than the end point or if
    /// the end point is greater than the length of the vector.
    ///
    /// # Leaking
    ///
    /// If the returned iterator goes out of scope without being dropped (due to
    /// [`mem::forget`], for example), the vector may have lost and leaked
    /// elements arbitrarily, including elements outside the range.
    ///
    pub fn drain<R>(&mut self, range: R) -> Drain<'_, T>
    where
        R: RangeBounds<usize>,
    {
        self.raw.drain(range)
    }

    /// Creates a splicing iterator that replaces the specified range in the vector
    /// with the given `replace_with` iterator and yields the removed items.
    /// `replace_with` does not need to be the same length as `range`.
    ///
    /// `range` is removed even if the iterator is not consumed until the end.
    ///
    /// It is unspecified how many elements are removed from the vector
    /// if the `Splice` value is leaked.
    ///
    /// The input iterator `replace_with` is only consumed when the `Splice` value is dropped.
    ///
    /// This is optimal if:
    ///
    /// * The tail (elements in the vector after `range`) is empty,
    /// * or `replace_with` yields fewer or equal elements than `range`’s length
    /// * or the lower bound of its `size_hint()` is exact.
    ///
    /// Otherwise, a temporary vector is allocated and the tail is moved twice.
    ///
    /// # Panics
    ///
    /// Panics if the starting point is greater than the end point or if
    /// the end point is greater than the length of the vector.
    ///
    pub fn splice<R, I>(
        &mut self,
        range: R,
        replace_with: I
    ) -> Splice<'_, <I as IntoIterator>::IntoIter, A>
    where
        R: RangeBounds<usize>,
        I: IntoIterator<Item = T>,
    {
        unsafe { self.raw.splice(&self.allocator, range, replace_with) }
    }

    /// Retains only the elements specified by the predicate.
    ///
    /// In other words, remove all elements `e` for which `f(&e)` returns `false`.
    /// This method operates in place, visiting each element exactly once in the
    /// original order, and preserves the order of the retained elements.
    pub fn retain<F>(&mut self, f: F)
    where
        F: FnMut(&T) -> bool,
    {
        self.raw.retain(f)
    }

    /// Retains only the elements specified by the predicate, passing a mutable reference to it.
    ///
    /// In other words, remove all elements `e` such that `f(&mut e)` returns `false`.
    /// This method operates in place, visiting each element exactly once in the
    /// original order, and preserves the order of the retained elements.
    pub fn retain_mut<F>(&mut self, f: F)
    where
        F: FnMut(&mut T) -> bool,
    {
        self.raw.retain_mut(f)
    }

    #[inline(always)]
    pub fn take(&mut self) -> Self
    where
        A: Clone,
    {
        let other = Vector {
            raw: self.raw.take(),
            allocator: self.allocator.clone(),
        };

        other
    }
}

impl<T, A: Allocator> Drop for Vector<T, A> {
    fn drop(&mut self) {
        unsafe {
            self.raw.deallocate(&self.allocator)
        }
    }
}

impl<T: Clone, A: Allocator + Clone> Clone for Vector<T, A> {
    fn clone(&self) -> Self {
        self.clone_buffer()
    }
}

impl<T: PartialEq<T>, A: Allocator> PartialEq<Vector<T, A>> for Vector<T, A> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: PartialEq<T>, A: Allocator> PartialEq<&[T]> for Vector<T, A> {
    fn eq(&self, other: &&[T]) -> bool {
        self.as_slice() == *other
    }
}

impl<T, A: Allocator> AsRef<[T]> for Vector<T, A> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T, A: Allocator> AsMut<[T]> for Vector<T, A> {
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T> Default for Vector<T, Global> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, T, A: Allocator> IntoIterator for &'a Vector<T, A> {
    type Item = &'a T;
    type IntoIter = core::slice::Iter<'a, T>;
    fn into_iter(self) -> core::slice::Iter<'a, T> {
        self.as_slice().iter()
    }
}

impl<'a, T, A: Allocator> IntoIterator for &'a mut Vector<T, A> {
    type Item = &'a mut T;
    type IntoIter = core::slice::IterMut<'a, T>;
    fn into_iter(self) -> core::slice::IterMut<'a, T> {
        self.as_mut_slice().iter_mut()
    }
}

impl<T, A: Allocator, I> Index<I> for Vector<T, A>
where
    I: core::slice::SliceIndex<[T]>,
{
    type Output = <I as core::slice::SliceIndex<[T]>>::Output;
    fn index(&self, index: I) -> &Self::Output {
        self.as_slice().index(index)
    }
}

impl<T, A: Allocator, I> IndexMut<I> for Vector<T, A>
where
    I: core::slice::SliceIndex<[T]>,
{
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        self.as_mut_slice().index_mut(index)
    }
}

impl<T, A: Allocator> Deref for Vector<T, A> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T, A: Allocator> DerefMut for Vector<T, A> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T: Debug, A: Allocator> Debug for Vector<T, A> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> Result<(), core::fmt::Error> {
        self.as_slice().fmt(f)
    }
}

impl<T: Clone, A: Allocator + Clone> From<SharedVector<T, A>> for Vector<T, A> {
    fn from(shared: SharedVector<T, A>) -> Self {
        shared.into_unique()
    }
}

impl<T: Clone, A: Allocator + Clone> From<AtomicSharedVector<T, A>> for Vector<T, A> {
    fn from(shared: AtomicSharedVector<T, A>) -> Self {
        shared.into_unique()
    }
}

impl<T: core::hash::Hash, A: Allocator> core::hash::Hash for Vector<T, A> {
    fn hash<H>(&self, state: &mut H) where H: core::hash::Hasher {
        self.as_slice().hash(state)
    }
}

#[test]
fn bump_alloc() {
    use blink_alloc::BlinkAlloc;

    let allocator = BlinkAlloc::new();

    {
        let mut v1: Vector<u32, &BlinkAlloc> = Vector::try_with_capacity_in(4, &allocator).unwrap();
        v1.push(0);
        v1.push(1);
        v1.push(2);
        assert_eq!(v1.capacity(), 4);
        assert_eq!(v1.as_slice(), &[0, 1, 2]);

        // let mut v2 = crate::vector!(@ &allocator [10, 11]);
        let mut v2 = crate::vector!([10, 11] in &allocator);
        assert_eq!(v2.capacity(), 2);

        assert_eq!(v2.as_slice(), &[10, 11]);

        v1.push(3);
        v1.push(4);

        assert_eq!(v1.as_slice(), &[0, 1, 2, 3, 4]);

        assert!(v1.capacity() > 4);

        v2.push(12);
        v2.push(13);
        v2.push(14);

        let v2 = v2.into_shared();

        assert_eq!(v1.as_slice(), &[0, 1, 2, 3, 4]);
        assert_eq!(v2.as_slice(), &[10, 11, 12, 13, 14]);
    }
}

#[test]
fn basic_unique() {
    fn num(val: u32) -> Box<u32> {
        Box::new(val)
    }

    let mut a = Vector::with_capacity(256);

    a.push(num(0));
    a.push(num(1));
    a.push(num(2));

    let a = a.into_shared();

    assert_eq!(a.len(), 3);

    assert_eq!(a.as_slice(), &[num(0), num(1), num(2)]);

    assert!(a.is_unique());

    let b = Vector::from_slice(&[num(0), num(1), num(2), num(3), num(4)]);

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

    let mut d = Vector::with_capacity(64);
    d.extend_from_slice(&[num(0), num(1), num(2)]);
    d.extend_from_slice(&[]);
    d.extend_from_slice(&[num(3), num(4)]);

    assert_eq!(d.as_slice(), &[num(0), num(1), num(2), num(3), num(4)]);
}

#[test]
fn shrink() {
    let mut v: Vector<u32> = Vector::with_capacity(32);
    v.shrink_to(8);
}

#[test]
fn zst() {
    let mut v = Vector::new();
    v.push(());
    v.push(());
    v.push(());
    v.push(());

    assert_eq!(v.len(), 4);
}

#[test]
fn dyn_allocator() {
    let allocator: &dyn Allocator = &Global;
    let mut v = crate::vector!([1u32, 2, 3] in allocator);

    v.push(4);

    assert_eq!(&v[..], &[1, 2, 3, 4]);
}

#[test]
fn borrowd_dyn_alloc() {
    struct DataStructure<'a> {
        data: Vector<u32, &'a dyn Allocator>,
    }

    impl DataStructure<'static> {
        fn new() -> DataStructure<'static> {
            DataStructure {
                data: Vector::new_in(&Global as &'static dyn Allocator)
            }
        }
    }

    impl<'a> DataStructure<'a> {
        fn new_in(allocator: &'a dyn Allocator) -> DataStructure<'a> {
            DataStructure { data: Vector::new_in(allocator) }
        }

        fn push(&mut self, val: u32) {
            self.data.push(val);
        }
    }

    let mut ds1 = DataStructure::new();
    ds1.push(1);

    let alloc = Global; 
    let mut ds2 = DataStructure::new_in(&alloc);
    ds2.push(2);

}

#[test]
fn splice1() {
    let mut vec = Vector::new();
    vec.splice(0..0, vec![Box::new(1); 5].into_iter());
    vec.splice(0..0, vec![Box::new(2); 5].into_iter());
}

#[test]
fn drain1() {
    let mut vectors: [Vector<Box<u32>>; 4] = [
        Vector::new(),
        Vector::new(),
        Vector::new(),
        Vector::new(),
    ];
    vectors[0].shrink_to(3906369431118283232);
    vectors[2].extend_from_slice(&[Box::new(1), Box::new(2), Box::new(3)]);
    let vec = &mut vectors[2];
    let len = vec.len();
    let start = if len > 0 { 16059518370053021184 % len } else { 0 };
    let end = 16059518370053021185.min(len);
    vectors[2].drain(start..end);
}
