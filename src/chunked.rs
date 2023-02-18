use crate::{raw::BufferSize, vector::{ReferenceCount, RefCountedVector, AtomicRefCount, DefaultRefCount}, UniqueVector, SharedVector};

pub type AtomicSharedChunkVector<T> = RefCountedChunkVector<T, AtomicRefCount>;
pub type SharedChunkVector<T> = RefCountedChunkVector<T, DefaultRefCount>;

#[derive(Clone)]
pub struct RefCountedChunkVector<T, R: ReferenceCount> {
    chunks: RefCountedVector<RefCountedVector<T, R>, R>,
    chunk_size: usize,
    len: usize,
}

impl<T, R: ReferenceCount> RefCountedChunkVector<T, R> {
    pub fn new(chunk_size: usize) -> Self {
        assert!(chunk_size < BufferSize::MAX as usize);
        RefCountedChunkVector {
            chunks: RefCountedVector::new(),
            chunk_size,
            len: 0,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn num_chunks(&self) -> usize {
        self.chunks.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn push(&mut self, val: T) where T: Clone {
        if let Some(last) = self.chunks.last_mut() {
            if last.remaining_capacity() > 0 {
                last.push(val);
                self.len += 1;
                return;
            }
        }

        let mut new_chunk = RefCountedVector::with_capacity(self.chunk_size);
        new_chunk.push(val);
        self.chunks.push(new_chunk);
        self.len += 1;
    }

    pub fn pop(&mut self) -> Option<T> where T: Clone {
        let mut result = None;
        let mut pop_chunk = false;
        if let Some(chunk) = self.chunks.last_mut() {
            result = chunk.pop();
            pop_chunk = chunk.is_empty();
            if result.is_some() {
                self.len -= 1;
            }
        }

        if pop_chunk {
            self.chunks.pop();
        }

        result
    }

    pub fn push_chunk(&mut self, chunk: RefCountedVector<T, R>) {
        self.len += chunk.len();
        self.chunks.push(chunk);
    }

    pub fn pop_chunk(&mut self) -> Option<RefCountedVector<T, R>> {
        if let Some(chunk) = self.chunks.pop() {
            self.len -= chunk.len();
            return Some(chunk);
        }

        None
    }

    pub fn clear(&mut self) {
        self.chunks.clear();
        self.len = 0;
    }

    pub fn chunks(&self) -> impl Iterator<Item = &[T]> {
        self.chunks.iter().map(RefCountedVector::as_slice)
    }

    pub fn chunks_mut(&mut self) -> impl Iterator<Item = &mut [T]> where T: Clone {
        self.chunks.iter_mut().map(RefCountedVector::as_mut_slice)
    }
    
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.chunks.iter().flat_map(RefCountedVector::iter)
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> where T: Clone {
        self.chunks.iter_mut().flat_map(RefCountedVector::iter_mut)
    }

    #[inline]
    pub fn new_ref(&self) -> Self where T: Clone {
        Self {
            chunks: self.chunks.new_ref(),
            chunk_size: self.chunk_size,
            len: self.len,
        }
    }

    pub fn into_unique(mut self) -> UniqueChunkVector<T> where T: Clone {
        for chunk in self.chunks.iter_mut() {
            chunk.ensure_unique();
        }

        let last_full = self.chunks.last().map(|chunk| chunk.remaining_capacity() == 0).unwrap_or(true);
        let head = if last_full {
            UniqueVector::new()
        } else {
            self.chunks.pop().unwrap().into_unique()
        };

        UniqueChunkVector {
            head,
            chunks: unsafe { std:: mem::transmute(self.chunks.into_unique()) },
            chunk_size: self.chunk_size,
            len: self.len,
        }
    }
}

pub struct UniqueChunkVector<T> {
    head: UniqueVector<T>,
    chunks: UniqueVector<SharedVector<T>>,
    chunk_size: usize,
    len: usize,
}

impl<T> UniqueChunkVector<T> {
    pub fn new(chunk_size: usize) -> Self {
        assert!(chunk_size > 0);
        assert!(chunk_size < BufferSize::MAX as usize);
        UniqueChunkVector {
            head: UniqueVector::new(),
            chunks: UniqueVector::new(),
            chunk_size,
            len: 0,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn num_chunks(&self) -> usize {
        self.chunks.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn push(&mut self, val: T) where T: Clone {
        if self.head.capacity() == 0 {
            self.head.reserve(self.chunk_size);
        }

        self.head.push(val);
        self.len += 1;

        if self.head.remaining_capacity() == 0 {
            let chunk = std::mem::replace(&mut self.head, UniqueVector::new());
            self.chunks.push(chunk.into_shared());
        }
    }

    pub fn pop(&mut self) -> Option<T> where T: Clone {
        if self.head.is_empty() {
            if let Some(chunk) = self.chunks.pop() {
                self.head = chunk.into_unique();
            }
        }

        let result = self.head.pop();

        if result.is_some() {
            self.len -= 1;
        }

        result
    }

    pub fn into_shared(mut self) -> SharedChunkVector<T> {
        if !self.head.is_empty() {
            self.chunks.push(self.head.into_shared());
        }

        SharedChunkVector {
            chunks: self.chunks.into_shared(),
            len: self.len,
            chunk_size: self.chunk_size,
        }
    }

    pub fn into_shared_atomic(self) -> AtomicSharedChunkVector<T> {
        unsafe {
            std::mem::transmute(self.into_shared())
        }
    }

    pub fn chunks(&self) -> impl Iterator<Item = &[T]> {
        let head: Option<&[T]> = if self.head.is_empty() {
            None
        } else {
            Some(self.head.as_slice())
        };

        self.chunks.iter().map(RefCountedVector::as_slice).chain(head)
    }

    pub fn chunks_mut(&mut self) -> impl Iterator<Item = &mut[T]> where T: Clone {
        let head: Option<&mut[T]> = if self.head.is_empty() {
            None
        } else {
            Some(self.head.as_mut_slice())
        };

        self.chunks.iter_mut().map(RefCountedVector::as_mut_slice).chain(head)
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.chunks.iter().flat_map(SharedVector::iter).chain(self.head.iter())
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> where T: Clone {
        self.chunks.iter_mut().flat_map(SharedVector::iter_mut).chain(self.head.iter_mut())
    }
}

#[test]
fn chunks_basic() {
    chunks_basic_impl::<DefaultRefCount>();
    chunks_basic_impl::<AtomicRefCount>();

    fn chunks_basic_impl<R: ReferenceCount>() {
        let mut v: RefCountedChunkVector<u32, R> = RefCountedChunkVector::new(16);
        for i in 0..40 {
            v.push(i);
            assert_eq!(v.len(), i as usize + 1);
        }
        let mut v2 = v.new_ref();
        for i in 40..80 {
            v.push(i);
            assert_eq!(v.len(), i as usize + 1);
        }

        let items: Vec<u32> = v.iter().cloned().collect();
        assert_eq!(items.len(), 80);
        for i in 0..80 {
            assert_eq!(items[i], i as u32);
        }

        let items: Vec<u32> = v2.iter().cloned().collect();
        assert_eq!(items.len(), 40);
        for i in 0..40 {
            assert_eq!(items[i], i as u32);
        }

        for i in 0..80 {
            let idx = 79 - i;
            assert_eq!(v.pop(), Some(idx));
            assert_eq!(v.len(), idx as usize);
        }

        v2.clear();
    
        assert!(v.pop().is_none());
        assert!(v2.pop().is_none());
    }
}

#[test]
fn unique_chunks() {
    let mut v = UniqueChunkVector::new(100);

    for i in 0..512u32 {
        v.push(i);
        assert_eq!(v.len(), i as usize + 1);
    }

    assert_eq!(v.len(), 512);

    for i in 0..50 {
        let idx = 511 - i;
        assert_eq!(v.pop(), Some(idx));
        assert_eq!(v.len(), idx as usize);
    }

    assert_eq!(v.len(), 462);

    let items: Vec<u32> = v.iter().cloned().collect();
    assert_eq!(items.len(), 462);
    for i in 0..462 {
        assert_eq!(items[i], i as u32);
    }

    let shared = v.into_shared();

    assert_eq!(shared.len(), 462);

    let items: Vec<u32> = shared.iter().cloned().collect();
    assert_eq!(items.len(), 462);
    for i in 0..462 {
        assert_eq!(items[i], i as u32);
    }
}
