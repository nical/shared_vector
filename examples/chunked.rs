use shared_vector::{
    AtomicRefCount, BufferSize, DefaultRefCount, RefCount, RefCountedVector, SharedVector, Vector,
};

pub type AtomicSharedChunkVector<T> = RefCountedChunkVector<T, AtomicRefCount>;
pub type SharedChunkVector<T> = RefCountedChunkVector<T, DefaultRefCount>;

/// A reference counted container split into multiple chunks of contiguous data.
///
/// <svg width="810" height="160" viewBox="0 0 214.31 42.33" xmlns:xlink="http://www.w3.org/1999/xlink" xmlns="http://www.w3.org/2000/svg"><defs><linearGradient id="a"><stop offset="0" stop-color="#141b62"/><stop offset="1" stop-color="#7a2c92"/></linearGradient><linearGradient id="b"><stop offset="0" stop-color="#491c9c"/><stop offset="1" stop-color="#d54b27"/></linearGradient><linearGradient xlink:href="#a" id="d" gradientUnits="userSpaceOnUse" x1="6.27" y1="34.86" x2="87.72" y2="13.24" gradientTransform="translate(-2.64 -18.48)"/><linearGradient xlink:href="#b" id="f" gradientUnits="userSpaceOnUse" gradientTransform="translate(-2.64 -18.48)" x1="6.27" y1="34.86" x2="87.72" y2="13.24"/><linearGradient xlink:href="#b" id="e" gradientUnits="userSpaceOnUse" gradientTransform="translate(-2.64 -18.48)" x1="6.27" y1="34.86" x2="87.72" y2="13.24"/><linearGradient xlink:href="#b" id="g" gradientUnits="userSpaceOnUse" gradientTransform="translate(-3.96 -19.01)" x1="6.27" y1="34.86" x2="87.72" y2="13.24"/><linearGradient xlink:href="#a" id="c" gradientUnits="userSpaceOnUse" gradientTransform="translate(-2.64 -18.48)" x1="6.27" y1="34.86" x2="87.72" y2="13.24"/></defs><g transform="translate(22.49 7.94)"><rect width="10.57" height="10.66" x="84.67" y="-5.29" ry="1.37" fill="#3dbdaa"/><circle cx="89.96" cy="5.29" r=".79" fill="#666"/><path d="M99.22 7.94c-3.97 0-9.26 0-9.26-2.65" fill="none" stroke="#999" stroke-width=".86" stroke-linecap="round"/><g transform="translate(96.57 -.04)"><rect width="68.79" height="10.58" x="2.65" y="2.68" ry="1.37" fill="url(#c)"/><rect width="15.35" height="9.51" x="3.18" y="3.21" ry=".9" fill="#78a2d4"/><rect width="9.26" height="9.51" x="19.85" y="3.2" ry=".9" fill="#8e9ddd"/><rect width="9.26" height="9.51" x="29.64" y="3.22" ry=".9" fill="#8e9ddd"/><rect width="9.26" height="9.51" x="39.43" y="3.22" ry=".9" fill="#8e9ddd"/><rect width="9.26" height="9.51" x="49.22" y="3.21" ry=".9" fill="#7785c1"/><circle cx="62.84" cy="7.97" r=".66" fill="#eaa577"/><circle cx="64.7" cy="7.97" r=".66" fill="#eaa577"/><circle cx="66.55" cy="7.97" r=".66" fill="#eaa577"/></g><circle cx="120.91" cy="12.7" r=".79" fill="#666"/><circle cx="130.7" cy="12.7" r=".79" fill="#666"/><circle cx="140.49" cy="12.7" r=".79" fill="#666"/><path d="M54.24 21.17c0-5.3 76.46 0 76.46-8.47" fill="none" stroke="#b3b3b3" stroke-width=".86" stroke-linecap="round"/><path d="M-15.88 21.17c0-6.62 136.8-1.86 136.8-8.47" fill="none" stroke="#b3b3b3" stroke-width=".86" stroke-linecap="round"/><rect width="10.57" height="10.66" x="-6.61" y="-5.37" ry="1.37" fill="#3dbdaa"/><rect width="10.57" height="10.66" x="-19.83" y="-5.33" ry="1.37" fill="#3dbdaa"/><circle cx="-14.55" cy="5.29" r=".79" fill="#666"/><circle cx="-1.32" cy="5.29" r=".79" fill="#666"/><path d="M7.94 7.94c-3.97 0-9.26 0-9.26-2.65m9.26 3.97c-3.97 0-22.5 0-22.5-3.97" fill="none" stroke="#999" stroke-width=".86" stroke-linecap="round"/><g transform="translate(5.29 -.04)"><rect width="68.79" height="10.58" x="2.65" y="2.68" ry="1.37" fill="url(#d)"/><rect width="15.35" height="9.51" x="3.18" y="3.21" ry=".9" fill="#78a2d4"/><rect width="9.26" height="9.51" x="19.85" y="3.2" ry=".9" fill="#8e9ddd"/><rect width="9.26" height="9.51" x="29.64" y="3.22" ry=".9" fill="#8e9ddd"/><rect width="9.26" height="9.51" x="39.43" y="3.22" ry=".9" fill="#7785c1"/><rect width="9.26" height="9.51" x="49.22" y="3.21" ry=".9" fill="#7785c1"/><circle cx="62.84" cy="7.97" r=".66" fill="#eaa577"/><circle cx="64.7" cy="7.97" r=".66" fill="#eaa577"/><circle cx="66.55" cy="7.97" r=".66" fill="#eaa577"/></g><circle cx="30.43" cy="12.7" r=".79" fill="#666"/><circle cx="39.69" cy="12.7" r=".79" fill="#666"/><path d="M-17.2 21.17c0-6.62 47.63-1.86 47.63-8.47m22.49 8.47c0-6.62-13.23-1.86-13.23-8.47" fill="none" stroke="#999" stroke-width=".86" stroke-linecap="round"/><g transform="translate(47.62 18.48)"><rect width="68.79" height="10.58" x="2.65" y="2.68" ry="1.37" fill="url(#e)"/><rect width="15.35" height="9.51" x="3.18" y="3.21" ry=".9" fill="#78a2d4"/><rect width="9.26" height="9.51" x="19.85" y="3.2" ry=".9" fill="#eaa577"/><rect width="9.26" height="9.51" x="29.64" y="3.22" ry=".9" fill="#eaa577"/><rect width="9.26" height="9.51" x="39.43" y="3.22" ry=".9" fill="#eaa577"/><rect width="9.26" height="9.51" x="49.22" y="3.21" ry=".9" fill="#eaa577"/><circle cx="62.84" cy="7.97" r=".66" fill="#eaa577"/><circle cx="64.7" cy="7.97" r=".66" fill="#eaa577"/><circle cx="66.55" cy="7.97" r=".66" fill="#eaa577"/></g><g transform="translate(-22.5 18.48)"><rect width="68.79" height="10.58" x="2.65" y="2.68" ry="1.37" fill="url(#f)"/><rect width="15.35" height="9.51" x="3.18" y="3.21" ry=".9" fill="#78a2d4"/><rect width="9.26" height="9.51" x="19.85" y="3.2" ry=".9" fill="#eaa577"/><rect width="9.26" height="9.51" x="29.64" y="3.22" ry=".9" fill="#eaa577"/><rect width="9.26" height="9.51" x="39.43" y="3.22" ry=".9" fill="#eaa577"/><rect width="9.26" height="9.51" x="49.22" y="3.21" ry=".9" fill="#eaa577"/><circle cx="62.84" cy="7.97" r=".66" fill="#eaa577"/><circle cx="64.7" cy="7.97" r=".66" fill="#eaa577"/><circle cx="66.55" cy="7.97" r=".66" fill="#eaa577"/></g><path d="M123.03 21.17c0-6.62 17.46-1.86 17.46-8.47" fill="none" stroke="#b3b3b3" stroke-width=".86" stroke-linecap="round"/><g transform="translate(119.06 19.01)"><rect width="68.79" height="10.58" x="1.33" y="2.15" ry="1.37" fill="url(#g)"/><rect width="15.35" height="9.51" x="1.86" y="2.68" ry=".9" fill="#78a2d4"/><rect width="9.26" height="9.51" x="18.53" y="2.68" ry=".9" fill="#eaa577"/><rect width="9.26" height="9.51" x="28.32" y="2.69" ry=".9" fill="#eaa577"/><rect width="9.26" height="9.51" x="38.11" y="2.69" ry=".9" fill="#eaa577"/><rect width="9.26" height="9.51" x="47.89" y="2.68" ry=".9" fill="#ca8659"/><circle cx="61.52" cy="7.45" r=".66" fill="#eaa577"/><circle cx="63.37" cy="7.45" r=".66" fill="#eaa577"/><circle cx="65.23" cy="7.45" r=".66" fill="#eaa577"/></g></g></svg>
///
#[derive(Clone)]
pub struct RefCountedChunkVector<T, R: RefCount> {
    chunks: RefCountedVector<RefCountedVector<T, R>, R>,
    chunk_size: usize,
    len: usize,
}

impl<T, R: RefCount> RefCountedChunkVector<T, R> {
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

    pub fn push(&mut self, val: T)
    where
        T: Clone,
    {
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

    pub fn pop(&mut self) -> Option<T>
    where
        T: Clone,
    {
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

    pub fn chunks_mut(&mut self) -> impl Iterator<Item = &mut [T]>
    where
        T: Clone,
    {
        self.chunks.iter_mut().map(RefCountedVector::as_mut_slice)
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.chunks.iter().flat_map(|chunk| chunk.iter())
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T>
    where
        T: Clone,
    {
        self.chunks.iter_mut().flat_map(|chunk| chunk.iter_mut())
    }

    #[inline]
    pub fn new_ref(&self) -> Self
    where
        T: Clone,
    {
        Self {
            chunks: self.chunks.new_ref(),
            chunk_size: self.chunk_size,
            len: self.len,
        }
    }

    pub fn into_unique(mut self) -> ChunkVector<T>
    where
        T: Clone,
    {
        for chunk in self.chunks.iter_mut() {
            chunk.ensure_unique();
        }

        let last_full = self
            .chunks
            .last()
            .map(|chunk| chunk.remaining_capacity() == 0)
            .unwrap_or(true);
        let head = if last_full {
            Vector::new()
        } else {
            self.chunks.pop().unwrap().into_unique()
        };

        ChunkVector {
            head,
            chunks: unsafe { std::mem::transmute(self.chunks.into_unique()) },
            chunk_size: self.chunk_size,
            len: self.len,
        }
    }
}

pub struct ChunkVector<T> {
    head: Vector<T>,
    chunks: Vector<SharedVector<T>>,
    chunk_size: usize,
    len: usize,
}

impl<T> ChunkVector<T> {
    pub fn new(chunk_size: usize) -> Self {
        assert!(chunk_size > 0);
        assert!(chunk_size < BufferSize::MAX as usize);
        ChunkVector {
            head: Vector::new(),
            chunks: Vector::new(),
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

    pub fn push(&mut self, val: T)
    where
        T: Clone,
    {
        if self.head.capacity() == 0 {
            self.head.reserve(self.chunk_size);
        }

        self.head.push(val);
        self.len += 1;

        if self.head.remaining_capacity() == 0 {
            let chunk = std::mem::replace(&mut self.head, Vector::new());
            self.chunks.push(chunk.into_shared());
        }
    }

    pub fn pop(&mut self) -> Option<T>
    where
        T: Clone,
    {
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
        unsafe { std::mem::transmute(self.into_shared()) }
    }

    pub fn chunks(&self) -> impl Iterator<Item = &[T]> {
        let head: Option<&[T]> = if self.head.is_empty() {
            None
        } else {
            Some(self.head.as_slice())
        };

        self.chunks
            .iter()
            .map(RefCountedVector::as_slice)
            .chain(head)
    }

    pub fn chunks_mut(&mut self) -> impl Iterator<Item = &mut [T]>
    where
        T: Clone,
    {
        let head: Option<&mut [T]> = if self.head.is_empty() {
            None
        } else {
            Some(self.head.as_mut_slice())
        };

        self.chunks
            .iter_mut()
            .map(RefCountedVector::as_mut_slice)
            .chain(head)
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.chunks
            .iter()
            .flat_map(|chunk| chunk.iter())
            .chain(self.head.iter())
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T>
    where
        T: Clone,
    {
        self.chunks
            .iter_mut()
            .flat_map(|chunk| chunk.iter_mut())
            .chain(self.head.iter_mut())
    }
}

fn chunks_basic() {
    chunks_basic_impl::<DefaultRefCount>();
    chunks_basic_impl::<AtomicRefCount>();

    fn chunks_basic_impl<R: RefCount>() {
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

fn unique_chunks() {
    let mut v = ChunkVector::new(100);

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

fn main() {
    chunks_basic();
    unique_chunks();
}
