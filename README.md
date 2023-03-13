Shared and mutable vectors.

- [crate](https://crates.io/crates/shared_vector)
- [doc](https://docs.rs/shared_vector)

# Overview

This crate provides the following two types:
- `SharedVector<T>`/`AtomicSharedVector<T>`, an immutable reference counted vector (with an atomically
   reference counted variant).
- `Vector<T>`, an unique vector type with an API similar to `std::Vec<T>`.

Internally these types are a little different from the standard `Vec<T>`.
`SharedVector` and `AtomicSharedVector` hold a single pointer to a buffer containing:
- A 16 bytes header (plus possible padding for alignment),
- the contiguous sequence of items of type `T`.

The header is where the vector's length, caoacity and reference count are stored.

<svg width="280" height="120" viewBox="0 0 74.08 31.75" xmlns:xlink="http://www.w3.org/1999/xlink" xmlns="http://www.w3.org/2000/svg"><defs><linearGradient id="a"><stop offset="0" stop-color="#491c9c"/><stop offset="1" stop-color="#d54b27"/></linearGradient><linearGradient xlink:href="#a" id="b" gradientUnits="userSpaceOnUse" x1="6.27" y1="34.86" x2="87.72" y2="13.24" gradientTransform="translate(-2.64 -18.48)"/></defs><rect width="10.57" height="10.66" x="2.66" y="18.48" ry="1.37" fill="#3dbdaa"/><rect width="10.57" height="10.66" x="15.88" y="18.52" ry="1.37" fill="#3dbdaa"/><rect width="10.57" height="10.66" x="29.11" y="18.52" ry="1.37" fill="#3dbdaa"/><circle cx="33.87" cy="18.56" r=".79" fill="#666"/><circle cx="7.41" cy="18.56" r=".79" fill="#666"/><circle cx="20.64" cy="18.56" r=".79" fill="#666"/><path d="M7.38 18.54c.03-2.63-3.41-2.66-3.41-5.31" fill="none" stroke="#999" stroke-width=".86" stroke-linecap="round"/><path d="M20.64 18.56c0-2.91-15.35-1.36-15.35-5.33" fill="none" stroke="#999" stroke-width=".86" stroke-linecap="round"/><path d="M33.87 18.56c0-3.97-27.26-2.68-27.26-5.33" fill="none" stroke="#999" stroke-width=".86" stroke-linecap="round"/><rect width="68.79" height="10.58" x="2.65" y="2.68" ry="1.37" fill="url(#b)"/><rect width="15.35" height="9.51" x="3.18" y="3.21" ry=".9" fill="#78a2d4"/><rect width="9.26" height="9.51" x="19.85" y="3.2" ry=".9" fill="#eaa577"/><rect width="9.26" height="9.51" x="29.64" y="3.22" ry=".9" fill="#eaa577"/><rect width="9.26" height="9.51" x="39.43" y="3.22" ry=".9" fill="#eaa577"/><rect width="9.26" height="9.51" x="49.22" y="3.21" ry=".9" fill="#eaa577"/><circle cx="62.84" cy="7.97" r=".66" fill="#eaa577"/><circle cx="64.7" cy="7.97" r=".66" fill="#eaa577"/><circle cx="66.55" cy="7.97" r=".66" fill="#eaa577"/></svg>

`Vector`'s representation is closer to `Vec<T>`: it stores the length and capacity information inline and only writes them into the header if/when converting into a shared vector. The allocated buffer does leave room for the header so that converting to and from `SharedVector` is fast.

<svg width="280" height="120" viewBox="0 0 74.08 31.75" xmlns:xlink="http://www.w3.org/1999/xlink" xmlns="http://www.w3.org/2000/svg"><defs><linearGradient id="a"><stop offset="0" stop-color="#491c9c"/><stop offset="1" stop-color="#d54b27"/></linearGradient><linearGradient xlink:href="#a" id="b" gradientUnits="userSpaceOnUse" x1="6.27" y1="34.86" x2="87.72" y2="13.24" gradientTransform="translate(-2.64 -18.48)"/></defs><rect width="10.57" height="10.66" x="7.94" y="18.45" ry="1.37" fill="#3dbdaa"/><circle cx="12.7" cy="18.48" r=".79" fill="#666"/><path d="M12.7 18.48c0-3.93 7.14-1.28 7.14-5.25" fill="none" stroke="#999" stroke-width=".86" stroke-linecap="round"/><rect width="68.79" height="10.58" x="2.65" y="2.68" ry="1.37" fill="url(#b)"/><rect width="15.35" height="9.51" x="3.18" y="3.21" ry=".9" fill="#78a2d4"/><rect width="9.26" height="9.51" x="19.85" y="3.2" ry=".9" fill="#eaa577"/><rect width="9.26" height="9.51" x="29.64" y="3.22" ry=".9" fill="#eaa577"/><rect width="9.26" height="9.51" x="39.43" y="3.22" ry=".9" fill="#eaa577"/><rect width="9.26" height="9.51" x="49.22" y="3.21" ry=".9" fill="#eaa577"/><circle cx="62.84" cy="7.97" r=".66" fill="#eaa577"/><circle cx="64.7" cy="7.97" r=".66" fill="#eaa577"/><circle cx="66.55" cy="7.97" r=".66" fill="#eaa577"/></svg>

This allows very cheap conversion between the two:
- shared to unique: a new allocation is made only if there are multiple handles to the same buffer (the reference count is greather than one).
- unique to shared: always fast since unique buffers are guaranteed to be sole owners of their buffer.

# Use cases

## `Arc<Vec<T>>` without the indirection.

A mutable vector can be be built using a Vec-style API, and then made immutable and reference counted for various use case (easy multi-threading or simply shared ownership).

Using the standard library one might be tempted to first build a `Vec<T>` and share it via `Arc<Vec<T>>`. This is a fine approach at the cost of an extra pointer indirection that could be avoid in principle. Another approach is to share it as an `Arc<[T]>` which removes the indirection at the cost of the need to copy from the vector.

Using this crate there is no extra indirection in the resulting shared vector nor any copy between the unique and shared versions.

```
use shared_vector::Vector;
let mut builder = Vector::new();
builder.push(1u32);
builder.push(2);
builder.push(3);
// Make it reference counted, no allocation.
let mut shared = builder.into_shared();
// We can now create new references
let shared_2 = shared.new_ref();
let shared_3 = shared.new_ref();
```

## Immutable data structures

`SharedVector` and `AtomicSharedVector` behave like simple immutable data structures and
are good building blocks for creating more complicated ones.

```
use shared_vector::{SharedVector, rc_vector};
let mut a = rc_vector![1u32, 2, 3];

// `new_ref` (you can also use `clone`) creates a second reference to the same buffer.
// future mutations of `a` won't affect `b`.
let mut b = a.new_ref();
// Because both a and b point to the same buffer, the next mutation allocates a new
// copy under the hood.
a.push(4);
// Now that a and b are unique pointers to their own buffers, no allocation happens
// when pushing to them.
a.push(5);
b.push(6);

assert_eq!(a.as_slice(), &[1, 2, 3, 4, 5]);
assert_eq!(b.as_slice(), &[1, 2, 3, 6]);
```

Note that `SharedVector` is *not* a RRB vector implementation.

### ChunkVector

As a very light experiment towards making custom immutable data structures on top of shared vectors, there is a very simple chunked vector implementation in the examples folder.

<svg width="810" height="160" viewBox="0 0 214.31 42.33" xmlns:xlink="http://www.w3.org/1999/xlink" xmlns="http://www.w3.org/2000/svg"><defs><linearGradient id="a"><stop offset="0" stop-color="#141b62"/><stop offset="1" stop-color="#7a2c92"/></linearGradient><linearGradient id="b"><stop offset="0" stop-color="#491c9c"/><stop offset="1" stop-color="#d54b27"/></linearGradient><linearGradient xlink:href="#a" id="d" gradientUnits="userSpaceOnUse" x1="6.27" y1="34.86" x2="87.72" y2="13.24" gradientTransform="translate(-2.64 -18.48)"/><linearGradient xlink:href="#b" id="f" gradientUnits="userSpaceOnUse" gradientTransform="translate(-2.64 -18.48)" x1="6.27" y1="34.86" x2="87.72" y2="13.24"/><linearGradient xlink:href="#b" id="e" gradientUnits="userSpaceOnUse" gradientTransform="translate(-2.64 -18.48)" x1="6.27" y1="34.86" x2="87.72" y2="13.24"/><linearGradient xlink:href="#b" id="g" gradientUnits="userSpaceOnUse" gradientTransform="translate(-3.96 -19.01)" x1="6.27" y1="34.86" x2="87.72" y2="13.24"/><linearGradient xlink:href="#a" id="c" gradientUnits="userSpaceOnUse" gradientTransform="translate(-2.64 -18.48)" x1="6.27" y1="34.86" x2="87.72" y2="13.24"/></defs><g transform="translate(22.49 7.94)"><rect width="10.57" height="10.66" x="84.67" y="-5.29" ry="1.37" fill="#3dbdaa"/><circle cx="89.96" cy="5.29" r=".79" fill="#666"/><path d="M99.22 7.94c-3.97 0-9.26 0-9.26-2.65" fill="none" stroke="#999" stroke-width=".86" stroke-linecap="round"/><g transform="translate(96.57 -.04)"><rect width="68.79" height="10.58" x="2.65" y="2.68" ry="1.37" fill="url(#c)"/><rect width="15.35" height="9.51" x="3.18" y="3.21" ry=".9" fill="#78a2d4"/><rect width="9.26" height="9.51" x="19.85" y="3.2" ry=".9" fill="#8e9ddd"/><rect width="9.26" height="9.51" x="29.64" y="3.22" ry=".9" fill="#8e9ddd"/><rect width="9.26" height="9.51" x="39.43" y="3.22" ry=".9" fill="#8e9ddd"/><rect width="9.26" height="9.51" x="49.22" y="3.21" ry=".9" fill="#7785c1"/><circle cx="62.84" cy="7.97" r=".66" fill="#eaa577"/><circle cx="64.7" cy="7.97" r=".66" fill="#eaa577"/><circle cx="66.55" cy="7.97" r=".66" fill="#eaa577"/></g><circle cx="120.91" cy="12.7" r=".79" fill="#666"/><circle cx="130.7" cy="12.7" r=".79" fill="#666"/><circle cx="140.49" cy="12.7" r=".79" fill="#666"/><path d="M54.24 21.17c0-5.3 76.46 0 76.46-8.47" fill="none" stroke="#b3b3b3" stroke-width=".86" stroke-linecap="round"/><path d="M-15.88 21.17c0-6.62 136.8-1.86 136.8-8.47" fill="none" stroke="#b3b3b3" stroke-width=".86" stroke-linecap="round"/><rect width="10.57" height="10.66" x="-6.61" y="-5.37" ry="1.37" fill="#3dbdaa"/><rect width="10.57" height="10.66" x="-19.83" y="-5.33" ry="1.37" fill="#3dbdaa"/><circle cx="-14.55" cy="5.29" r=".79" fill="#666"/><circle cx="-1.32" cy="5.29" r=".79" fill="#666"/><path d="M7.94 7.94c-3.97 0-9.26 0-9.26-2.65m9.26 3.97c-3.97 0-22.5 0-22.5-3.97" fill="none" stroke="#999" stroke-width=".86" stroke-linecap="round"/><g transform="translate(5.29 -.04)"><rect width="68.79" height="10.58" x="2.65" y="2.68" ry="1.37" fill="url(#d)"/><rect width="15.35" height="9.51" x="3.18" y="3.21" ry=".9" fill="#78a2d4"/><rect width="9.26" height="9.51" x="19.85" y="3.2" ry=".9" fill="#8e9ddd"/><rect width="9.26" height="9.51" x="29.64" y="3.22" ry=".9" fill="#8e9ddd"/><rect width="9.26" height="9.51" x="39.43" y="3.22" ry=".9" fill="#7785c1"/><rect width="9.26" height="9.51" x="49.22" y="3.21" ry=".9" fill="#7785c1"/><circle cx="62.84" cy="7.97" r=".66" fill="#eaa577"/><circle cx="64.7" cy="7.97" r=".66" fill="#eaa577"/><circle cx="66.55" cy="7.97" r=".66" fill="#eaa577"/></g><circle cx="30.43" cy="12.7" r=".79" fill="#666"/><circle cx="39.69" cy="12.7" r=".79" fill="#666"/><path d="M-17.2 21.17c0-6.62 47.63-1.86 47.63-8.47m22.49 8.47c0-6.62-13.23-1.86-13.23-8.47" fill="none" stroke="#999" stroke-width=".86" stroke-linecap="round"/><g transform="translate(47.62 18.48)"><rect width="68.79" height="10.58" x="2.65" y="2.68" ry="1.37" fill="url(#e)"/><rect width="15.35" height="9.51" x="3.18" y="3.21" ry=".9" fill="#78a2d4"/><rect width="9.26" height="9.51" x="19.85" y="3.2" ry=".9" fill="#eaa577"/><rect width="9.26" height="9.51" x="29.64" y="3.22" ry=".9" fill="#eaa577"/><rect width="9.26" height="9.51" x="39.43" y="3.22" ry=".9" fill="#eaa577"/><rect width="9.26" height="9.51" x="49.22" y="3.21" ry=".9" fill="#eaa577"/><circle cx="62.84" cy="7.97" r=".66" fill="#eaa577"/><circle cx="64.7" cy="7.97" r=".66" fill="#eaa577"/><circle cx="66.55" cy="7.97" r=".66" fill="#eaa577"/></g><g transform="translate(-22.5 18.48)"><rect width="68.79" height="10.58" x="2.65" y="2.68" ry="1.37" fill="url(#f)"/><rect width="15.35" height="9.51" x="3.18" y="3.21" ry=".9" fill="#78a2d4"/><rect width="9.26" height="9.51" x="19.85" y="3.2" ry=".9" fill="#eaa577"/><rect width="9.26" height="9.51" x="29.64" y="3.22" ry=".9" fill="#eaa577"/><rect width="9.26" height="9.51" x="39.43" y="3.22" ry=".9" fill="#eaa577"/><rect width="9.26" height="9.51" x="49.22" y="3.21" ry=".9" fill="#eaa577"/><circle cx="62.84" cy="7.97" r=".66" fill="#eaa577"/><circle cx="64.7" cy="7.97" r=".66" fill="#eaa577"/><circle cx="66.55" cy="7.97" r=".66" fill="#eaa577"/></g><path d="M123.03 21.17c0-6.62 17.46-1.86 17.46-8.47" fill="none" stroke="#b3b3b3" stroke-width=".86" stroke-linecap="round"/><g transform="translate(119.06 19.01)"><rect width="68.79" height="10.58" x="1.33" y="2.15" ry="1.37" fill="url(#g)"/><rect width="15.35" height="9.51" x="1.86" y="2.68" ry=".9" fill="#78a2d4"/><rect width="9.26" height="9.51" x="18.53" y="2.68" ry=".9" fill="#eaa577"/><rect width="9.26" height="9.51" x="28.32" y="2.69" ry=".9" fill="#eaa577"/><rect width="9.26" height="9.51" x="38.11" y="2.69" ry=".9" fill="#eaa577"/><rect width="9.26" height="9.51" x="47.89" y="2.68" ry=".9" fill="#ca8659"/><circle cx="61.52" cy="7.45" r=".66" fill="#eaa577"/><circle cx="63.37" cy="7.45" r=".66" fill="#eaa577"/><circle cx="65.23" cy="7.45" r=".66" fill="#eaa577"/></g></g></svg>

Just like the vector types, "chunk vector" comes in three flavors: `SharedChunkVector`, `AtomicSharedChunkVector` and `UniqueChunkVector`.

They are internally represented as a reference counted table of reference counted memory blocks (or "chunks"). In the illustration above, two chunked vectors point to the same table, while another points to a different table but till shares some of the storage chunks. In practice the chunked vector types are very little more than `SharedVector<SharedVector<T>>`

# Limitiations

- These vector types can hold at most `u32::MAX` elements.
- I'm not proactively implementing all of `Vec`'s featureset because there are only so many hours in a day, if you are missing something, don't hesitate to submit a pull request.

# License

Licensed under either of:

- Apache License, Version 2.0 (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license (LICENSE-MIT or http://opensource.org/licenses/MIT)

# Contributions

See the [contribution guidelines](https://github.com/nical/shared_vector/blob/master/CONTRIBUTING.md).
