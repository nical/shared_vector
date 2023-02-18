//! Shared and mutable vectors.
//!
//! # A word of caution
//! 
//! This crate is pretty light in test coverage right now, I would not recommand using it in anything security-critical.
//!
//! # Overview
//! 
//! This crate provides the following two types:
//! - `SharedVector<T>`/`AtomicSharedVector<T>`, an immutable reference counted vector (with an atomically
//!    reference counted variant).
//! - `UniqueVector<T>`, an unique vector type with an API similar to `std::Vec<T>`.
//!
//! Internally these types are a little different from the standard `Vec<T>`.
//! `SharedVector` and `AtomicSharedVector` hold a single pointer to a buffer containing:
//! - A 16 bytes header (plus possible padding for alignment),
//! - the contiguous sequence of items of type `T`.
//!
//! The header is where the vector's length, caoacity and reference count are stored.
//! 
//! ```ascii
//!  +---+ SharedVector (8 bytes on 64bit systems)
//!  |   |         +---+
//!  +---+  +------|   |
//!    |    |      +---+
//!    v    V
//!  +----------++----+----+----+----+----+----+----+----+
//!  |          ||    |    |    |    |    |    |    |    |
//!  +----------++----+----+----+----+----+----+----+----+
//!   \________/  \_____________________________________/
//!     Header                  Items
//! ```
//!
//! `UniqueVector`'s representation is closer to `Vec<T>`: it stores the length and capacity information
//! inline and only writes them into the header if/when converting into a shared vector.
//! The allocated buffer does leave room for the header so that converting to and from `SharedVector` is fast.
//!
//! ```ascii
//!            +---+---+---+
//!            |   |len|cap|   UniqueVector (16 bytes on 64bit systems)
//!            +---+---+---+
//!              |
//!              v
//!  +----------++----+----+----+----+----+----+----+----+
//!  |          ||    |    |    |    |    |    |    |    |
//!  +----------++----+----+----+----+----+----+----+----+
//!   \________/  \_____________________________________/
//!     Header                  Items
//!   (16 bytes)
//! ```
//!
//! This allows very cheap conversion between the two:
//! - shared to unique: a new allocation is made only if there are multiple handles to the same buffer (the reference
//!   count is greather than one).
//! - unique to shared: always fast since unique buffers are guaranteed to be sole owners of their buffer.
//!
//! # Use cases
//!
//! ## `Arc<Vec<T>>` without the indirection.
//!
//! A mutable vector can be be built using a Vec-style API, and then made immutable and reference counted for various
//! use case (easy multi-threading or simply shared ownership).
//!
//! Using the standard library one might be tempted to first build a `Vec<T>` and share it via `Arc<Vec<T>>`. This is
//! a fine approach at the cost of an extra pointer indirection that could be avoid in principle.
//! Another approach is to share it as an `Arc<[T]>` which removes the indirection at the cost of the need to copy
//! from the vector.
//!
//! Using this crate there is no extra indirection in the resulting shared vector nor any copy between the unique
//! and shared versions.
//!
//! ```
//! use immutable::UniqueVector;
//! let mut builder = UniqueVector::new();
//! builder.push(1u32);
//! builder.push(2);
//! builder.push(3);
//! // Make it reference counted, no allocation.
//! let mut shared = builder.into_shared();
//! // We can now create new references
//! let shared_2 = shared.new_ref();
//! let shared_3 = shared.new_ref();
//! ```
//!
//! ## Immutable data structures
//!
//! `SharedVector` and `AtomicSharedVector` behave like simple immutable data structures and
//! are good building blocks for creating more complicated ones.
//!
//! ```
//! use immutable::SharedVector;
//! let mut a = shared_vector![1u32, 2, 3];
//!
//! // `new_ref` (you can also use `clone`) creates a second reference to the same buffer.
//! // future mutations of `a` won't affect `b`.
//! let b = a.new_ref();
//! // Because both a and b point to the same buffer, the next mutation allocates a new
//! // copy under the hood.
//! a.push(4);
//! // Now that a and b are unique pointers to their own buffers, no allocation happens
//! // when pushing to them.
//! a.push(5);
//! b.push(6);
//!
//! assert_eq!(a.as_slice(), &[1, 2, 3, 4, 5]);
//! assert_eq!(b.as_slice(), &[1, 2, 3, 6]);
//! ```
//!
//! Note that `SharedVector` is *not* a RRB vector implementation.
//!
//! ### ChunkVector
//!
//! As a very light experiment towards making custom immutable data structures, this crate contains
//! a simple "chunked" vector implementation:
//!
//! ```ascii
//! SharedChunkVector<T>
//! +---+
//! | a |---------+
//! +---+         |
//!               V
//! +---+       +----------++---+---+---+
//! | b |------>|          ||   |   |   | Chunk table
//! +---+       +----------++---+---+---+
//!                           |   |   +----------------------// etc.
//!          +----------------+   +---------+
//!          V                              V
//!        +----------++---+---+---+---+  +----------++---+---+---+---+
//!        |          ||   |   |   |   |  |          ||   |   |   |   | Chunks
//!        +----------++---+---+---+---+  +----------++---+---+---+---+
//!          ^
//!          |
//!          +----------------+
//!                           |
//! +---+       +----------++---+---+---+
//! | c |------>|          ||   |   |   | chunk table
//! +---+       +----------++---+---+---+
//! ```
//!
//! Just like the vector types, "chunk vector" comes in three flavors: `SharedChunkVector`,
//! `AtomicSharedChunkVector` and `UniqueChunkVector`.
//! 
//! They are internally represented as a reference counted table of referenc counted memory
//! blocks (or "chunks"). In the illustration above, a and b both point to the same table,
//! while c points to another table but till shared one of the blocks.
//!
//! The chunked vector types are very little more than `SharedVector<SharedVector<T>>`
//!
//! # Limitiations
//!
//! - These vector types can hold at most `u32::MAX` elements.
//! - I'm not proactively implementing all of `Vec`'s featureset because there are only so many
//!   hours in a day, if you are missing something, don't hesitate to submit a pull request.
//!

mod raw;
mod vector;
mod chunked;

pub use vector::{SharedVector, AtomicSharedVector, UniqueVector};
pub use chunked::{SharedChunkVector, AtomicSharedChunkVector, UniqueChunkVector};
