#![forbid(unsafe_code)]

//! Graphiti is a crate for constructing and analyzing graphs.
//! 
//! Graphiti was inspired by [Nauty and Traces](https://pallini.di.uniroma1.it/), a collection of C programs for working with graphs.
//! These work great as executables, but leave something to be desired as a library.
//! For instance, many of their graph generation functions require an output file parameter, where they simply dump all the results;
//! in Graphiti, this will instead be implemented as an iterator.
//! Graphiti aims to eventually reimplement many of the same core features as a rust library, which can then be easily included into any program.
//! 
//! **Note: this crate is *not* a rust port of Nauty and Traces.**
//! If you would rather have rust bindings to the full Nauty and Traces library, try [nauty-Traces-sys](https://crates.io/crates/nauty-Traces-sys).

use std::io::Write;

use std::collections::{BTreeMap, BTreeSet};
use std::convert::TryFrom;
use std::ops::{Add, Index, Range};
use std::iter::{Fuse, FusedIterator};
use std::{iter, vec, mem, fmt, fmt::Debug};
use std::rc::Rc;
use std::cmp::Ordering;

use std::sync::Mutex;

use itertools::{Itertools, Combinations};
use superslice::Ext;

#[macro_use]
extern crate lazy_static;

#[cfg(debug)]
fn is_sorted<T: Ord>(arr: &[T]) -> bool {
    arr.windows(2).all(|w| w[0] <= w[1])
}

/// Trait that must be implemented for any weight type.
/// 
/// The inherited trait bounds denote the raw requirements, and the implementation controls extra details such as weight aggregation for computing path costs.
/// This is implemented for all the integer primitive types; notably, it is not implemented for `f32` and `f64` because they are not [`Ord`].
/// For `()`, this is implemented to perform aggregation as `usize` with a meta-weight of `1`;
/// this allows space to be saved in an "unweighted" graph, which really just uses `()` as the weight type.
/// 
/// # Requirements
/// 
/// Where `a` and `b` are instances of `T: Weight`, it is always true that `a.cmp(&b) == a.into_meta().cmp(&b.into_meta())`.
/// That is, the meta-weight conversion must not change the ordering relation.
/// For instance, it would not be allowed to implement `Weight` with `Meta` being some type with less precision.
pub trait Weight: Ord + Clone + Debug {
    /// The meta-weight type to use for computing aggregated weight sums.
    type Meta: Ord + Clone + Add;
    /// Converts a weight object into its meta-weight value, which can be aggregated.
    fn to_meta(&self) -> Self::Meta;
}
macro_rules! impl_weight {
    ($($t:ty),+) => {$(
        impl Weight for $t {
            type Meta = $t;
            fn to_meta(&self) -> $t { *self }
        }
    )+}
}
impl_weight! { i8, i16, i32, i64, i128, u8, u16, u32, u64, u128, isize, usize }
impl Weight for () {
    type Meta = usize;
    fn to_meta(&self) -> usize { 1 }
}

/// Error type for constructing a [`Morphism`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MorphismError {
    /// The input had out of bounds values.
    /// This could also be consider a `NotSurjective` error, but is differentiated for convenience.
    OutOfBounds,
    /// The input had duplicate values.
    NotInjective,
    /// The input did not cover all values.
    NotSurjective,
}
/// Error type for applying or composing a [`Morphism`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MorphismUsageError {
    /// The items were defined for different domains (different number of vertices).
    DifferentOrder,
}

/// A morphism (relabeling rule) on a labeled graph.
/// 
/// A morphism is a vertex label permutation, which in this library is formally defined as a bijection `Zn -> Zn` where `n` is the number of vertices.
/// In essence, vertex `i` will be mapped to `morphism[i]`.
/// 
/// Arbitrary morphisms can be constructed directly via [`Morphism::try_from`], which checks the bijection constraints.
/// 
/// [`Eq`] is implemented to check that both the order (number of vertices) and mappings are equivalent.
#[derive(Clone, PartialEq, Eq)]
pub struct Morphism(Vec<usize>);
impl Morphism {
    /// Returns the number of vertices the morphism is defined over, being `0..order()`.
    pub fn order(&self) -> usize {
        self.0.len()
    }
}
impl Morphism {
    /// Iterates over all the output values of the morphism function.
    /// To get the input values as well, wrap it in `enumerate()`.
    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.0.iter().copied()
    }
}
impl Index<usize> for Morphism {
    type Output = usize;
    fn index(&self, index: usize) -> &usize {
        &self.0[index]
    }
}
impl TryFrom<Vec<usize>> for Morphism {
    type Error = MorphismError;
    fn try_from(val: Vec<usize>) -> Result<Self, Self::Error> {
        let mut flags = vec![false; val.len()];
        for &v in val.iter() {
            if v >= val.len() { return Err(MorphismError::OutOfBounds); }
            if flags[v] { return Err(MorphismError::NotInjective); }
            flags[v] = true;
        }
        if flags.into_iter().any(|v| !v) { return Err(MorphismError::NotSurjective); }
        Ok(Self(val))
    }
}
impl fmt::Debug for Morphism {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

/// Error type for adding an edge to a directed multigraph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MultiDigraphEdgeError {
    /// One of the vertices was invalid (out of bounds).
    OutOfBounds,
}
/// Error type for adding an edge to a (simple) graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphEdgeError {
    /// One of the vertices was invalid (out of bounds).
    OutOfBounds,
    /// The requested edge is forms a loop.
    FormsLoop,
    /// The edge already existed (perhaps with a different weight).
    AlreadyExisted,
}

/// Represents the adjacency list of a vertex in a graph.
/// 
/// This is effectively a collection of edges denoted by a target vertex `usize` and an edge weight `T`.
/// Note that depending on the graph, this could include duplicate edges or loops.
#[derive(PartialEq, Eq, Clone)]
pub struct Adjacency<'a, T: Weight>(&'a [(usize, T)]);
impl<'a, T: Weight> Adjacency<'a, T> {
    /// Returns all the outgoing edges from this vertex.
    /// This collection is sorted first by the target vertex, and then by ascending weight.
    pub fn edges(&self) -> impl Iterator<Item = (usize, &T)> {
        self.0.iter().map(|e| (e.0, &e.1))
    }
    /// Returns the subset of outgoing edges which lead to the specified destination vertex.
    /// This collection is sorted by ascending weight; thus, `edges_to(dest).next()` will be the cheapest edge to `dest` (or `None` if no edge exists).
    pub fn edges_to(&self, dest: usize) -> impl Iterator<Item = (usize, &T)> {
        self.0[self.0.equal_range_by(|a| a.0.cmp(&dest))].iter().map(|e| (e.0, &e.1))
    }
}
impl<'a, T: Weight + Debug> Debug for Adjacency<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.0 {
            [a, b @ ..] => {
                write!(f, "{{{}:{:?}", a.0, a.1)?;
                for e in b {
                    write!(f, ", {}:{:?}", e.0, e.1)?;
                }
                write!(f, "}}")
            }
            [] => write!(f, "{{}}"),
        }
    }
}

type PartitionBucketLevel0<T> = Rc<Vec<Vec<Vec<(bool, Vec<T>)>>>>;
type PartitionBucket<T> = PartitionBucketLevel0<T>;//Vec<Vec<PartitionBucketLevel0<T>>>;
type Partition<T> = BTreeMap<PartitionBucket<T>, Vec<usize>>;

/// A directed multigraph.
/// 
/// This is effectively a superset of all the graph types available in Graphiti, allowing directed edges, multiple edges, and loops.
/// 
/// This type implements [`Eq`] to test equivalence as a labeled graph (not isomorpic equivalence).
/// If you want to compare graphs based on isomorphism, use [`Self::get_isomorphism`].
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone)]
pub struct MultiDigraph<T: Weight>(Vec<Vec<(usize, T)>>);
impl<T: Weight> MultiDigraph<T> {
    /// Returns a new, empty graph.
    pub fn new() -> Self {
        Self(vec![])
    }
    /// Returns a disconnected graph with the specified number of vertices (labeled `0..order`).
    /// This is equivalent to repeatedly calling [`Self::add_vertex`].
    pub fn with_order(order: usize) -> Self {
        let mut verts = Vec::with_capacity(order);
        for _ in 0..order { verts.push(Default::default()); }
        Self(verts)
    }
    /// Returns the graph order, the number of vertices.
    pub fn order(&self) -> usize {
        self.0.len()
    }
    /// Checks if this is an empty graph (no vertices).
    pub fn is_empty(&self) -> bool {
        self.order() == 0
    }
    /// Removes all vertices (and edges) from the graph.
    pub fn clear(&mut self) {
        self.0.clear();
    }
    /// Removes all edges from the graph, but leaves the vertices.
    pub fn clear_edges(&mut self) {
        for n in self.0.iter_mut() {
            n.clear();
        }
    }

    /// Adds a new (disconnected) vertex to the graph and returns its index (used for other operations).
    /// The returned indices simply start at 0 and increment by 1; they are only returned for convenience.
    pub fn add_vertex(&mut self) -> usize {
        self.0.push(Default::default());
        self.0.len() - 1
    }
    /// Adds a directed edge `a -> b` with the specified weight.
    /// Note that this will create a duplicate (multi) edge if such an edge already existed.
    /// Fails if `a` or `b` is not a valid vertex index.
    pub fn add_directed_edge(&mut self, a: usize, b: usize, weight: T) -> Result<(), MultiDigraphEdgeError> {
        if a >= self.order() || b >= self.order() { return Err(MultiDigraphEdgeError::OutOfBounds); }
        let val = (b, weight);
        let adj = &mut self.0[a];
        let pos = adj.upper_bound(&val);
        Ok(adj.insert(pos, val))
    }
    /// Adds an undirected edge `a <-> b` with the specified weight.
    /// If `a != b`, this is equivalent to creating two directed edges.
    /// If `a == b`, this is equivalent to creating one directed edge.
    /// Note that this will create a duplicate edge if such an edge already existed.
    /// Fails if `a` or `b` is not a valid vertex index.
    pub fn add_undirected_edge(&mut self, a: usize, b: usize, weight: T) -> Result<(), MultiDigraphEdgeError> {
        if a != b { self.add_directed_edge(b, a, weight.clone())?; }
        self.add_directed_edge(a, b, weight)
    }

    /// Gets the adjacency list for the specified vertex.
    /// Returns `None` if the vertex index is invalid.
    pub fn adjacency(&self, vert: usize) -> Option<Adjacency<T>> {
        self.0.get(vert).map(|e| Adjacency(e))
    }
    /// Iterates through all the vertices in order, as if by calling [`Self::adjacency`] repeatedly.
    /// To get the vertex index as well, you can wrap this in `enumerate()`.
    pub fn adjacencies(&self) -> impl Iterator<Item = Adjacency<T>> {
        self.0.iter().map(|e| Adjacency(e))
    }

    /// Returns a [`Shells`] iterator object centered on the given vertex.
    /// If the vertex is invalid (out of bounds), returns `None`.
    pub fn shells(&self, vert: usize) -> Option<Shells<T>> {
        if vert >= self.order() { return None; }
        let mut ball = vec![false; self.order()];
        ball[vert] = true;
        let edge = ball.clone();
        Some(Shells { g: self, first: Some(vert), ball, edge })
    }

    /// Applies the morphism to the current graph.
    /// Fails if the graph and morphism have different orders.
    pub fn apply_morphism(&mut self, morphism: &Morphism) -> Result<(), MorphismUsageError> {
        if morphism.order() != self.order() { return Err(MorphismUsageError::DifferentOrder); }
        let old = mem::take(&mut self.0);
        self.0.reserve(old.len());
        for _ in 0..old.len() { self.0.push(Default::default()); }
        for (v, new) in morphism.iter().zip(old) {
            self.0[v] = new;
        }
        for n in self.0.iter_mut() {
            for v in n.iter_mut() {
                v.0 = morphism[v.0];
            }
            n.sort();
        }
        Ok(())
    }

    /// Creates a sorted partitioning table: equiv class --> unsorted vertices
    fn partition(&self) -> Partition<T> {
        let all_shells: Vec<_> = (0..self.order()).map(|v| self.shells(v).unwrap().collect::<Vec<_>>()).collect();
        
        let all_keys_level_0: Vec<PartitionBucketLevel0<T>> = all_shells.iter().enumerate().map(|(v, shells)| {
            Rc::new(shells.iter().map(|shell| { // keys shall be tiered by shell expansion rings
                let mut d: Vec<_> = shell.iter().map(|&v| { // each will consist of one value per vertex in the shell
                    #[cfg(debug)] // it complains without this because reasons
                    debug_assert!(is_sorted(&self.0[v].0)); // we need these to be sorted before we group_by (should already be sorted)

                    let mut w: Vec<_> = (&self.0[v].iter().group_by(|e| e.0)).into_iter().map(|(g, s)| { // for each vertex we look at its edges, grouped into equiv classes by label
                        let w: Vec<_> = s.map(|e| e.1.clone()).collect(); // we keep track of the list of weights
                        
                        #[cfg(debug)] // it complains without this because reasons
                        debug_assert!(is_sorted(&w)); // sort to be label invariant (should already be sorted)
                        
                        (g == v, w) // we record this, as well as a flag denoting if this edge equiv class represents loops, which is label-invariant
                    }).collect();
                    w.sort(); // sort to be label invariant
                    w
                }).collect();
                d.sort(); // sort to be label invariant
                d
            }).collect())
        }).collect();

        let all_keys: Vec<PartitionBucket<T>> = all_shells.iter().enumerate().map(|(v, shells)| {
            // shells.iter().map(|shell| {
            //     let mut d: Vec<_> = shell.iter().map(|&v| {
            //         all_keys_level_0[v].clone()
            //     }).collect();
            //     d.sort(); // sort to be label invariant
            //     d
            // }).collect()
            all_keys_level_0[v].clone()
        }).collect();

        let mut partition: Partition<T> = Default::default();
        for (v, key) in all_keys.into_iter().enumerate() {
            partition.entry(key).or_default().push(v);
        }
        partition
    }
    fn get_isomorphism_partitioned(&self, other: &Self, self_part: &Partition<T>, other_part: &Partition<T>) -> Option<Morphism> {
        if self_part.len() != other_part.len() { return None; } // if we have different number of buckets, definitely not iso
        
        let mut morphism_src = Vec::with_capacity(self_part.len());
        let mut morphism_dst = Vec::with_capacity(self_part.len());
        for (a, b) in self_part.iter().zip(other_part) {
            if a.0 != b.0 { return None; }             // if buckets are different, definitely not iso
            if a.1.len() != b.1.len() { return None; } // if equiv classes are not same size, definitely not iso
            let k = b.1.len();
            morphism_src.push(a.1);
            morphism_dst.push(b.1.into_iter().permutations(k));
        }

        let mut morphism = Morphism(vec![0; self.order()]); // not yet a valid morphism
        for morphism_pieces in morphism_dst.into_iter().multi_cartesian_product() {
            debug_assert_eq!(morphism_src.len(), morphism_pieces.len());
            for (src, dst) in morphism_src.iter().zip(morphism_pieces) {
                debug_assert_eq!(src.len(), dst.len());
                for (&a, &b) in src.iter().zip(dst) {
                    morphism.0[a] = b;
                }
            }
            let mut g = self.clone();
            g.apply_morphism(&morphism).unwrap();
            if g == *other { return Some(morphism); }
        }

        None // failed to find an isomorphism
    }
    /// Checks if this graph is isomorphic to another graph.
    /// If they are isomorphic, returns a morphism such that applying it to this graph yields the other graph.
    /// If they are not isomorphic, returns `None`.
    pub fn get_isomorphism(&self, other: &Self) -> Option<Morphism> {
        if self.order() != other.order() { return None; } // if they have different vertex counts, definitely not iso

        let self_part = self.partition();
        let other_part = other.partition();
        self.get_isomorphism_partitioned(other, &self_part, &other_part)
    }

    /// Flips all the (directed) edges in the graph.
    pub fn transpose(&mut self) {
        let old = mem::take(&mut self.0);
        self.0.reserve(old.len());
        for _ in 0..old.len() { self.0.push(Default::default()); }
        
        for (a, adj) in old.into_iter().enumerate() {
            for (b, w) in adj {
                self.0[b].push((a, w)); // reverse all the edges
            }
        }
        for n in self.0.iter_mut() {
            n.sort(); // then sort the neighborhoods (required)
        }
    }

    /// Checks if the graph is strongly connected.
    /// Specifically, returns true if and only if every distinct vertex pair has a (directed) path.
    /// Thus, any empty or singleton graph is considered connected.
    pub fn is_strongly_connected(&self) -> bool {
        if self.is_empty() { return true; } // if the graph is empty, it is connected
        if self.shells(0).unwrap().map(|s| s.len()).sum::<usize>() != self.order() { return false; } // make sure we can get from 0 to anywhere
        let transpose = { let mut t = self.clone(); t.transpose(); t };
        transpose.shells(0).unwrap().map(|s| s.len()).sum::<usize>() == self.order() // make sure we can get from anywhere to 0
    }
}
impl<T: Weight> Default for MultiDigraph<T> {
    fn default() -> Self {
        Self::new()
    }
}
impl<T: Weight + Debug> Debug for MultiDigraph<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut adj = self.adjacencies();
        match adj.next() {
            None => write!(f, "[]"),
            Some(first) => {
                write!(f, "[{:?}", first)?;
                for n in adj {
                    write!(f, ", {:?}", n)?;
                }
                write!(f, "]")
            }
        }
    }
}

/// A breadth-first search iterator type.
/// 
/// This is an object which iterates over all the shells centered on a given vertex.
/// Where `s` is the `i`th result (starting at 0), `s` is the sorted list of vertices which are at distance `i` from the center (by edge count, not by weight).
/// Note that this iterator will always yield at least one item, because the center is at distance `0` from itself.
/// Additionally, this iterator will never yield an empty array.
pub struct Shells<'a, T: Weight> {
    g: &'a MultiDigraph<T>,
    first: Option<usize>,
    ball: Vec<bool>,
    edge: Vec<bool>,
}
impl<'a, T: Weight> Iterator for Shells<'a, T> {
    type Item = Vec<usize>;
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(first) = self.first {
            self.first = None;
            return Some(vec![first]);
        }

        let mut next_edge = vec![false; self.g.order()];
        for (v, x) in self.edge.iter_mut().enumerate() {
            if !*x { continue }
            *x = false;
            for (u, _) in self.g.0[v].iter() {
                if !self.ball[*u] {
                    next_edge[*u] = true;
                    self.ball[*u] = true;
                }
            }
        }
        if !next_edge.iter().any(|x| *x) { return None; }
        let res = next_edge.iter().enumerate().filter(|(_, x)| **x).map(|(i, _)| i).collect();
        self.edge = next_edge;
        Some(res)
    }
}
impl<'a, T: Weight> FusedIterator for Shells<'a, T> {}

#[test]
fn test_morphism() {
    let mut g = MultiDigraph::with_order(5);
    assert_eq!(g.order(), 5);
    assert_eq!(g, MultiDigraph::with_order(5));
    assert_ne!(g, MultiDigraph::with_order(4));
    for &(a, b) in &[(0, 1), (0, 2), (1, 2), (1, 4), (2, 3), (3, 4)] {
        g.add_undirected_edge(a, b, ()).unwrap();
    }
    assert_ne!(g, MultiDigraph::with_order(5));
    let old_g = g.clone();
    assert_eq!(g, old_g);

    let mut g2 = MultiDigraph::with_order(5);
    for &(a, b) in &[(2, 1), (2, 4), (1, 4), (0, 1), (3, 4), (0, 3)] {
        g2.add_undirected_edge(a, b, ()).unwrap();
    }
    assert_ne!(g, g2);
    let morph = Morphism::try_from(vec![2, 1, 4, 3, 0]).unwrap();
    g.apply_morphism(&morph).unwrap();
    assert_eq!(g, g2);

    let morph = old_g.get_isomorphism(&g2).unwrap();
    let mut gg = old_g.clone();
    gg.apply_morphism(&morph).unwrap();
    assert_eq!(gg, g2);

    let mut g3 = Graph::with_order(9);
    for e in &[(0,1), (1,2), (3,4), (4,5), (6,7), (7,8), (0,3), (3,6), (1,4), (4,7), (2,5), (5,8)] {
        g3.add_edge(e.0, e.1, ()).unwrap()
    }
    {
        let mut gg = g3.clone();
        gg.apply_morphism(&Morphism::try_from(vec![1, 7, 0, 5, 8, 6, 3, 4, 2]).unwrap()).unwrap();
        assert_eq!(gg.adjacency(0).unwrap().edges().map(|(v, _)| v).collect::<Vec<_>>(), vec![6, 7]);
        assert_eq!(gg.adjacency(1).unwrap().edges().map(|(v, _)| v).collect::<Vec<_>>(), vec![5, 7]);
        assert_eq!(gg.adjacency(2).unwrap().edges().map(|(v, _)| v).collect::<Vec<_>>(), vec![4, 6]);
        assert_eq!(gg.adjacency(3).unwrap().edges().map(|(v, _)| v).collect::<Vec<_>>(), vec![4, 5]);
        assert_eq!(gg.adjacency(4).unwrap().edges().map(|(v, _)| v).collect::<Vec<_>>(), vec![2, 3, 8]);
        assert_eq!(gg.adjacency(5).unwrap().edges().map(|(v, _)| v).collect::<Vec<_>>(), vec![1, 3, 8]);
        assert_eq!(gg.adjacency(6).unwrap().edges().map(|(v, _)| v).collect::<Vec<_>>(), vec![0, 2, 8]);
        assert_eq!(gg.adjacency(7).unwrap().edges().map(|(v, _)| v).collect::<Vec<_>>(), vec![0, 1, 8]);
        assert_eq!(gg.adjacency(8).unwrap().edges().map(|(v, _)| v).collect::<Vec<_>>(), vec![4, 5, 6, 7]);
    }
}
#[test]
fn test_bigger_morphism() {
    let mut g = MultiDigraph::with_order(20);
    for &(a, b) in &[(0, 3), (0, 2), (2, 3), (3, 4), (1, 4), (4, 11), (11, 12), (12, 13), (13, 14), (14, 11), (7, 11), (7, 10), (7, 4),
                    (6, 3), (6, 7), (6, 15), (6, 16), (15, 17), (16, 18), (17, 19), (19, 18), (6, 9), (5, 9), (5, 8)]
    {
        g.add_undirected_edge(a, b, ()).unwrap();
    }

    let mut shells = g.shells(4).unwrap();
    assert_eq!(shells.next().unwrap(), &[4]);
    assert_eq!(shells.next().unwrap(), &[1, 3, 7, 11]);
    assert_eq!(shells.next().unwrap(), &[0, 2, 6, 10, 12, 14]);
    assert_eq!(shells.next().unwrap(), &[9, 13, 15, 16]);
    assert_eq!(shells.next().unwrap(), &[5, 17, 18]);
    assert_eq!(shells.next().unwrap(), &[8, 19]);
    for _ in 0..128 { assert_eq!(shells.next(), None); }

    let g2 = {
        let mut t = g.clone();
        t.apply_morphism(&Morphism::try_from(vec![8, 9, 7, 14, 12, 19, 18, 2, 13, 1, 6, 4, 5, 15, 3, 16, 11, 10, 17, 0]).unwrap()).unwrap();
        t
    };
    assert!(g2.is_strongly_connected());

    {
        let morph = g.get_isomorphism(&g2).unwrap();
        let mut gg = g.clone();
        gg.apply_morphism(&morph).unwrap();
        assert_eq!(gg, g2);
    }
}
#[test]
fn test_complete_weighted() {
    let mut g = MultiDigraph::with_order(20);
    for (w, e) in (0..20).combinations(2).enumerate() {
        g.add_undirected_edge(e[0], e[1], w).unwrap();
    }
    assert!(g.is_strongly_connected());

    let g2 = {
        let mut t = g.clone();
        t.apply_morphism(&Morphism::try_from(vec![8, 9, 7, 14, 12, 19, 18, 2, 13, 1, 6, 4, 5, 15, 3, 16, 11, 10, 17, 0]).unwrap()).unwrap();
        t
    };
    assert!(g2.is_strongly_connected());

    {
        let morph = g.get_isomorphism(&g2).unwrap();
        let mut gg = g.clone();
        gg.apply_morphism(&morph).unwrap();
        assert_eq!(gg, g2);
    }
}
#[test]
fn test_no_default() {
    #[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Debug)]
    struct NoDefault(i32);
    impl Weight for NoDefault {
        type Meta = i32;
        fn to_meta(&self) -> i32 { self.0 }
    }
    let g: MultiDigraph<NoDefault> = Default::default();
    assert!(g.is_empty());
}

/// A (simple) graph.
/// 
/// Simple graphs use undirected edges and forbid multi-edges and loops.
/// 
/// This type implements [`Eq`] to test equivalence as a labeled graph (not isomorpic equivalence).
/// If you want to compare graphs based on isomorphism, use [`Self::get_isomorphism`].
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone)]
pub struct Graph<T: Weight>(MultiDigraph<T>);
impl<T: Weight> Graph<T> {
    /// Returns a new, empty graph.
    pub fn new() -> Self { Self(Default::default()) }
    /// Returns a disconnected graph with the specified number of vertices (labeled `0..order`).
    /// This is equivalent to repeatedly calling [`Self::add_vertex`].
    pub fn with_order(order: usize) -> Self { Self(MultiDigraph::with_order(order)) }
    /// Returns the graph order, the number of vertices.
    pub fn order(&self) -> usize { self.0.order() }
    /// Checks if this is an empty graph (no vertices).
    pub fn is_empty(&self) -> bool { self.0.is_empty() }
    /// Removes all vertices (and edges) from the graph.
    pub fn clear(&mut self) { self.0.clear() }
    /// Removes all edges from the graph, but leaves the vertices.
    pub fn clear_edges(&mut self) { self.0.clear_edges() }

    /// Adds a new (disconnected) vertex to the graph and returns its index (used for other operations).
    /// The returned indices simply start at 0 and increment by 1; they are only returned for convenience.
    pub fn add_vertex(&mut self) -> usize { self.0.add_vertex() }
    /// Adds an (undirected) edges `a <-> b` with the specified weight.
    /// Fails if either vertex is invalid (out of bounds), if such an edge already exists, or if this would create a loop.
    pub fn add_edge(&mut self, a: usize, b: usize, weight: T) -> Result<(), GraphEdgeError> {
        if a >= self.order() || b >= self.order() { return Err(GraphEdgeError::OutOfBounds) }
        if a == b { return Err(GraphEdgeError::FormsLoop); }
        match self.0.0[a].binary_search_by(|v| v.0.cmp(&b)) {
            Ok(_) => return Err(GraphEdgeError::AlreadyExisted),
            Err(pos) => self.0.0[a].insert(pos, (b, weight.clone())),
        }
        match self.0.0[b].binary_search_by(|v| v.0.cmp(&a)) {
            Ok(_) => unreachable!(),
            Err(pos) => self.0.0[b].insert(pos, (a, weight)),
        }
        Ok(())
    }

    /// Gets the adjacency list for the specified vertex.
    /// Returns `None` if the vertex index is invalid.
    pub fn adjacency(&self, vert: usize) -> Option<Adjacency<T>> { self.0.adjacency(vert) }
    /// Iterates through all the vertices in order, as if by calling [`Self::adjacency`] repeatedly.
    /// To get the vertex index as well, you can wrap this in `enumerate()`.
    pub fn adjacencies(&self) -> impl Iterator<Item = Adjacency<T>> { self.0.adjacencies() }

    /// Returns a [`Shells`] iterator object centered on the given vertex.
    /// If the vertex is invalid (out of bounds), returns `None`.
    pub fn shells(&self, vert: usize) -> Option<Shells<T>> { self.0.shells(vert) }

    /// Applies the morphism to the current graph.
    /// Fails if the graph and morphism have different orders.
    pub fn apply_morphism(&mut self, morphism: &Morphism) -> Result<(), MorphismUsageError> { self.0.apply_morphism(morphism) }

    /// Checks if this graph is isomorphic to another graph.
    /// If they are isomorphic, returns a morphism such that applying it to this graph yields the other graph.
    /// If they are not isomorphic, returns `None`.
    pub fn get_isomorphism(&self, other: &Self) -> Option<Morphism> { self.0.get_isomorphism(&other.0) }

    /// Checks if the graph is connected.
    /// Specifically, returns true if and only if every distinct vertex pair has a path.
    /// Thus, any empty or singleton graph is considered connected.
    pub fn is_connected(&self) -> bool {
        if self.is_empty() { return true; } // if the graph is empty, it is connected
        self.shells(0).unwrap().map(|s| s.len()).sum::<usize>() == self.order() // make sure we can get from 0 to anywhere
    }

    /// Returns an iterator over all the unique graphs with the specified number of vertices and edges.
    /// This iterator will never return two graphs which are isomorphic.
    pub fn unique(order: usize, edges: usize, weight: T) -> UniqueGraphs<T> {
        let edges = (0..order).combinations(2).combinations(edges).fuse();
        UniqueGraphs { done: Default::default(), order, edges, weight }
    }
}
impl<T: Weight> Default for Graph<T> {
    fn default() -> Self {
        Self::new()
    }
}
impl<T: Weight + Debug> Debug for Graph<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

pub struct UniqueGraphs<T: Weight> {
    // classes: BTreeMap<Vec<(PartitionBucket<T>, usize)>, Vec<(Graph<T>, Partition<T>)>>,
    done: BTreeSet<Graph<T>>,
    order: usize,
    edges: Fuse<Combinations<Combinations<Range<usize>>>>,
    weight: T,
}
impl<T: Weight + Debug> Iterator for UniqueGraphs<T> {
    type Item = Graph<T>;
    fn next(&mut self) -> Option<Graph<T>> {
        let mut g: Graph<T> = Graph::with_order(self.order);
        'next_graph: while let Some(edges) = self.edges.next() {
            g.clear_edges();
            for e in edges {
                g.add_edge(e[0], e[1], self.weight.clone()).unwrap();
            }
            let canon = Graph(canonize(&g.0));
            if !self.done.contains(&canon) {
                self.done.insert(canon);
                return Some(g);
            }
            // let g_part = g.0.partition();
            // let buckets: Vec<_> = g_part.iter().map(|x| (x.0.clone(), x.1.len())).collect();
            // match self.classes.get_mut(&buckets) {
            //     None => { self.classes.insert(buckets, vec![(g.clone(), g_part)]); },
            //     Some(others) => {
            //         if others.len() != 1 { std::io::stdout().lock().write_all(format!("others: {}\n", others.len()).as_bytes()).unwrap(); }

            //         for (other, other_part) in others.iter() {
            //             if g.0.get_isomorphism_partitioned(&other.0, &g_part, other_part).is_some() {
            //                 continue 'next_graph;
            //             }
            //         }
            //         others.push((g.clone(), g_part));
            //     }
            // }
            //std::io::stdout().lock().write_all(format!("classes: {:?}\n", self.classes.len()).as_bytes()).unwrap();
            // return Some(g);
        }
        None
    }
}

// paper: http://www.math.unl.edu/~aradcliffe1/Papers/Canonical.pdf

#[derive(Clone, Default)]
struct OrderedPartition(Vec<BTreeSet<usize>>);
impl OrderedPartition {
    fn unit(n: usize) -> Self {
        Self(vec![(0..n).collect()])
    }
    // fn is_finer_than(&self, other: &Self) -> bool {
    //     if !self.0.iter().all(|v| other.0.iter().any(|w| v.is_subset(w))) { return false }

    //     for (i, vi) in self.0.iter().enumerate() {
    //         for (k, wk) in other.0.iter().enumerate() {
    //             if !vi.is_subset(wk) { continue }

    //             for vj in self.0[i..].iter() {
    //                 for (l, wl) in other.0.iter().enumerate() {
    //                     if vj.is_subset(wl) && k > l { return false }
    //                 }
    //             }
    //         }
    //     }

    //     true
    // }

    fn deg<T: Weight>(g: &MultiDigraph<T>, v: usize, part: &BTreeSet<usize>) -> Vec<T> {
        let mut res: Vec<_> = g.adjacency(v).unwrap().edges().filter_map(|(adj, w)| if part.contains(&adj) { Some(w.clone()) } else { None }).collect();
        res.sort();
        res
    }
    fn refine<T: Weight>(&self, g: &MultiDigraph<T>) -> OrderedPartition {
        let mut t = self.clone();
        loop {
            let mut shatter_target = None; // MUST BE LEXICOGRAPHICALLY SMALLEST (i,j) PAIR
            'shatter_target: for (i, vi) in t.0.iter().enumerate() {
                for (j, vj) in t.0.iter().enumerate() {
                    let mut iter = vi.iter();
                    while let Some(v) = iter.next().copied() {
                        for &w in iter.clone() {
                            if Self::deg(g, v, vj) != Self::deg(g, w, vj) {
                                shatter_target = Some((i, j));
                                break 'shatter_target;
                            }
                        }
                    }
                }
            }
            match shatter_target {
                None => return t,
                Some((i, j)) => {
                    let (vi, vj) = (&t.0[i], &t.0[j]);
                    let viarr: Vec<_> = vi.iter().copied().collect();
                    let degs: BTreeMap<usize, Vec<T>> = viarr.iter().map(|&v| (v, Self::deg(g, v, vj))).collect();

                    let shatter_info: BTreeSet<_> = viarr.iter().map(|v| (&degs[v], *v)).collect();
                    let mut shattered = vec![];
                    let mut x = vec![];
                    let mut current_deg = shatter_info.iter().next().unwrap().0;
                    for &(deg_vj, v) in shatter_info.iter() { // GROUPS MUST BE IN DEGREE ORDER
                        if deg_vj != current_deg {
                            shattered.push(mem::take(&mut x));
                            current_deg = deg_vj;
                        }
                        x.push(v);
                    }
                    if !x.is_empty() { shattered.push(x) }

                    let mut old_t = mem::take(&mut t).0.into_iter();
                    for _ in 0..i { t.0.push(old_t.next().unwrap()) }
                    for x in shattered { t.0.push(x.into_iter().collect()) }
                    t.0.extend(old_t.skip(1));
                }
            }
        }
    }
}

#[cfg(test)]
macro_rules! make_partition {
    ($($($e:literal)+)|*) => { OrderedPartition(vec![$(vec![$($e),+].into_iter().collect()),*]) }
}

#[test]
fn test_refine() {
    let mut g = Graph::with_order(9);
    for e in &[(0,1), (1,2), (3,4), (4,5), (6,7), (7,8), (0,3), (3,6), (1,4), (4,7), (2,5), (5,8)] {
        g.add_edge(e.0, e.1, ()).unwrap()
    }

    assert_eq!(OrderedPartition::unit(9).refine(&g.0).0, make_partition!(0 2 6 8|1 3 5 7|4).0);
    assert_eq!(make_partition!(0|2 6 8|1 3 5 7|4).refine(&g.0).0, make_partition!(0|2 6|8|5 7|1 3|4).0);
    assert_eq!(make_partition!(2|0 6 8|1 3 5 7|4).refine(&g.0).0, make_partition!(2|0 8|6|3 7|1 5|4).0);
    assert_eq!(make_partition!(8|0 2 6|1 3 5 7|4).refine(&g.0).0, make_partition!(8|2 6|0|1 3|5 7|4).0);
}

fn canonize<T: Weight>(g: &MultiDigraph<T>) -> MultiDigraph<T> {
    let mut queue = vec![OrderedPartition::unit(g.order()).refine(g)];
    let mut best = None;

    while let Some(p) = queue.pop() {
        let mut best_split = None;
        for (i, split) in p.0.iter().enumerate() {
            if split.len() == 1 { continue }
            match &best_split {
                None => best_split = Some((i, split)),
                Some((_, vi)) => if split.len() < vi.len() { best_split = Some((i, split)) },
            }
        }
        match best_split {
            Some((i, vi)) => for val in vi {
                let a = iter::once(*val).collect();
                let mut b = vi.clone();
                b.remove(val);

                let mut next_partition = OrderedPartition::default();
                let mut vals = p.0.iter().cloned();
                for _ in 0..i { next_partition.0.push(vals.next().unwrap()) }
                next_partition.0.push(a);
                next_partition.0.push(b);
                next_partition.0.extend(vals.skip(1));

                queue.push(next_partition.refine(g));
            }
            None => {
                let mut morph = Morphism(vec![0; g.order()]);
                for (i, vi) in p.0.iter().enumerate() {
                    morph.0[*vi.iter().next().unwrap()] = i;
                }

                let mut applied = g.clone();
                applied.apply_morphism(&morph).unwrap();

                match &best {
                    None => best = Some(applied),
                    Some(prev) => if applied > *prev { best = Some(applied) }
                }
                
            }
        }
    }

    best.unwrap()
}

#[test]
fn test_canon_disconnected() {
    let mut a = Graph::with_order(5);
    a.add_edge(0, 1, ()).unwrap();
    a.add_edge(0, 2, ()).unwrap();

    let mut b = Graph::with_order(5);
    b.add_edge(0, 1, ()).unwrap();
    b.add_edge(0, 3, ()).unwrap();

    assert_ne!(a, b);
    assert!(a.get_isomorphism(&b).is_some());
    let canon_a = Graph(canonize(&a.0));
    let canon_b = Graph(canonize(&b.0));
    assert_eq!(canon_a, canon_b);
}

#[test]
fn test_canonize() {
    let mut g = Graph::with_order(9);
    for e in &[(0,1), (1,2), (3,4), (4,5), (6,7), (7,8), (0,3), (3,6), (1,4), (4,7), (2,5), (5,8)] {
        g.add_edge(e.0, e.1, ()).unwrap()
    }

    let canon = Graph(canonize(&g.0));
    println!("\nCg  = {:?}\nCg = {:?}", g, canon);

    let canon_canon = Graph(canonize(&canon.0));
    println!("\nCCg = {:?}", canon_canon);
    assert_eq!(canon, canon_canon);

    let uniso: Vec<_> = Graph::unique(5, 2, ()).collect();
    for pair in uniso.iter().combinations(2) {
        if let Some(_) = pair[0].get_isomorphism(&pair[1]) {
            println!("g1: {:?}", &pair[0]);
            println!("g2: {:?}", &pair[1]);
            panic!();
        }
    }
    assert_eq!(uniso.len(), 2);
}

#[test]
fn test_unique_count() {
    assert_eq!(Graph::unique(1, 0, ()).count(), 1);
    assert_eq!(Graph::unique(5, 2, ()).count(), 2);
    assert_eq!(Graph::unique(5, 6, ()).count(), 6);
    // assert_eq!(Graph::unique(15, 6, ()).count(), 68);
}