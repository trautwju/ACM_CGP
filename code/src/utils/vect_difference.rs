use std::collections::HashSet;

pub fn vect_difference(v1: &Vec<usize>, v2: &Vec<usize>) -> Vec<usize> {
    let s1: HashSet<usize, nohash_hasher::BuildNoHashHasher<usize>> = v1.iter().cloned().collect();
    let s2: HashSet<usize, nohash_hasher::BuildNoHashHasher<usize>> = v2.iter().cloned().collect();
    (&s1 - &s2).iter().cloned().collect()
}
