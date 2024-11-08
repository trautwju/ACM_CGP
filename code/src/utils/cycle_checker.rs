use std::collections::HashSet;

#[derive(Clone)]
pub struct CGPEdges {
    edges: Vec<Vec<usize>>,
}

impl CGPEdges {
    pub fn new(nbr_nodes: usize) -> Self {
        let mut edges: Vec<Vec<usize>> = vec![];
        for _ in 0..nbr_nodes {
            edges.push(Vec::with_capacity(32));
        }

        Self { edges }
    }

    pub fn add_edge(&mut self, node_id: usize, prev_node_id: usize) {
        self.edges[node_id].push(prev_node_id);
    }

    pub fn remove_edge(&mut self, node_id: usize, prev_node_id: usize) {
        let index = self.edges[node_id]
            .iter()
            .position(|x| *x == prev_node_id)
            .unwrap();
        self.edges[node_id].remove(index);
    }

    /// Returns true if prev_node_id -> node_id would lead to cycle
    pub fn check_cycle(&mut self, node_id: usize, prev_node_id: usize) -> bool {
        let mut to_check: Vec<usize> = Vec::with_capacity(64);
        let mut checked: HashSet<usize, nohash_hasher::BuildNoHashHasher<usize>> =
            HashSet::default();

        to_check.extend(&self.edges[prev_node_id]);
        checked.extend(&self.edges[prev_node_id]);

        while let Some(checking) = to_check.pop() {
            if checking == node_id {
                return true;
            }

            for new_edge in &self.edges[checking] {
                if !checked.contains(new_edge) {
                    to_check.push(*new_edge);
                    checked.insert(*new_edge);
                }
            }
        }
        return false;
    }
}
