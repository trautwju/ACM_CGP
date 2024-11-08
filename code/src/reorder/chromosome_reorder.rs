use crate::global_params::CgpParameters as g_params;
use crate::reorder::node_reorder::NodeReorder;
use crate::utils::fitness_metrics;
use crate::utils::node_type::NodeType;
use ndarray::prelude::*;
use rand::prelude::IteratorRandom;
use rand::{thread_rng, Rng};
use std::collections::{HashMap, HashSet};
use std::fmt::{Display, Formatter};

#[derive(Clone)]
pub struct Chromosome {
    pub params: g_params,
    pub nodes_grid: Vec<NodeReorder>,
    pub output_node_ids: Vec<usize>,
    pub active_nodes: Option<Vec<usize>>,
}

impl Display for Chromosome {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        writeln!(f, "+++++++++++++++++ Chromosome +++++++++++")?;
        writeln!(f, "Nodes:")?;
        for node in &self.nodes_grid {
            write!(f, "{}", *node)?;
        }
        writeln!(f, "Active_nodes: {:?}", self.active_nodes)?;
        writeln!(f, "Output_nodes: {:?}", self.output_node_ids)
    }
}

impl Chromosome {
    pub fn new(params: g_params) -> Self {
        let mut nodes_grid: Vec<NodeReorder> = vec![];
        let mut output_node_ids: Vec<usize> = vec![];
        nodes_grid.reserve(params.nbr_inputs + params.graph_width + params.nbr_outputs);
        output_node_ids.reserve(params.nbr_outputs);

        // input nodes
        for position in 0..params.nbr_inputs {
            nodes_grid.push(NodeReorder::new(
                position,
                params.nbr_inputs,
                params.graph_width,
                NodeType::InputNode,
            ));
        }
        // computational nodes
        for position in params.nbr_inputs..(params.nbr_inputs + params.graph_width) {
            nodes_grid.push(NodeReorder::new(
                position,
                params.nbr_inputs,
                params.graph_width,
                NodeType::ComputationalNode,
            ));
        }
        // output nodes
        for position in (params.nbr_inputs + params.graph_width)
            ..(params.nbr_inputs + params.graph_width + params.nbr_outputs)
        {
            nodes_grid.push(NodeReorder::new(
                position,
                params.nbr_inputs,
                params.graph_width,
                NodeType::OutputNode,
            ));
        }

        for position in (params.nbr_inputs + params.graph_width)
            ..(params.nbr_inputs + params.graph_width + params.nbr_outputs)
        {
            output_node_ids.push(position);
        }

        Self {
            params,
            nodes_grid,
            output_node_ids,
            active_nodes: None,
        }
    }

    pub fn evaluate(&mut self, inputs: &Array2<bool>, labels: &Array2<bool>) -> f32 {
        // let active_nodes = self.get_active_nodes_id();
        // self.active_nodes = Some(self.get_active_nodes_id());
        self.get_active_nodes_id();

        let output_size = inputs.slice(s![.., 0]).len();
        let nbr_nodes = self.params.nbr_inputs + self.params.graph_width + self.params.nbr_outputs;
        let nbr_nodes = nbr_nodes as usize;
        let mut outputs = Array::from_elem((nbr_nodes, output_size), true);

        for node_id in self.active_nodes.as_ref().unwrap() {
            // let node_id = *_node_id as usize;
            let current_node: &NodeReorder = &self.nodes_grid[*node_id];

            match current_node.node_type {
                NodeType::InputNode => {
                    // get the input from index node_id
                    let slice = inputs.slice(s![.., *node_id]);
                    // copy the slice to index node_id in outputs
                    let mut output_slice = outputs.slice_mut(s![*node_id, ..]);
                    output_slice.assign(&slice);
                }
                NodeType::OutputNode => {
                    let con1 = current_node.connection1;
                    let (mut output_slice, prev_output) =
                        outputs.multi_slice_mut((s![*node_id, ..], s![con1, ..]));
                    output_slice.assign(&prev_output);
                }
                NodeType::ComputationalNode => {
                    let con1 = current_node.connection1;
                    let con2 = current_node.connection2;
                    let con1_slice = outputs.slice(s![con1, ..]);
                    let con2_slice = outputs.slice(s![con2, ..]);

                    let out = current_node.execute(&con1_slice, &con2_slice);
                    let mut output_slice = outputs.slice_mut(s![*node_id, ..]);
                    output_slice.assign(&out);
                }
            }
        }
        let output_start_id = (self.params.nbr_inputs + self.params.graph_width) as usize;
        let output_end_id =
            (self.params.nbr_inputs + self.params.graph_width + self.params.nbr_outputs) as usize;
        let outs = outputs.slice(s![output_start_id..output_end_id, ..]);
        let outs = outs.t();

        let fitness = fitness_metrics::fitness_boolean(&outs, &labels);

        return fitness;
    }

    pub fn get_active_nodes_id(&mut self) {
        let mut active: HashSet<usize, nohash_hasher::BuildNoHashHasher<usize>> =
            HashSet::default();
        active.reserve(self.params.nbr_inputs + self.params.graph_width + self.params.nbr_outputs);

        let mut to_visit: Vec<usize> = vec![];
        to_visit
            .reserve(self.params.nbr_inputs + self.params.graph_width + self.params.nbr_outputs);

        for output_node_id in &self.output_node_ids {
            active.insert(*output_node_id);
            to_visit.push(*output_node_id);
        }

        while let Some(current_node_id) = to_visit.pop() {
            let current_node: &NodeReorder = &self.nodes_grid[current_node_id];

            match current_node.node_type {
                NodeType::InputNode => continue,

                NodeType::ComputationalNode => {
                    let connection1 = current_node.connection1;
                    if !active.contains(&connection1) {
                        to_visit.push(connection1);
                        active.insert(connection1);
                    }

                    let connection2 = current_node.connection2;
                    if !active.contains(&connection2) {
                        to_visit.push(connection2);
                        active.insert(connection2);
                    }
                }

                NodeType::OutputNode => {
                    let connection1 = current_node.connection1;
                    if !active.contains(&connection1) {
                        to_visit.push(connection1);
                        active.insert(connection1);
                    }
                }
            }
        }

        let mut active: Vec<usize> = active.into_iter().collect();
        active.sort_unstable();
        self.active_nodes = Some(active);
    }

    pub fn mutate_single(&mut self) {
        self.reorder();

        let start_id = self.params.nbr_inputs;
        let end_id = self.params.nbr_inputs + self.params.graph_width + self.params.nbr_outputs;
        loop {
            let random_node_id = rand::thread_rng().gen_range(start_id..=end_id - 1) as usize;
            self.nodes_grid[random_node_id].mutate();

            if self
                .active_nodes
                .as_ref()
                .unwrap()
                .contains(&random_node_id)
            {
                break;
            }
        }
    }

    pub fn mutate_prob(&mut self, prob: f32) {
        self.reorder();

        let start_id = self.params.nbr_inputs;
        let end_id = self.params.nbr_inputs + self.params.graph_width + self.params.nbr_outputs;
        for node_id in start_id..end_id {
            let random_prob: f32 = rand::thread_rng().gen::<f32>();
            if random_prob < prob {
                self.nodes_grid[node_id].mutate();
            };
        }
    }

    fn determine_node_dependency(
        &self,
    ) -> HashMap<usize, Vec<usize>, nohash_hasher::BuildNoHashHasher<usize>> {
        // Get list with connections of each computational node.
        // Input nodes are already removed from this list
        let mut node_dependencies: HashMap<
            usize,
            Vec<usize>,
            nohash_hasher::BuildNoHashHasher<usize>,
        > = HashMap::default();

        // init hashmaps
        for node_index in self.params.nbr_inputs..self.params.nbr_inputs + self.params.graph_width {
            node_dependencies.insert(node_index, Vec::with_capacity(4));
        }

        // iterate through each computational node and get their connections
        for node_index in self.params.nbr_inputs..self.params.nbr_inputs + self.params.graph_width {
            let current_node = &self.nodes_grid[node_index];
            let con1 = current_node.connection1;
            let con2 = current_node.connection2;

            // if con1 or con2 not input nodes; add them
            if !(0..self.params.nbr_inputs).contains(&con1) {
                node_dependencies.get_mut(&node_index).unwrap().push(con1)
            }

            if !(0..self.params.nbr_inputs).contains(&con2) {
                node_dependencies.get_mut(&node_index).unwrap().push(con2)
            }
        }

        return node_dependencies;
    }

    fn get_addable(
        &self,
        node_dependencies: &mut HashMap<usize, Vec<usize>, nohash_hasher::BuildNoHashHasher<usize>>,
        addable: &mut Vec<usize>,
    ) {
        // let mut addable: Vec<usize>;

        // get all empty node_ids -> get all nodes which link condition is satisfied
        let temp_addable: Vec<usize> = node_dependencies
            .iter()
            .filter(|(_, y)| y.is_empty())
            .map(|(&x, _)| x)
            .collect();
        if !temp_addable.is_empty() {
            addable.extend(temp_addable);
        }

        node_dependencies.retain(|_, v| *v != []);
    }

    fn update_node_index(
        &mut self,
        location_mapping: &HashMap<usize, usize, nohash_hasher::BuildNoHashHasher<usize>>,
    ) {
        // Nodes are not swapped because that could destroy ordering
        // Instead, create a new node_list by cloning the old one
        let mut new_nodes_grid: Vec<NodeReorder> = self.nodes_grid.clone();
        // Input and output nodes are ignored, as they do not change
        // Insert computational nodes
        for (old_position, new_position) in location_mapping.iter() {
            let mut node = self.nodes_grid[*old_position].clone();
            node.set_new_position(*new_position, false);

            new_nodes_grid[*new_position] = node;
        }

        self.nodes_grid = new_nodes_grid;
    }

    fn update_node_connections(
        &mut self,
        location_mapping: &HashMap<usize, usize, nohash_hasher::BuildNoHashHasher<usize>>,
    ) {
        let total_nbr_nodes =
            self.params.nbr_inputs + self.params.graph_width + self.params.nbr_outputs;

        for node_id in self.params.nbr_inputs..total_nbr_nodes {
            let node = &mut self.nodes_grid[node_id];

            if location_mapping.get(&node.connection1).is_some() {
                node.set_connection1(*location_mapping.get(&node.connection1).unwrap());
            }
            if location_mapping.get(&node.connection2).is_some() {
                node.set_connection2(*location_mapping.get(&node.connection2).unwrap());
            }
        }
    }

    pub fn reorder(&mut self) {
        let total_nbr_nodes =
            self.params.nbr_inputs + self.params.graph_width + self.params.nbr_outputs;

        // vars with paper names and description
        // index ; new location of the node in the grid
        let mut new_node_index = self.params.nbr_inputs;
        // new_loc ; save the old location and the position of the new location
        let mut changed_locations: HashMap<usize, usize, nohash_hasher::BuildNoHashHasher<usize>> =
            HashMap::default();
        changed_locations.reserve(total_nbr_nodes);
        // - ; which nodes are already visited?
        let mut used_node_indices: Vec<usize> = Vec::with_capacity(total_nbr_nodes);
        // input_locations (?) ; get the index of each node with its connection-id. input-node-id is removed
        let mut node_dependencies: HashMap<
            usize,
            Vec<usize>,
            nohash_hasher::BuildNoHashHasher<usize>,
        > = self.determine_node_dependency();
        node_dependencies.reserve(total_nbr_nodes);
        // addable ; addable nodes
        let mut addable: Vec<usize> = vec![];
        // let mut addable: Vec<usize> = self.get_addable(&mut node_dependencies, None);
        self.get_addable(&mut node_dependencies, &mut addable);

        while addable.len() > 0 {
            // current_node_id is also the position of the node in the grid
            // let (idx, current_node_id) = addable
            //     .iter()
            //     .enumerate()
            //     .choose(&mut thread_rng())
            //     .unwrap();
            // addable.remove(idx);
            let i = (0..addable.len()).choose(&mut thread_rng()).unwrap();
            let current_node_id = addable.swap_remove(i);

            // map old location to new location
            // changed_locations.insert(*current_node_id, new_node_index);
            changed_locations.insert(current_node_id, new_node_index);

            // update dependencies
            // check if the current node id exists in each dependency entry. if exists, remove
            for val in node_dependencies.values_mut() {
                // val.retain(|&x| x != *current_node_id);
                val.retain(|&x| x != current_node_id);
            }
            // update params
            new_node_index += 1;
            // used_node_indices.push(*current_node_id);
            used_node_indices.push(current_node_id);
            // addable = self.get_addable(&mut node_dependencies, Some(&used_node_indices));
            self.get_addable(&mut node_dependencies, &mut addable);
        }

        self.update_node_index(&changed_locations);
        self.update_node_connections(&changed_locations);

        assert_eq!(changed_locations.len(), self.params.graph_width);
        assert_eq!(used_node_indices.len(), self.params.graph_width);
    }
}
