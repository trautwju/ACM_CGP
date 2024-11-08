use crate::global_params::CgpParameters as g_params;
use crate::reorder::linspace::linspace;
use crate::reorder::node_reorder::NodeReorder;
use crate::utils::fitness_metrics;
use crate::utils::node_type::NodeType;
use crate::utils::vect_difference::vect_difference;
use ndarray::prelude::*;
use rand::Rng;
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

    pub fn reorder(&mut self) {
        // remove output nodes
        for output_node_id in ((self.params.nbr_inputs + self.params.graph_width)
            ..(self.params.nbr_inputs + self.params.graph_width + self.params.nbr_outputs))
            .rev()
        {
            let index = self
                .active_nodes
                .as_ref()
                .unwrap()
                .iter()
                .position(|x| *x == output_node_id)
                .unwrap();
            self.active_nodes.as_mut().unwrap().remove(index);
        }

        // remove input nodes, as only computational nodes are going to be swapped
        for input_node_id in (0..self.params.nbr_inputs).rev() {
            let index = self
                .active_nodes
                .as_ref()
                .unwrap()
                .iter()
                .position(|x| *x == input_node_id);
            if index.is_some() {
                self.active_nodes.as_mut().unwrap().remove(index.unwrap());
            }
        }

        if self.active_nodes.as_ref().unwrap().len() == 0 {
            return;
        }

        self.swap_nodes();
    }

    fn swap_nodes(&mut self) {
        let new_pos_active: Vec<usize> = linspace(
            self.params.nbr_inputs,
            self.params.nbr_inputs + self.params.graph_width - 1,
            self.active_nodes.as_ref().unwrap().len(),
        );

        let comp_nodes_ids: Vec<usize> =
            (self.params.nbr_inputs..self.params.nbr_inputs + self.params.graph_width).collect();
        let mut old_pos_inactive =
            vect_difference(&comp_nodes_ids, &self.active_nodes.as_ref().unwrap());
        let mut new_pos_inactive = vect_difference(&comp_nodes_ids, &new_pos_active);

        old_pos_inactive.sort_unstable();
        new_pos_inactive.sort_unstable();

        assert_eq!(
            self.active_nodes.as_ref().unwrap().len(),
            new_pos_active.len()
        );
        assert_eq!(
            old_pos_inactive.len(),
            new_pos_inactive.len(),
            "actives: \n{:?} \n{:?}",
            self.active_nodes.as_ref().unwrap(),
            new_pos_active
        );

        let mut swapped_pos_indices: HashMap<
            usize,
            usize,
            nohash_hasher::BuildNoHashHasher<usize>,
        > = HashMap::default();
        swapped_pos_indices
            .reserve(self.params.nbr_inputs + self.params.graph_width + self.params.nbr_outputs);

        // Nodes are not swapped because that could destroy ordering
        // Instead, create a new node_list by cloning the old one
        let mut new_nodes_grid: Vec<NodeReorder> = self.nodes_grid.clone();

        // Input nodes are ignored, as they do not change
        // Insert active computational nodes and change their position
        for (old_node_id, new_node_id) in self
            .active_nodes
            .as_ref()
            .unwrap()
            .iter()
            .zip(new_pos_active.iter())
        {
            let mut node = self.nodes_grid[*old_node_id].clone();
            node.set_new_position(*new_node_id, false);

            new_nodes_grid[*new_node_id] = node;

            swapped_pos_indices.insert(*old_node_id, *new_node_id);
        }

        // Now distribute all inactive nodes to the free indice
        for (old_node_id, new_node_id) in old_pos_inactive.iter().zip(new_pos_inactive.iter()) {
            assert!(!new_pos_active.contains(new_node_id));

            let mut node = self.nodes_grid[*old_node_id].clone();
            node.set_new_position(*new_node_id, true);
            new_nodes_grid[*new_node_id] = node;

            assert!(
                new_nodes_grid[*new_node_id].position > new_nodes_grid[*new_node_id].connection1,
                "assert 2 for node: {}",
                *new_node_id
            );
            assert!(
                new_nodes_grid[*new_node_id].position > new_nodes_grid[*new_node_id].connection2,
                "assert 3 for node: {}",
                *new_node_id
            );
        }

        // update connections of active nodes
        for node_id in &new_pos_active {
            Chromosome::update_connections(&mut new_nodes_grid, *node_id, &mut swapped_pos_indices);
        }

        // update connections for output nodes
        for node_id in (self.params.nbr_inputs + self.params.graph_width)
            ..(self.params.nbr_inputs + self.params.graph_width + self.params.nbr_outputs)
        {
            Chromosome::update_connections(&mut new_nodes_grid, node_id, &mut swapped_pos_indices);
        }

        self.nodes_grid = new_nodes_grid;

        self.get_active_nodes_id();
    }

    fn update_connections(
        new_nodes_grid: &mut Vec<NodeReorder>,
        node_id: usize,
        swapped_pos_indices: &mut HashMap<usize, usize, nohash_hasher::BuildNoHashHasher<usize>>,
    ) {
        let con1 = new_nodes_grid[node_id].connection1;
        let con2 = new_nodes_grid[node_id].connection2;

        new_nodes_grid[node_id].connection1 =
            *swapped_pos_indices.get(&con1).unwrap_or_else(|| &con1);
        new_nodes_grid[node_id].connection2 =
            *swapped_pos_indices.get(&con2).unwrap_or_else(|| &con2);
    }
}
