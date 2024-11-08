use crate::global_params::CgpParameters as g_params;
use crate::utils::fitness_metrics;
use crate::utils::node_type::NodeType;
use crate::vanilla_cgp::node::Node;
use ndarray::prelude::*;
use rand::Rng;
use std::collections::HashSet;
use std::fmt::{Display, Formatter};

#[derive(Clone)]
pub struct Chromosome {
    pub params: g_params,
    pub nodes_grid: Vec<Node>,
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
        let mut nodes_grid: Vec<Node> = vec![];
        let mut output_node_ids: Vec<usize> = vec![];
        nodes_grid.reserve(params.nbr_inputs + params.graph_width + params.nbr_outputs);
        output_node_ids.reserve(params.nbr_outputs);

        // input nodes
        for position in 0..params.nbr_inputs {
            nodes_grid.push(Node::new(
                position,
                params.nbr_inputs,
                params.graph_width,
                NodeType::InputNode,
            ));
        }
        // computational nodes
        for position in params.nbr_inputs..(params.nbr_inputs + params.graph_width) {
            nodes_grid.push(Node::new(
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
            nodes_grid.push(Node::new(
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
            let current_node: &Node = &self.nodes_grid[*node_id];

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
            let current_node: &Node = &self.nodes_grid[current_node_id];

            match current_node.node_type {
                NodeType::InputNode => continue,
                NodeType::ComputationalNode => {
                    let connection0 = current_node.connection1;
                    if !active.contains(&connection0) {
                        to_visit.push(connection0);
                        active.insert(connection0);
                    }
                    let connection1 = current_node.connection2;
                    if !active.contains(&connection1) {
                        to_visit.push(connection1);
                        active.insert(connection1);
                    }
                }
                NodeType::OutputNode => {
                    let connection0 = current_node.connection1;
                    if !active.contains(&connection0) {
                        to_visit.push(connection0);
                        active.insert(connection0);
                    }
                }
            }
        }
        let mut active: Vec<usize> = active.into_iter().collect();
        active.sort_unstable();
        self.active_nodes = Some(active);
    }

    pub fn mutate_single(&mut self) {
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
        let start_id = self.params.nbr_inputs;
        let end_id = self.params.nbr_inputs + self.params.graph_width + self.params.nbr_outputs;
        for node_id in start_id..end_id {
            let random_prob: f32 = rand::thread_rng().gen::<f32>();
            if random_prob < prob {
                self.nodes_grid[node_id].mutate();
            };
        }
    }
}
