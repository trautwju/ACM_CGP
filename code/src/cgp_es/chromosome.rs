use crate::global_params::CgpParameters as g_params;
use crate::utils::node_type::NodeType;
use crate::cgp_es::node::Node;
use rand::Rng;
use std::collections::HashSet;
use std::fmt::{Display, Formatter};
use std;
use std::collections::HashMap;

#[derive(Clone)]
pub struct Chromosome {
    pub params: g_params,
    pub nodes_grid: Vec<Node>,
    pub output_node_ids: Vec<usize>,
    pub active_nodes: Option<Vec<usize>>,
    pub mae:f32,
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
        let mae = 0.;

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
            mae,
        }
    }

    pub fn evaluate(&mut self, inputs: &Vec<Vec<f32>>, labels: &Vec<Vec<f32>>,) -> f32 {
        self.get_active_nodes_id();
        let mut outputs = HashMap::new();
        let mut final_outputs = HashMap::new();

        for node_id in self.active_nodes.as_ref().unwrap(){
            let current_node: &Node = &self.nodes_grid[*node_id];

            match current_node.node_type {
                NodeType::InputNode => {
                    // get the input from index node_id
                    let column: Vec<f32> = inputs
                        .iter()  // iterate over rows
                        .map(|x| x[*node_id]) // get the node_id-th element from each row
                        .collect();
                    outputs.insert((*node_id).to_string(), column,);
                }
                NodeType::OutputNode => {
                    let con1 = current_node.connection1.to_string();
                    let out = &outputs[&con1];
                    let mut slice = vec![0.; out.len()];
                    for i in 0..out.len(){
                        slice[i] = out[i];
                    }
                    final_outputs.insert((*node_id).to_string(), slice,);
                }
                NodeType::ComputationalNode => {
                    let con1 = current_node.connection1.to_string();
                    let con2 = current_node.connection2.to_string();
                    let con1_slice = &outputs[&con1];
                    let mut con2_slice = con1_slice;
                    if current_node.function_id < 4{
                        con2_slice = &outputs[&con2];
                    }
                    let out = current_node.execute(con1_slice, con2_slice);
                    outputs.insert((*node_id).to_string(), out,);
                }
            }
        }
        let output_start_id = (self.params.nbr_inputs + self.params.graph_width) as usize;
        let output_end_id =
            (self.params.nbr_inputs + self.params.graph_width + self.params.nbr_outputs) as usize;
        let mut outs:Vec<Vec<f32>> = vec![vec![0.; self.params.nbr_outputs];  inputs.len()];

        for i in output_start_id..output_end_id{
            let output_col = &final_outputs[&(i.to_string())];
            for j in 0..output_col.len(){
                outs[j][i-output_start_id] = output_col[j];
            }
        }
        let mut fitness = 0.;
        if labels[0].len() == 1{
            fitness += self.mse(&outs, &labels);
            self.mae = self.calc_mae(&outs, &labels);
        } else {
            fitness +=  self.mcc(&outs, &labels);
        }
        return fitness;
    }

    pub fn mse(&mut self, outputs: &Vec<Vec<f32>>, labels: &Vec<Vec<f32>>) -> f32{
        let mut mse = 0.;
        let j = outputs.len() as f32;
        for i in 0..outputs.len(){
            mse += (outputs[i][0] - labels[i][0]) * (outputs[i][0] - labels[i][0]);
        }
        mse /= j;
        if mse.is_nan(){
            mse = f32::INFINITY;
        }
        return mse;
    }

    pub fn calc_mae(&mut self, outputs: &Vec<Vec<f32>>, labels: &Vec<Vec<f32>>) -> f32{
        let mut mae = 0.;
        let j = outputs.len() as f32;
        for i in 0..outputs.len(){
            mae += (outputs[i][0] - labels[i][0]).abs();
        }
        mae /= j;
        if mae.is_nan(){
            mae = f32::INFINITY;
        }
        return mae;
    }

    pub fn mcc(&mut self, outputs: &Vec<Vec<f32>>, labels: &Vec<Vec<f32>>) -> f32 {
        let mut c:f32 = 0.;
        let mut tk:Vec<f32> = vec![0.; outputs[0].len()];
        let mut pk:Vec<f32> = vec![0.; outputs[0].len()];
        for i in 0..outputs.len(){
            let pred = get_argmax(&outputs[i]);
            let gt = get_argmax(&labels[i]);
            if pred == gt{
                c += 1.;
            }
            tk[gt] += 1.;
            pk[pred] += 1.;
        }
        let num_samples = outputs.len() as f32;
        let mut mcc = c * num_samples;
        let mut preds_square:f32 = 0.;
        let mut gts_square:f32 = 0.;
        for i in 0..outputs[0].len(){
            mcc -= tk[i] * pk[i];
            preds_square += pk[i].powi(2);
            gts_square += pk[i].powi(2);
        }
        let divisor = (num_samples.powi(2) - preds_square).sqrt() * (num_samples.powi(2) - gts_square).sqrt();
        if divisor != 0.{
            mcc = mcc / divisor;
        }else{
            mcc = 0.;
        }
        mcc = 1. - mcc.abs();
        return mcc
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
                    if current_node.function_id < 4{
                        let connection1 = current_node.connection2;
                        if !active.contains(&connection1) {
                            to_visit.push(connection1);
                            active.insert(connection1);
                        }
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

    pub fn get_mae(&mut self) -> f32{
        return self.mae;
    }

}

fn get_argmax(nets: &Vec<f32>) -> usize {
    nets.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, _)| index)
        .unwrap()
}
