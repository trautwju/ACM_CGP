use crate::utils::boolean_functions as bf;
use crate::utils::cycle_checker::CGPEdges;
use crate::utils::node_type::NodeType;
use ndarray::prelude::*;
use rand::Rng;
use std::fmt::{Display, Formatter};
use usize;

#[derive(Clone)]
pub struct NodeDAG {
    pub position: usize,
    pub node_type: NodeType,
    pub nbr_inputs: usize,
    pub graph_width: usize,
    pub function_id: usize,
    pub connection1: usize,
    pub connection2: usize,
}

impl Display for NodeDAG {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Node Pos: {}, ", self.position)?;
        write!(f, "Node Type: {}, ", self.node_type)?;
        match self.function_id {
            0 => {
                write!(f, "Func: AND, ")?;
            }
            1 => {
                write!(f, "Func: OR, ")?;
            }
            2 => {
                write!(f, "Func: NAND, ")?;
            }
            3 => {
                write!(f, "Func: NOR, ")?;
            }
            _ => panic!(),
        };
        return writeln!(
            f,
            "Connections: ({}, {}), ",
            self.connection1, self.connection2
        );
    }
}

impl NodeDAG {
    pub fn new(
        position: usize,
        nbr_inputs: usize,
        graph_width: usize,
        node_type: NodeType,
        cgp_edges: &mut CGPEdges,
    ) -> Self {
        let function_id = rand::thread_rng().gen_range(0..=3) as usize;
        let connection1: usize;
        let connection2: usize;

        match node_type {
            NodeType::InputNode => {
                connection1 = usize::MAX;
                connection2 = usize::MAX;
            }
            NodeType::ComputationalNode => {
                connection1 = rand::thread_rng().gen_range(0..=position - 1) as usize;
                connection2 = rand::thread_rng().gen_range(0..=position - 1) as usize;
                cgp_edges.add_edge(position, connection1);
                cgp_edges.add_edge(position, connection2);
            }
            NodeType::OutputNode => {
                connection1 =
                    rand::thread_rng().gen_range(0..=nbr_inputs + graph_width - 1) as usize;
                connection2 = usize::MAX;
            }
        }

        Self {
            position,
            node_type,
            nbr_inputs,
            graph_width,
            function_id,
            connection1,
            connection2,
        }
    }

    pub fn execute(
        &self,
        conn1_value: &ArrayView1<bool>,
        conn2_value: &ArrayView1<bool>,
    ) -> Array1<bool> {
        assert!(self.node_type != NodeType::InputNode);

        match self.function_id {
            0 => bf::and(&conn1_value, &conn2_value),
            1 => bf::or(&conn1_value, &conn2_value),
            2 => bf::nand(&conn1_value, &conn2_value),
            3 => bf::nor(&conn1_value, &conn2_value),
            _ => panic!("wrong function id: {}", self.function_id),
        }
    }

    pub fn mutate(&mut self, cgp_edges: &mut CGPEdges) {
        assert!(self.node_type != NodeType::InputNode);

        match self.node_type {
            NodeType::OutputNode => self.mutate_output_node(),
            NodeType::ComputationalNode => self.mutate_computational_node(cgp_edges),
            _ => {
                panic!("Trying to mutate input node")
            }
        }
    }

    /// Upper Range excluded
    fn mutate_connection(
        connection: &mut usize,
        position: usize,
        upper_range: usize,
        cgp_edges: &mut CGPEdges,
    ) {
        let new_connection_id =
            gen_random_connection(*connection, position, upper_range, cgp_edges);
        cgp_edges.remove_edge(position, *connection);
        cgp_edges.add_edge(position, new_connection_id);
        *connection = new_connection_id;
    }

    fn mutate_function(&mut self) {
        self.function_id = gen_random_function_id(self.function_id, 4);
    }

    fn mutate_output_node(&mut self) {
        loop {
            let rand_nbr: usize =
                rand::thread_rng().gen_range(0..=self.nbr_inputs + self.graph_width - 1);

            if rand_nbr != self.connection1 {
                self.connection1 = rand_nbr;
                break;
            }
        }
    }

    fn mutate_computational_node(&mut self, cgp_edges: &mut CGPEdges) {
        let rand_nbr = rand::thread_rng().gen_range(0..=2);
        match rand_nbr {
            0 => NodeDAG::mutate_connection(
                &mut self.connection1,
                self.position,
                self.nbr_inputs + self.graph_width - 1,
                cgp_edges,
            ),

            1 => NodeDAG::mutate_connection(
                &mut self.connection2,
                self.position,
                self.nbr_inputs + self.graph_width - 1,
                cgp_edges,
            ),

            2 => self.mutate_function(),

            _ => {
                panic!("Mutation: output node something wrong")
            }
        };
    }
}

fn gen_random_function_id(excluded: usize, upper_range: usize) -> usize {
    loop {
        let rand_nbr: usize = rand::thread_rng().gen_range(0..=upper_range - 1);
        if rand_nbr != excluded {
            return rand_nbr;
        }
    }
}

fn gen_random_connection(
    previous_connection: usize,
    position: usize,
    upper_range: usize,
    cgp_edges: &mut CGPEdges,
) -> usize {
    loop {
        let rand_nbr: usize = rand::thread_rng().gen_range(0..=upper_range - 1);

        if (rand_nbr != previous_connection) & (rand_nbr != position) {
            if !cgp_edges.check_cycle(position, rand_nbr) {
                return rand_nbr;
            }
        }
    }
}
