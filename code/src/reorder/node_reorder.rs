use crate::utils::boolean_functions as bf;
use crate::utils::node_type::NodeType;
use ndarray::prelude::*;
use rand::Rng;
use std::fmt::{Display, Formatter};
use usize;

#[derive(Clone)]
pub struct NodeReorder {
    pub position: usize,
    pub node_type: NodeType,
    pub nbr_inputs: usize,
    pub graph_width: usize,
    pub function_id: usize,
    pub connection1: usize,
    pub connection2: usize,
}

impl Display for NodeReorder {
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

impl NodeReorder {
    pub fn new(
        position: usize,
        nbr_inputs: usize,
        graph_width: usize,
        node_type: NodeType,
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

    pub fn mutate(&mut self) {
        assert!(self.node_type != NodeType::InputNode);

        match self.node_type {
            NodeType::OutputNode => self.mutate_output_node(),
            NodeType::ComputationalNode => self.mutate_computational_node(),
            _ => {
                panic!("Trying to mutate input node")
            }
        }
    }

    fn mutate_connection(connection: &mut usize, upper_range: usize) {
        *connection = gen_random_number(*connection, upper_range);
    }

    fn mutate_function(&mut self) {
        self.function_id = gen_random_number(self.function_id, 4);
    }

    fn mutate_output_node(&mut self) {
        NodeReorder::mutate_connection(&mut self.connection1, self.graph_width + self.nbr_inputs);

        assert!(self.connection1 < self.position);
    }

    fn mutate_computational_node(&mut self) {
        let rand_nbr = rand::thread_rng().gen_range(0..=2);
        match rand_nbr {
            0 => NodeReorder::mutate_connection(&mut self.connection1, self.position),

            1 => NodeReorder::mutate_connection(&mut self.connection2, self.position),

            2 => self.mutate_function(),

            _ => {
                panic!("Mutation: output node something wrong")
            }
        };

        assert!(
            self.connection1 < self.position,
            "what was mutatet?: {}",
            rand_nbr
        );
        assert!(
            self.connection2 < self.position,
            "what was mutatet?: {}",
            rand_nbr
        );
    }

    pub fn set_new_position(&mut self, new_pos: usize, mutate_new_connections: bool) {
        if mutate_new_connections {
            if self.connection1 >= new_pos {
                NodeReorder::mutate_connection(&mut self.connection1, new_pos - 1);
            }
            if self.connection2 >= new_pos {
                NodeReorder::mutate_connection(&mut self.connection2, new_pos - 1);
            }
        }
        self.position = new_pos;
    }

    pub fn set_connection1(&mut self, new_con: usize) {
        self.connection1 = new_con;
    }

    pub fn set_connection2(&mut self, new_con: usize) {
        self.connection2 = new_con;
    }
}

fn gen_random_number(excluded: usize, upper_range: usize) -> usize {
    if upper_range <= 1 {
        return 0;
    }
    loop {
        let rand_nbr: usize = rand::thread_rng().gen_range(0..=upper_range - 1);
        if rand_nbr != excluded {
            return rand_nbr;
        }
    }
}
