use crate::utils::node_type::NodeType;
use rand::Rng;
use std::fmt::{Display, Formatter};
use usize;

#[derive(Clone)]
pub struct Node {
    pub position: usize,
    pub node_type: NodeType,
    pub nbr_inputs: usize,
    pub graph_width: usize,
    pub function_id: usize,
    pub connection1: usize,
    pub connection2: usize,
}

impl Display for Node {
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

impl Node {
    pub fn new(
        position: usize,
        nbr_inputs: usize,
        graph_width: usize,
        node_type: NodeType,
    ) -> Self {
        let function_id = rand::thread_rng().gen_range(0..=13) as usize;
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
        conn1_value: &Vec<f32>,
        conn2_value: &Vec<f32>,
    ) -> Vec<f32> {
        assert!(self.node_type != NodeType::InputNode);
        let mut res= vec![0.; conn1_value.len()];
        match self.function_id {
            0 => {
                for i in 0..conn1_value.len(){
                    res[i] = conn2_value[i] + conn1_value[i];
                }
                return res;
            },
            1 => {
                    for i in 0..conn1_value.len(){
                        res[i] = conn2_value[i] - conn1_value[i];
                    }
                    return res;
                },
            2 => {
                for i in 0..conn1_value.len(){
                    res[i] = conn2_value[i] * conn1_value[i];
                }
                return res;
            },
            3 => {
                for i in 0..conn1_value.len(){
                    res[i] = conn1_value[i] / (conn2_value[i] + 0.0000000001);
                }
                return res;
            },
            4 => {
                for i in 0..conn1_value.len(){
                    res[i] = conn1_value[i] * -1.;
                }
                return res;
            },
            5 => {
                for i in 0..conn1_value.len(){
                    res[i] = conn1_value[i].sin();
                }
                return res;
            },
            6 => {
                for i in 0..conn1_value.len(){
                    res[i] = conn1_value[i].cos();
                }
                return res;
            },
            7 => {
                for i in 0..conn1_value.len(){
                    res[i] = conn1_value[i].tan();
                }
                return res;
            },
            8 => {
                for i in 0..conn1_value.len(){
                    res[i] = conn1_value[i].tanh();
                }
                return res;
            },
            9 => {
                for i in 0..conn1_value.len(){
                    if conn1_value[i] <= 0. {
                        res[i] = 0.;
                    }else{
                        res[i] = conn1_value[i];
                    }
                }
                return res;
            },
            10 => {
                for i in 0..conn1_value.len(){
                    res[i] = conn1_value[i].exp();
                }
                return res;
            },
            11 => {
                for i in 0..conn1_value.len(){
                    if conn1_value[i] <= 0. {
                        res[i] = 0.;
                    }else{
                        res[i] = conn1_value[i].ln();
                    };
                }
                return res;
            },
            12 => {
                for i in 0..conn1_value.len(){
                    res[i] = conn1_value[i].abs();
                }
                return res;
            },
            13 => {
                for i in 0..conn1_value.len(){
                    res[i] = 1. / (1. + (-conn1_value[i]).exp())
                }
                return res;
            },
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
        self.function_id = gen_random_number(self.function_id, 14);
    }

    fn mutate_output_node(&mut self) {
        Node::mutate_connection(&mut self.connection1, self.graph_width + self.nbr_inputs);

        assert!(self.connection1 < self.position);
    }

    fn mutate_computational_node(&mut self) {
        let rand_nbr = rand::thread_rng().gen_range(0..=2);
        match rand_nbr {
            0 => Node::mutate_connection(&mut self.connection1, self.position),

            1 => Node::mutate_connection(&mut self.connection2, self.position),

            2 => self.mutate_function(),

            _ => {
                panic!("Mutation: output node something wrong")
            }
        };

        assert!(self.connection1 < self.position);
        assert!(self.connection2 < self.position);
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
