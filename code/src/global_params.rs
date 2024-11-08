use std::fmt::{Display, Formatter};

#[derive(Clone)]
pub struct CgpParameters {
    pub graph_width: usize,
    pub iterations: usize,
    pub mu: usize,
    pub lambda: usize,
    pub eval_after_iterations: usize,
    pub nbr_inputs: usize,
    pub nbr_outputs: usize,
    pub mutation_type: i32,
    pub tau_0: f32,
    pub alpha: f32,
    pub beta:f32,
    pub roh:f32,
    pub distance_function:i32,
}

impl Default for CgpParameters {
    fn default() -> Self {
        CgpParameters {
            graph_width: 0,
            iterations: 1_000_000,
            mu: 1,
            lambda: 4,
            eval_after_iterations: 500,
            nbr_inputs: 0,
            nbr_outputs: 0,
            mutation_type: 0,
            tau_0: 1.,
            alpha: 1.,
            beta: 1.,
            roh:1.,
            distance_function:0,
        }
    }
}

impl Display for CgpParameters {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "############ Parameters ############\n")?;
        write!(f, "graph_width: {}\n", self.graph_width)?;
        write!(f, "iterations: {}\n", self.iterations)?;
        write!(f, "mu: {}\n", self.mu)?;
        write!(f, "lambda: {}\n", self.lambda)?;
        write!(f, "eval_after_iterations: {}\n", self.eval_after_iterations)?;
        write!(f, "nbr_inputs: {}\n", self.nbr_inputs)?;
        write!(f, "nbr_outputs: {}\n", self.nbr_outputs)?;
        write!(f, "mutation_type: {}\n", self.mutation_type)?;
        write!(f, "tau_ß: {}\n", self.tau_0)?;
        write!(f, "alpha: {}\n", self.alpha)?;
        write!(f, "beta: {}\n", self.beta)?;
        write!(f, "roh: {}\n", self.roh)?;
        write!(f, "distance_function: {}\n", self.distance_function)?;
        write!(f, "#########################\n")
    }
}
