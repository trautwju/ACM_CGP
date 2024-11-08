use crate::global_params::CgpParameters as g_params;
use crate::cgp_es::chromosome::Chromosome;
use float_eq::float_eq;
use rand::seq::SliceRandom;
use std::fmt::{Display, Formatter};
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rand::thread_rng;
use crate::utils::node_type::NodeType;

pub struct Runner {
    params: g_params,
    data: Vec<Vec<f32>>,
    label: Vec<Vec<f32>>,
    chromosomes: Vec<Chromosome>,
    best_fitness: f32,
    fitness_vals: Vec<f32>,
    parent_ids: Vec<usize>,
    mutation_type: usize,
    mutation_prob: f32,
    func_global_best_ant:Vec<i32>,
    conn_global_best_ant:Vec<i32>,
    global_best_fitness:f32,
    global_best:bool,
    func_tau_max:f32,
    func_tau_min:f32,
    conn_tau_max:f32,
    conn_tau_min:f32,
    p_best:f32,
    avg:f32,
    all_active_nodes:Vec<usize>,
    best_fitnesses:Vec<f32>,
    func_best_ants:Vec<Vec<i32>>,
    conn_best_ants:Vec<Vec<i32>>,
    mae_vals: Vec<f32>,
    exploration_rate:f32,
    two_conn_constant:usize,
    elitism_type:usize,
    num_active_nodes:Vec<usize>,
}

impl Display for Runner {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Parent: {}", self.chromosomes[self.parent_ids[0]])?;
        writeln!(f, "Fitness: {}", self.best_fitness)
    }
}

impl Runner {
    pub fn new(
        params: g_params,
        data: Vec<Vec<f32>>,
        label: Vec<Vec<f32>>,
        mut_type: usize,
        mut_prob: f32,
        global_best: bool,
        elitism_type: usize,
    ) -> Self {
        let mut chromosomes: Vec<Chromosome> = Vec::with_capacity(params.mu + params.lambda);
        let mut fitness_vals: Vec<f32> = Vec::with_capacity(params.mu + params.lambda);
        let best_fitnesses: Vec<f32> = vec![-1.; params.mu + params.lambda];
        let mut mae_vals: Vec<f32> = Vec::with_capacity(params.mu + params.lambda);

        for _ in 0..(params.mu + params.lambda) {
            let mut chromosome = Chromosome::new(params.clone());
            let fitness = chromosome.evaluate(&data, &label);
            fitness_vals.push(fitness);
            if label[0].len() == 1{
                mae_vals.push(chromosome.get_mae())
            }
            chromosomes.push(chromosome);
        }

        let best_fitness = get_min(&fitness_vals);
        let parent_ids = Vec::with_capacity(params.mu);
        let func_global_best_ant = vec![0; 1];
        let conn_global_best_ant = vec![0; 1];
        let func_best_ants = vec![vec![0;1];params.mu + params.lambda];
        let conn_best_ants = vec![vec![0;1];params.mu + params.lambda];
        let global_best_fitness = -1.;
        let global_best = global_best;
        let func_tau_max = 0.;
        let func_tau_min = 1. / 14.;
        let conn_tau_max = 0.;
        let conn_tau_min = 1. / 14.;
        let avg = 0.;
        let p_best = 0.;
        let all_active_nodes = Vec::new();
        let exploration_rate = 0.1;
        let two_conn_constant = (params.nbr_inputs + params.graph_width).pow(2) * 4;
        let num_active_nodes:Vec<usize> = vec![0; 1];

        Self {
            params,
            data,
            label,
            chromosomes,
            best_fitness,
            fitness_vals,
            parent_ids,
            mutation_type: mut_type,
            mutation_prob: mut_prob,
            func_global_best_ant,
            conn_global_best_ant,
            global_best_fitness,
            global_best,
            func_tau_max,
            func_tau_min,
            conn_tau_max,
            conn_tau_min,
            p_best,
            avg,
            all_active_nodes,
            best_fitnesses,
            func_best_ants,
            conn_best_ants,
            mae_vals,
            exploration_rate,
            two_conn_constant,
            elitism_type,
            num_active_nodes,
        }
    }

    pub fn learn_step(&mut self, replace_parents:bool) -> f32{
        self.mutate_chromosomes();
        self.eval_chromosomes();
        self.get_active_nodes();
        self.new_parents_by_method(replace_parents);
        return self.best_fitness;
    }

    pub fn new_parents_by_method(&mut self, replace_parents:bool){
        match self.elitism_type {
            0 => self.new_parent_by_neutral_search(replace_parents),
            1 => self.new_random_parents(replace_parents),
            2 => self.new_parents_by_fuss(replace_parents),
            _ => panic!("elitism_type not defined"),
        }
    }

    fn new_random_parents(&mut self, replace_parents:bool){
        let mut new_parents:Vec<usize> = Vec::from_iter(0..(self.params.mu + self.params.lambda));
        if self.parent_ids.len() == 0{
            new_parents = new_parents.choose_multiple(&mut thread_rng(), self.params.mu).cloned().collect();
            self.parent_ids = new_parents;
        }else{
            if replace_parents {
                self.parent_ids.sort();
                self.parent_ids.reverse();
                for i in 0..self.parent_ids.len() {
                    new_parents.remove(self.parent_ids[i]);
                }
            }
            new_parents = new_parents.choose_multiple(&mut thread_rng(), self.params.mu).cloned().collect();
            self.parent_ids = new_parents;
        }
    }

    fn new_parents_by_fuss(&mut self, replace_parents:bool){
        let mut new_parents_choices:Vec<usize> = Vec::from_iter(0..(self.params.mu + self.params.lambda));
        let mut new_parents:Vec<usize> = Vec::with_capacity(self.params.mu);
        if replace_parents {
            self.parent_ids.sort();
            self.parent_ids.reverse();
            for i in 0..self.parent_ids.len() {
                new_parents_choices.remove(self.parent_ids[i]);
            }
        }
        let mut observable_fitness_vals = self.fitness_vals.clone();
        observable_fitness_vals.retain(|x| *x !=f32::INFINITY);
        let mut max:f32 = f32::MAX;
        let mut min = f32::MAX;
        if observable_fitness_vals.len() != 0{
            max = get_max(&observable_fitness_vals);
            min = get_min(&observable_fitness_vals);
        }
        if observable_fitness_vals.len() < self.params.lambda{
            observable_fitness_vals = self.fitness_vals.clone();
        }
        if min == max{
            new_parents = new_parents_choices.choose_multiple(&mut thread_rng(), self.params.mu).cloned().collect();
            self.parent_ids = new_parents;
        }else{
            for _ in 0..self.params.mu{
                let val = thread_rng().gen_range(min..max);
                let mut min_dist = (val - self.fitness_vals[new_parents_choices[0]]).abs();
                let mut current_index:usize = 0;
                for i in 1..new_parents_choices.len(){
                    if (val - self.fitness_vals[new_parents_choices[i]]).abs() < min_dist {
                        min_dist = (val - self.fitness_vals[new_parents_choices[i]]).abs();
                        current_index = i;
                    }
                }
                new_parents.push(new_parents_choices[current_index]);
                new_parents_choices.remove(current_index);
            }
            self.parent_ids = new_parents;
        }
    }

    fn new_parent_by_neutral_search(&mut self, replace_parents:bool) {
        let mut new_parents_choices:Vec<usize> = Vec::from_iter(0..(self.params.mu + self.params.lambda));
        let mut remaining_fitness_vals = self.fitness_vals.clone();
        let mut new_parents:Vec<usize> = Vec::with_capacity(self.params.mu);
        if replace_parents{
            self.parent_ids.sort();
            self.parent_ids.reverse();
            for i in 0..self.parent_ids.len() {
                new_parents_choices.remove(self.parent_ids[i]);
                remaining_fitness_vals.remove(self.parent_ids[i]);
            }
            for _ in 0..self.params.mu{
                let index = get_argmin(&remaining_fitness_vals);
                new_parents.push(new_parents_choices[index]);
                new_parents_choices.remove(index);
                remaining_fitness_vals.remove(index);
            }
            self.parent_ids = new_parents;
        }else{
            for _ in 0..self.params.mu {
                let min = get_min(&remaining_fitness_vals);
                let mut res = Vec::with_capacity(self.params.mu + self.params.lambda);
                get_argmins_of_value(&remaining_fitness_vals, &mut res, min);
                res.sort();
                res.reverse();
                if new_parents.len() + res.len() >= self.params.mu {
                    let diff = self.params.mu - new_parents.len();
                    if diff == 0{
                        break;
                    }
                    if res.len() == diff{
                        for i in 0..res.len() {
                            new_parents.push(new_parents_choices[res[i]]);
                        }
                    }else{
                        for i in 0..res.len() {
                            if new_parents.len() + (res.len() - i) > self.params.mu{
                                if self.parent_ids.contains(&new_parents_choices[res[i]]){
                                    continue;
                                }
                            }
                            if new_parents.len() < self.params.mu{
                                new_parents.push(new_parents_choices[res[i]]);
                            }
                        }
                    }
                }else {
                    for i in 0..res.len() {
                        new_parents.push(new_parents_choices[res[i]]);
                        new_parents_choices.remove(res[i]);
                        remaining_fitness_vals.remove(res[i]);
                    }
                }
            }
            self.parent_ids = new_parents;
        }
    }

    fn mutate_chromosomes(&mut self) {
        let mut current_parent: usize = 0;
        // mutate new chromosomes; do not mutate parent
        for i in 0..(self.params.mu + self.params.lambda) {
            if self.parent_ids.contains(&i) {
                continue;
            }

            self.chromosomes[i] = self.chromosomes[self.parent_ids[current_parent]].clone();
            current_parent = (current_parent + 1) % self.params.mu;

            match self.mutation_type {
                0 => {
                    self.chromosomes[i].mutate_single();
                }
                1 => {
                    self.chromosomes[i].mutate_prob(self.mutation_prob);
                }

                _ => {
                    panic!("mutation type not def")
                }
            }
        }
    }

    pub fn evaluate_chromosomes_cgp(&mut self, data:Vec<Vec<f32>>, labels:Vec<Vec<f32>>) -> f32{
        self.data = data;
        self.label = labels;
        self.eval_chromosomes();
        let local_best = get_argmin(&self.fitness_vals);
        return self.fitness_vals[local_best];
    }

    pub fn best_ant_chromosome(&mut self, conn_pheromone_table:&Vec<Vec<f32>>, func_pheromone_table:&Vec<Vec<f32>>, data:Vec<Vec<f32>>, labels:Vec<Vec<f32>>) -> f32{
        self.data = data;
        self.label = labels;
        let connections = calculate_greedy(conn_pheromone_table);
        for i in 0..conn_pheromone_table.len(){
            if self.chromosomes[0].nodes_grid[self.params.nbr_inputs + i].node_type != NodeType::OutputNode {
                let conn1 = connections[i] as usize / (self.params.nbr_inputs + self.params.graph_width);
                let conn2 = connections[i] as usize % (self.params.nbr_inputs + self.params.graph_width);
                self.chromosomes[0].nodes_grid[self.params.nbr_inputs + i].connection1 = conn1;
                self.chromosomes[0].nodes_grid[self.params.nbr_inputs + i].connection2 = conn2;
            } else{
                let conn1 = connections[i] as usize;
                self.chromosomes[0].nodes_grid[self.params.nbr_inputs + i].connection1 = conn1;
            }
        }
        let functions = calculate_greedy(func_pheromone_table);
        for i in 0..functions.len() {
            self.chromosomes[0].nodes_grid[self.params.nbr_inputs + i].function_id = functions[i];
        }
        self.eval_chromosomes();
        return self.fitness_vals[0];
    }

    pub fn best_ant_one_table(&mut self, pheromone_table:&Vec<Vec<f32>>, data:Vec<Vec<f32>>, labels:Vec<Vec<f32>>) -> f32 {
        self.data = data;
        self.label = labels;
        let ant = calculate_greedy(pheromone_table);
        for i in 0..ant.len(){
            if i < self.params.graph_width{
                let mut func_id = ant[i] as usize;
                if func_id < self.two_conn_constant{
                    func_id = func_id / self.two_conn_constant;
                }else{
                    func_id = func_id - self.two_conn_constant;
                    func_id = func_id / (self.params.nbr_inputs + self.params.graph_width);
                }
                self.chromosomes[0].nodes_grid[self.params.nbr_inputs + i].function_id = func_id;
            }
            if self.chromosomes[0].nodes_grid[self.params.nbr_inputs + i].node_type != NodeType::OutputNode {
                let mut adjusted_j = ant[i] as usize;
                if adjusted_j < self.two_conn_constant{
                    let adjusted_j = adjusted_j % (self.params.nbr_inputs + self.params.graph_width).pow(2);
                    let conn1 = adjusted_j / (self.params.nbr_inputs + self.params.graph_width);
                    let conn2 = adjusted_j % (self.params.nbr_inputs + self.params.graph_width);
                    self.chromosomes[0].nodes_grid[self.params.nbr_inputs + i].connection1 = conn1;
                    self.chromosomes[0].nodes_grid[self.params.nbr_inputs + i].connection2 = conn2;
                }else{
                    adjusted_j -= self.two_conn_constant;
                    let conn = adjusted_j % (self.params.nbr_inputs + self.params.graph_width);
                    self.chromosomes[0].nodes_grid[self.params.nbr_inputs + i].connection1 = conn;
                }
            } else{
                let conn1 = ant[i];
                self.chromosomes[0].nodes_grid[self.params.nbr_inputs + i].connection1 = conn1;
            }
        }
        self.eval_chromosomes();
        return self.fitness_vals[0];
    }

    pub fn initialize_pheromone_table(&mut self, pheromone_table: &mut Vec<Vec<f32>>){
        for i in 0..pheromone_table.len(){
            if i < self.params.graph_width{
                for j in 0..pheromone_table[0].len() {
                    let current_node = i as f32 + self.params.nbr_inputs as f32;
                    let conn1 = (j / (self.params.nbr_inputs + self.params.graph_width)) as f32;
                    let conn2 = (j % (self.params.nbr_inputs + self.params.graph_width)) as f32;
                    if current_node > conn1 && current_node > conn2 {
                        pheromone_table[i][j] = self.params.tau_0;
                    }
                }
            }
            else{
                for j in 0..self.params.nbr_inputs + self.params.graph_width{
                    pheromone_table[i][j] = self.params.tau_0;
                }
            }
        }
        self.global_best_fitness = -1.;
    }

    pub fn ant_learn_two_tables(&mut self, pheromone_table_functions: &mut Vec<Vec<f32>>,  pheromone_table_connections: &mut Vec<Vec<f32>>, ant_type:usize) -> f32{
        let func_probs = self.calculate_function_probabilities(pheromone_table_functions, ant_type);
        let mut func_ants: Vec<Vec<i32>> = vec![vec![0; self.params.graph_width];  self.params.mu + self.params.lambda];
        let mut rng = thread_rng();
        let func_choices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13];
        let conn_probs = self.calculate_connection_probabilities(pheromone_table_connections, ant_type);
        let mut conn_ants: Vec<Vec<i32>> = vec![vec![0; self.params.graph_width + self.params.nbr_outputs];  self.params.mu + self.params.lambda];
        let conn_choices: Vec<i32> = (0..pheromone_table_connections[0].len() as i32).collect();
        if ant_type != 0{
            self.exploration_rate = 1.;
        }
        for i in 0..self.params.mu + self.params.lambda {
            for j in 0..self.params.graph_width + self.params.nbr_outputs{
                let p:f32 = rng.gen();
                if p <= self.exploration_rate {
                    // exploration
                    if j < self.params.graph_width{
                        let dist = WeightedIndex::new(&func_probs[j]).unwrap();
                        func_ants[i][j] = func_choices[dist.sample(&mut rng)];
                    }
                    let dist = WeightedIndex::new(&conn_probs[j]).unwrap();
                    conn_ants[i][j] = conn_choices[dist.sample(&mut rng)];
                }
                else{
                    // exploitation
                    if j < self.params.graph_width {
                        let argmax = get_argmax(&func_probs[j]);
                        func_ants[i][j] = func_choices[argmax];
                    }
                    let argmax = get_argmax(&conn_probs[j]);
                    conn_ants[i][j] = conn_choices[argmax];
                }
            }
        }
        for i in 0..self.params.mu + self.params.lambda {
            for j in 0..self.params.graph_width + self.params.nbr_outputs {
                if j < self.params.graph_width{
                    self.chromosomes[i].nodes_grid[self.params.nbr_inputs + j].function_id = func_ants[i][j] as usize;
                }
                if self.chromosomes[i].nodes_grid[self.params.nbr_inputs + j].node_type != NodeType::OutputNode {
                    let conn1 = conn_ants[i][j] as usize / (self.params.nbr_inputs + self.params.graph_width);
                    let conn2 = conn_ants[i][j] as usize % (self.params.nbr_inputs + self.params.graph_width);
                    self.chromosomes[i].nodes_grid[self.params.nbr_inputs + j].connection1 = conn1;
                    self.chromosomes[i].nodes_grid[self.params.nbr_inputs + j].connection2 = conn2;
                } else{
                    let conn1 = conn_ants[i][j] as usize;
                    self.chromosomes[i].nodes_grid[self.params.nbr_inputs + j].connection1 = conn1;
                }
            }
        }

        self.eval_chromosomes();
        self.get_active_nodes();

        if ant_type == 3{
            for i in 0..conn_ants.len(){
                if self.best_fitnesses[i] == -1.{
                    self.best_fitnesses[i] = self.fitness_vals[i];
                    self.conn_best_ants[i] = conn_ants[i].clone();
                    self.func_best_ants[i] = func_ants[i].clone();
                }
                if self.fitness_vals[i] < self.best_fitnesses[i]{
                    self.best_fitnesses[i] = self.fitness_vals[i];
                    self.conn_best_ants[i] = conn_ants[i].clone();
                    self.func_best_ants[i] = func_ants[i].clone();
                }
            }
        }

        let local_best_ant = get_argmin(&self.fitness_vals);
        if self.global_best_fitness == -1.{
            self.global_best_fitness = self.fitness_vals[local_best_ant];
            self.func_global_best_ant = func_ants[local_best_ant].clone();
            self.conn_global_best_ant = conn_ants[local_best_ant].clone();
            if ant_type == 2{
                let mut y = 1. - self.params.roh;
                if y == 0.{
                    y += 0.0001;
                }
                self.conn_tau_max = 1. / y * (1. / self.global_best_fitness);
                self.conn_tau_min = (self.conn_tau_max * (1. - self.p_best)) / ((self.avg - 1.) * self.p_best);

                self.func_tau_max = 1. / y * (1. / self.global_best_fitness);
                self.func_tau_min = (self.func_tau_max * (1. - 1. / 14.)) / ((14. - 1.) * (1. / 14.));
            }
        }
        if self.fitness_vals[local_best_ant] < self.global_best_fitness{
            self.global_best_fitness = self.fitness_vals[local_best_ant];
            self.func_global_best_ant = func_ants[local_best_ant].clone();
            self.conn_global_best_ant = conn_ants[local_best_ant].clone();
            if ant_type == 2{
                let mut y = 1. - self.params.roh;
                if y == 0.{
                    y += 0.0001;
                }
                self.conn_tau_max = 1. / y * (1. / self.global_best_fitness);
                self.conn_tau_min = (self.conn_tau_max * (1. - self.p_best)) / ((self.avg - 1.) * self.p_best);

                self.func_tau_max = 1. / y * (1. / self.global_best_fitness);
                self.func_tau_min = (self.func_tau_max * (1. - 1. / 14.)) / ((14. - 1.) * (1. / 14.));
            }
        }

        match ant_type{
            0 => self.acs_update(pheromone_table_functions, func_ants, local_best_ant, false),
            1 => self.as_update(pheromone_table_functions, func_ants),
            2 => self.mmas_update_func(pheromone_table_functions, func_ants, local_best_ant),
            3 => self.aslbt_update(pheromone_table_functions, func_ants),
            _ => panic!(),
        }

        match ant_type{
            0 => self.acs_update(pheromone_table_connections, conn_ants, local_best_ant, true),
            1 => self.as_update(pheromone_table_connections, conn_ants),
            2 => self.mmas_update_conn(pheromone_table_connections, conn_ants, local_best_ant),
            3 => self.aslbt_update(pheromone_table_connections, conn_ants),
            _ => panic!(),
        }

        return self.fitness_vals[local_best_ant];
    }

    pub fn ant_learn_one_table(&mut self, pheromone_table: &mut Vec<Vec<f32>>, ant_type:usize) -> f32 {
        let probs = self.calculate_one_probs(pheromone_table, ant_type);
        let mut rng = thread_rng();
        let choices: Vec<i32> = (0..pheromone_table[0].len() as i32).collect();
        let mut ants: Vec<Vec<i32>> = vec![vec![0; self.params.graph_width + self.params.nbr_outputs];  self.params.mu + self.params.lambda];
        for i in 0..self.params.mu + self.params.lambda {
            for j in 0..self.params.graph_width + self.params.nbr_outputs{
                let p:f32 = rng.gen();
                if p < self.exploration_rate {
                    // exploration
                    let dist = WeightedIndex::new(&probs[j]).unwrap();
                    ants[i][j] = choices[dist.sample(&mut rng)];
                }
                else{
                    // exploitation
                    let argmax = get_argmax(&probs[j]);
                    ants[i][j] = choices[argmax];
                }
            }
        }
        self.exploration_rate *= 0.99;

        for i in 0..self.params.mu + self.params.lambda {
            for j in 0..self.params.graph_width + self.params.nbr_outputs {
                if j < self.params.graph_width{
                    let mut func_id = ants[i][j] as usize;
                    if func_id < self.two_conn_constant{
                        func_id = func_id / self.two_conn_constant;
                    }else{
                        func_id = func_id - self.two_conn_constant;
                        func_id = func_id / (self.params.nbr_inputs + self.params.graph_width);
                    }
                    self.chromosomes[i].nodes_grid[self.params.nbr_inputs + j].function_id = func_id;
                }

                if self.chromosomes[i].nodes_grid[self.params.nbr_inputs + j].node_type != NodeType::OutputNode {
                    let mut adjusted_j = ants[i][j] as usize;
                    if adjusted_j < self.two_conn_constant{
                        let adjusted_j = adjusted_j % (self.params.nbr_inputs + self.params.graph_width).pow(2);
                        let conn1 = adjusted_j / (self.params.nbr_inputs + self.params.graph_width);
                        let conn2 = adjusted_j % (self.params.nbr_inputs + self.params.graph_width);
                        self.chromosomes[i].nodes_grid[self.params.nbr_inputs + j].connection1 = conn1;
                        self.chromosomes[i].nodes_grid[self.params.nbr_inputs + j].connection2 = conn2;
                    }else{
                        adjusted_j -= self.two_conn_constant;
                        let conn = adjusted_j % (self.params.nbr_inputs + self.params.graph_width);
                        self.chromosomes[i].nodes_grid[self.params.nbr_inputs + j].connection1 = conn;
                    }
                } else{
                    let conn1 = ants[i][j] as usize;
                    self.chromosomes[i].nodes_grid[self.params.nbr_inputs + j].connection1 = conn1;
                }
            }
        }

        self.eval_chromosomes();
        self.get_active_nodes();

        if ant_type == 3{
            for i in 0..ants.len(){
                if self.fitness_vals[i] < self.best_fitnesses[i]{
                    self.best_fitnesses[i] = self.fitness_vals[i];
                    self.func_best_ants[i] = ants[i].clone();
                }
            }
        }

        let local_best_ant = get_argmin(&self.fitness_vals);

        if self.fitness_vals[local_best_ant] < self.global_best_fitness{
            self.global_best_fitness = self.fitness_vals[local_best_ant];
            self.func_global_best_ant = ants[local_best_ant].clone();
            if ant_type == 2{
                self.conn_tau_max = 1. / (1. - self.params.roh) * (1. / self.global_best_fitness);
                self.conn_tau_min = (self.conn_tau_max * (1. - self.p_best)) / ((self.avg - 1.) * self.p_best);
            }
        }

        match ant_type{
            0 => self.acs_update(pheromone_table, ants, local_best_ant, true),
            1 => self.as_update(pheromone_table, ants),
            2 => self.mmas_update_func(pheromone_table, ants, local_best_ant),
            3 => self.aslbt_update_func(pheromone_table),
            _ => panic!(),
        }

        return self.fitness_vals[local_best_ant];
    }

    pub fn init_one_pheromone_table(&mut self, pheromone_table: &mut Vec<Vec<f32>>){
        for i in 0..pheromone_table.len(){
            if i < self.params.graph_width{
                for j in 0..pheromone_table[0].len() {
                    let current_node = i as f32 + self.params.nbr_inputs as f32;
                    if j < self.two_conn_constant{
                        let adjusted_j = j % (self.params.nbr_inputs + self.params.graph_width).pow(2);
                        let conn1 = (adjusted_j / (self.params.nbr_inputs + self.params.graph_width)) as f32;
                        let conn2 = (adjusted_j % (self.params.nbr_inputs + self.params.graph_width)) as f32;
                        if current_node > conn1 && current_node > conn2 {
                            pheromone_table[i][j] = self.params.tau_0;
                        }
                    }
                    else{
                        let mut adjusted_j = j as f32 - self.two_conn_constant as f32;
                        adjusted_j = adjusted_j % (self.params.nbr_inputs + self.params.graph_width) as f32;
                        if current_node > adjusted_j{
                            pheromone_table[i][j] = self.params.tau_0;
                        }
                    }
                }
            }
            else{
                for j in 0..self.params.nbr_inputs + self.params.graph_width{
                    pheromone_table[i][j] = self.params.tau_0;
                }
            }
        }
        self.global_best_fitness = 1000.;
    }


    pub fn as_update(&mut self, pheromone_table:&mut Vec<Vec<f32>>, ants:Vec<Vec<i32>>){
        for i in 0..pheromone_table.len(){
            for j in 0..pheromone_table[0].len() {
                pheromone_table[i][j] = pheromone_table[i][j] * self.params.roh;
            }
        }
        for a in 0..ants.len(){
            for i in 0..ants[0].len(){
                let val = ants[a][i] as usize;
                pheromone_table[i][val] = pheromone_table[i][val] + 1. / (1. + self.fitness_vals[a]);
            }
        }
    }

    pub fn acs_update(&mut self, pheromone_table:&mut Vec<Vec<f32>>, ants:Vec<Vec<i32>>, best_ant:usize, conn:bool){
        //local update
        for a in 0..ants.len(){
            for i in 0..ants[0].len(){
                let val = ants[a][i] as usize;
                pheromone_table[i][val] = self.params.roh * pheromone_table[i][val] + (1. - self.params.roh) * self.params.tau_0;
            }
        }

        //global update
        for i in 0..pheromone_table.len(){
            for j in 0..pheromone_table[0].len(){
                pheromone_table[i][j] = pheromone_table[i][j] * (1. - self.params.alpha);
            }
            if self.global_best{
                let mut val  = self.conn_global_best_ant[i] as usize;
                if conn == false{
                    val = self.func_global_best_ant[i] as usize;
                }
                pheromone_table[i][val] += self.params.alpha / (1. + self.global_best_fitness);
            }else{
                let val = ants[best_ant][i] as usize;
                pheromone_table[i][val] += self.params.alpha / (1. + self.fitness_vals[best_ant]);
            }
        }
    }

    pub fn mmas_update_func(&mut self, pheromone_table:&mut Vec<Vec<f32>>, ants:Vec<Vec<i32>>, best_ant:usize){
        for i in 0..pheromone_table.len(){
            for j in 0..pheromone_table[0].len(){
                if pheromone_table[i][j] != 0.{
                    pheromone_table[i][j] = (pheromone_table[i][j] * self.params.roh).max(self.func_tau_min);
                }
            }
            if self.global_best{
                let val = self.func_global_best_ant[i] as usize;
                pheromone_table[i][val] =  (pheromone_table[i][val] + 1./ (1. + self.global_best_fitness)).min(self.func_tau_max);
            }else{
                let val = ants[best_ant][i] as usize;
                pheromone_table[i][val] = (pheromone_table[i][val] + 1./ (1. + self.fitness_vals[best_ant])).min(self.func_tau_max);
            }
        }
    }

    pub fn mmas_update_conn(&mut self, pheromone_table:&mut Vec<Vec<f32>>, ants:Vec<Vec<i32>>, best_ant:usize){
        for i in 0..pheromone_table.len(){
            for j in 0..pheromone_table[0].len(){
                if pheromone_table[i][j] != 0.{
                    pheromone_table[i][j] = (pheromone_table[i][j] * self.params.roh).max(self.conn_tau_min);
                }
            }
            if self.global_best{
                let val = self.conn_global_best_ant[i] as usize;
                pheromone_table[i][val] =  (pheromone_table[i][val] + 1./ (1. + self.global_best_fitness)).min(self.conn_tau_max);
            }else{
                let val = ants[best_ant][i] as usize;
                pheromone_table[i][val] = (pheromone_table[i][val] + 1./ (1. + self.fitness_vals[best_ant])).min(self.conn_tau_max);
            }
        }
    }

    pub fn aslbt_update_conn(&mut self, pheromone_table:&mut Vec<Vec<f32>>){
        for i in 0..pheromone_table.len(){
            for j in 0..pheromone_table[0].len(){
                    pheromone_table[i][j] = pheromone_table[i][j] * self.params.roh;
            }
            for j in 0..self.conn_best_ants.len(){
                let val = self.conn_best_ants[j][i] as usize;
                if self.best_fitnesses[j] == f32::INFINITY{
                    self.best_fitnesses[j] = 10000000000.;
                }
                pheromone_table[i][val] = pheromone_table[i][val] + self.global_best_fitness / self.best_fitnesses[j];
            }
        }
    }

    pub fn aslbt_update(&mut self, pheromone_table:&mut Vec<Vec<f32>>, ants:Vec<Vec<i32>>){
        for i in 0..pheromone_table.len(){
            for j in 0..pheromone_table[0].len() {
                pheromone_table[i][j] = pheromone_table[i][j] * self.params.roh;
            }
        }
        for a in 0..ants.len(){
            for i in 0..ants[0].len(){
                let val = ants[a][i] as usize;
                if (self.fitness_vals[a] != f32::INFINITY) && (self.best_fitnesses[a] != f32::INFINITY){
                    pheromone_table[i][val] = pheromone_table[i][val] + (self.best_fitnesses[a] / self.fitness_vals[a]);
                }
            }
        }
    }

    pub fn aslbt_update_func(&mut self, pheromone_table:&mut Vec<Vec<f32>>){
        for i in 0..pheromone_table.len(){
            for j in 0..pheromone_table[0].len(){
                pheromone_table[i][j] = pheromone_table[i][j] * self.params.roh;
            }
            for j in 0..self.func_best_ants.len(){
                let val = self.func_best_ants[j][i] as usize;
                pheromone_table[i][val] = pheromone_table[i][val] + self.global_best_fitness / self.best_fitnesses[j];
            }
        }
    }

    pub fn initialize_mmas(&mut self, pheromone_table:&mut Vec<Vec<f32>>){
        for i in 0..pheromone_table.len(){
            if self.chromosomes[0].nodes_grid[self.params.nbr_inputs + i].node_type != NodeType::OutputNode {
                if self.chromosomes[0].nodes_grid[self.params.nbr_inputs + i].function_id < 4{
                    let num_poss:f32 = ((self.params.nbr_inputs + i) * (self.params.nbr_inputs + i)) as f32;
                    self.avg = self.avg + num_poss;
                }else{
                    let num_poss:f32 = (self.params.nbr_inputs + i) as f32;
                    self.avg = self.avg + num_poss;
                }
            } else{
                let num_poss:f32 = (self.params.nbr_inputs + i) as f32;
                self.avg = self.avg + num_poss;
            }
        }
        self.avg = self.avg / pheromone_table.len() as f32;
        //self.p_best = 1. / self.avg;
        self.p_best = 0.05;
    }

    pub fn calculate_one_probs(&mut self, pheromone_table: &Vec<Vec<f32>>, ant_type:usize) -> Vec<Vec<f32>>{
        let mut probs = vec![vec![0.0 as f32; pheromone_table[0].len()];  pheromone_table.len()];
        for i in 0..pheromone_table.len(){
            for j in 0..pheromone_table[0].len(){
                if i < self.params.graph_width {
                    let current_node = i as f32 + self.params.nbr_inputs as f32;
                    if j < self.two_conn_constant {
                        let adjusted_j = j % (self.params.nbr_inputs + self.params.graph_width).pow(2);
                        let mut conn1 = (adjusted_j / (self.params.nbr_inputs + self.params.graph_width)) as f32;
                        conn1 = conn1.floor();
                        if conn1 < self.params.nbr_inputs as f32 {
                            conn1 = self.params.nbr_inputs as f32 - 1.;
                        }
                        let mut conn2 = (adjusted_j % (self.params.nbr_inputs + self.params.graph_width)) as f32;
                        if conn2 < self.params.nbr_inputs as f32 {
                            conn2 = self.params.nbr_inputs as f32 - 1.;
                        }
                        if pheromone_table[i][j] != 0. {
                            let mut x = match self.params.distance_function {
                                0 => euclidean(current_node, conn1, conn2),
                                1 => manhattan(current_node, conn1, conn2),
                                2 => logarithmic(current_node, conn1, conn2),
                                3 => constant(),
                                _ => panic!("Wrong Distance Function"),
                            };
                            x = (1. / x).powf(self.params.beta);
                            if ant_type == 0{
                                probs[i][j] = pheromone_table[i][j] * x;
                            }else{
                                probs[i][j] = pheromone_table[i][j].powf(self.params.alpha) * x;
                            }
                        }
                    } else {
                        let mut adjusted_j = j as f32 - self.two_conn_constant as f32;
                        adjusted_j = adjusted_j % (self.params.nbr_inputs + self.params.graph_width) as f32;
                        if pheromone_table[i][j] != 0. {
                            let mut x = match self.params.distance_function {
                                0 => euclidean(current_node, adjusted_j, current_node),
                                1 => manhattan(current_node, adjusted_j, current_node),
                                2 => logarithmic_one(current_node, adjusted_j),
                                3 => constant(),
                                _ => panic!("Wrong Distance Function"),
                            };
                            x = (1. / x).powf(self.params.beta);
                            if ant_type == 0{
                                probs[i][j] = pheromone_table[i][j] * x;
                            }else{
                                probs[i][j] = pheromone_table[i][j].powf(self.params.alpha) * x;
                            }
                        }
                    }
                }else{
                    let current_node = i as f32 + self.params.nbr_inputs as f32;
                    if pheromone_table[i][j] != 0. {
                        let mut x = match self.params.distance_function {
                            0 => euclidean(current_node, j as f32, current_node),
                            1 => manhattan(current_node, j as f32, current_node),
                            2 => logarithmic_one(current_node, j as f32),
                            3 => constant(),
                            _ => panic!("Wrong Distance Function"),
                        };
                        x = (1. / x).powf(self.params.beta);
                        if ant_type == 0{
                            probs[i][j] = pheromone_table[i][j] * x;
                        }else{
                            probs[i][j] = pheromone_table[i][j].powf(self.params.alpha) * x;
                        }
                    }
                }
            }
            let column_sum : f32 = probs[i].iter().sum();
            for j in 0..pheromone_table[0].len(){
                probs[i][j] = probs[i][j] / column_sum;
            }
        }
        return probs;
    }


    pub fn calculate_connection_probabilities(&mut self, pheromone_table: &Vec<Vec<f32>>, ant_type:usize) -> Vec<Vec<f32>>{
        let mut probs = vec![vec![0.0 as f32; pheromone_table[0].len()];  pheromone_table.len()];
        for i in 0..pheromone_table.len(){
            for j in 0..pheromone_table[0].len() {

                if i < self.params.graph_width {
                    let current_node = i as f32 + self.params.nbr_inputs as f32;
                    let mut conn1 = (j / (self.params.nbr_inputs + self.params.graph_width)) as f32;
                    conn1 = conn1.floor();
                    if conn1 < self.params.nbr_inputs as f32 {
                        conn1 = self.params.nbr_inputs as f32 - 1.;
                    }
                    let mut conn2 = (j % (self.params.nbr_inputs + self.params.graph_width)) as f32;
                    if conn2 < self.params.nbr_inputs as f32 {
                        conn2 = self.params.nbr_inputs as f32 - 1.;
                    }
                    if pheromone_table[i][j] != 0. {
                        let mut x = match self.params.distance_function {
                            0 => euclidean(current_node, conn1, conn2),
                            1 => manhattan(current_node, conn1, conn2),
                            2 => logarithmic(current_node, conn1, conn2),
                            3 => constant(),
                            _ => panic!("Wrong Distance Function"),
                        };
                        if x == 0.{
                            x += 1.;
                        }
                        x = (1. / x).powf(self.params.beta);
                        if ant_type == 0{
                            probs[i][j] = pheromone_table[i][j] * x;
                        }else{
                            probs[i][j] = pheromone_table[i][j].powf(self.params.alpha) * x;
                        }
                    }
                } else{
                    let current_node = i as f32 + self.params.nbr_inputs as f32;
                    if pheromone_table[i][j] != 0. {
                        let mut x = match self.params.distance_function {
                            0 => euclidean(current_node, j as f32, current_node),
                            1 => manhattan(current_node, j as f32, current_node),
                            2 => logarithmic_one(current_node, j as f32),
                            3 => constant(),
                            _ => panic!("Wrong Distance Function"),
                        };
                        if x == 0.{
                            x += 1.;
                        }
                        x = (1. / x).powf(self.params.beta);
                        if ant_type == 0{
                            probs[i][j] = pheromone_table[i][j] * x;
                        }else{
                            probs[i][j] = pheromone_table[i][j].powf(self.params.alpha) * x;
                        }
                    }
                }
            }

            let column_sum : f32 = probs[i].iter().sum();
            for j in 0..pheromone_table[0].len(){
                probs[i][j] = probs[i][j] / column_sum;
            }
        }
        return probs;
    }

    pub fn get_active_nodes(&mut self){
        self.all_active_nodes = Vec::new();
        self.num_active_nodes = Vec::new();
        for i in 0..self.chromosomes.len(){
            self.chromosomes[i].get_active_nodes_id();
            let active_nodes: Vec<_> = self.chromosomes[i].active_nodes.iter().collect();
            let active_nodes = active_nodes[0];
            self.num_active_nodes.push(active_nodes.len());
            for j in 0..active_nodes.len(){
                self.all_active_nodes.push(active_nodes[j]);
            }
        }
    }


    fn eval_chromosomes(&mut self) {
        for i in 0..(self.params.mu + self.params.lambda) {
            let mut fitness = self.chromosomes[i].evaluate(&self.data, &self.label);
            if self.label[0].len() == 1{
                self.mae_vals[i] = self.chromosomes[i].get_mae();
            }
            if fitness.is_nan(){
                fitness = f32::INFINITY;
            }
            self.fitness_vals[i] = fitness;
        }
        let best_fitness = get_min(&self.fitness_vals);

        self.best_fitness = best_fitness;
    }

    pub fn get_best_fitness(&self) -> f32 {
        return self.best_fitness;
    }

    pub fn get_best_mae(&self) -> f32 {
        let index = get_argmax(&self.fitness_vals);
        return self.mae_vals[index];
    }

    pub fn get_num_active_nodes(&self)  -> Vec<usize>{ return self.num_active_nodes.clone();}

    pub fn get_all_active_nodes(&self) -> Vec<usize>{
        return self.all_active_nodes.clone();
    }

    pub fn get_parent(&self) -> Chromosome {
        return self.chromosomes[self.parent_ids[0]].clone();
    }

    pub fn get_fitnesses(&self) -> Vec<f32>{
        return self.fitness_vals.clone();
    }

    pub fn get_maes(&self) -> Vec<f32>{
        return self.mae_vals.clone();
    }

    pub fn get_average_fitness(&self) -> f32{
        let sum:f32 = self.fitness_vals.iter().sum();
        return  sum / self.fitness_vals.len() as f32;
    }

    pub fn get_average_mae(&self) -> f32{
        let sum:f32 = self.mae_vals.iter().sum();
        return  sum / self.mae_vals.len() as f32;
    }
    pub fn calculate_function_probabilities(&self, pheromone_table: &Vec<Vec<f32>>, ant_type:usize) -> Vec<Vec<f32>>{
        let mut probs :Vec<Vec<f32>>= vec![vec![0.0; 14];  pheromone_table.len()];
        for i in 0..pheromone_table.len() {
            for j in 0..pheromone_table[0].len(){
                if ant_type == 0{
                    probs[i][j] = pheromone_table[i][j];
                }else{
                    probs[i][j] = pheromone_table[i][j].powf(self.params.alpha);
                }
            }
        }

        for i in 0..pheromone_table.len(){
            let column_sum : f32 = probs[i].iter().sum();
            for j in 0..pheromone_table[0].len(){
                probs[i][j] = probs[i][j] / column_sum;
            }
        }
        return probs
    }
}

fn euclidean(current:f32, conn1:f32, conn2:f32) -> f32{
    let mut distance:f32 = (current - conn1).powi(2);
    distance += (current - conn2).powi(2);
    distance = distance.sqrt();
    return distance;
}

fn logarithmic(current:f32, conn1:f32, conn2:f32) -> f32 {
    let mut distance = (current - conn1).ln();
    distance += (current - conn2).ln();
    distance = distance.sqrt();
    return distance
}

fn logarithmic_one(current:f32, conn:f32) -> f32{
    return (current - conn).ln().sqrt();
}

fn manhattan(current:f32, conn1:f32, conn2:f32) -> f32 {
    let mut distance:f32 = current - conn1;
    distance += current - conn2;
    return distance;
}

fn constant() -> f32 {
    return 0.;
}
fn get_argmin(nets: &Vec<f32>) -> usize {
    nets.iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, _)| index)
        .unwrap()
}

fn get_argmax(nets: &Vec<f32>) -> usize {
    nets.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, _)| index)
        .unwrap()
}

fn get_min(nets: &Vec<f32>) -> f32 {
    *nets
        .into_iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap()
}

fn get_max(nets: &Vec<f32>) -> f32 {
    *nets
        .into_iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap()
}

fn get_argmins_of_value(vecs: &Vec<f32>, res: &mut Vec<usize>, comp_value: f32) {
    vecs.iter().enumerate().for_each(|(i, v)| {
        if float_eq!(*v, comp_value, abs <= 0.000_1) {
            res.push(i);
        }
    });
}

fn calculate_greedy(pheromone_table: &Vec<Vec<f32>>) -> Vec<usize>{
    let mut greedy: Vec<usize> = Vec::with_capacity(pheromone_table.len());
    for i in 0..pheromone_table.len(){
        greedy.push(get_argmax(&pheromone_table[i]));
    }
    return greedy
}

