use crate::global_params::CgpParameters as g_params;
use crate::cgp_es::chromosome::Chromosome;
use float_eq::float_eq;
use rand::seq::SliceRandom;
use std::fmt::{Display, Formatter};

pub struct Runner {
    params: g_params,
    data: Vec<Vec<f32>>,
    label: Vec<Vec<f32>>,
    chromosomes: Vec<Chromosome>,
    fitness_vals_sorted: Vec<f32>,
    fitness_vals: Vec<f32>,
    parent_ids: Vec<usize>,
}

impl Display for Runner {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        for p in &self.parent_ids {
            write!(f, "Parents: {}", self.chromosomes[*p])?;
        }
        writeln!(f, "Fitnesses: {:?}", self.fitness_vals)
    }
}

impl Runner {
    pub fn new(params: g_params, data: Vec<Vec<f32>>, label: Vec<Vec<f32>>) -> Self {
        let mut chromosomes: Vec<Chromosome> = Vec::with_capacity(params.mu + params.lambda);
        let mut fitness_vals: Vec<f32> = Vec::with_capacity(params.mu + params.lambda);

        for _ in 0..(params.mu + params.lambda) {
            let mut chromosome = Chromosome::new(params.clone());
            let fitness = chromosome.evaluate(&data, &label);
            fitness_vals.push(fitness);

            chromosomes.push(chromosome);
        }

        // Get sorted fitness vals
        let mut fitness_vals_sorted = fitness_vals.clone();
        fitness_vals_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Reverse fitness_vals_sorted to pop the best fitness first
        let mut temp_fitness_vals_sorted = fitness_vals_sorted.clone();
        temp_fitness_vals_sorted.reverse();
        temp_fitness_vals_sorted.dedup();

        let mut parent_ids: Vec<usize> = vec![];
        while parent_ids.len() < params.mu {
            let current_best_fitness_val = temp_fitness_vals_sorted.pop();

            get_argmins_of_value(
                &fitness_vals,
                &mut parent_ids,
                current_best_fitness_val.unwrap(),
            );
        }

        parent_ids.truncate(params.mu);

        assert_eq!(parent_ids.len(), params.mu);

        Self {
            params,
            data,
            label,
            chromosomes,
            fitness_vals,
            fitness_vals_sorted,
            parent_ids,
        }
    }

    pub fn learn_step(&mut self) {
        // self.crossover();

        self.mutate_chromosomes();

        self.eval_chromosomes();

        self.new_parent_by_neutral_search();
        // exit(1);
    }

    fn new_parent_by_neutral_search(&mut self) {
        // Get mu - many best fitness vals
        let mut sorted_fitness_vals = self.fitness_vals_sorted.clone();
        // remove duplicates
        sorted_fitness_vals.dedup();

        let mut new_parent_ids: Vec<usize> = vec![];
        for (i, current_best_fitness_val) in sorted_fitness_vals.iter().enumerate() {
            let mut min_keys: Vec<usize> = Vec::with_capacity(self.params.mu + self.params.lambda);

            get_argmins_of_value(&self.fitness_vals, &mut min_keys, *current_best_fitness_val);

            if i == 0 {
                // In the first iteration; if the Nbr of min_keys <= mu, then these all have to become
                // parents. Independent of neutral search
                if min_keys.len() <= self.params.mu {
                    new_parent_ids.extend(min_keys);

                    if new_parent_ids.len() == self.params.mu {
                        break;
                    }

                    continue;
                }
            }
            if min_keys.len() == self.params.mu - new_parent_ids.len() {
                new_parent_ids.extend(min_keys);
                break;
            }

            // let mut min_keys_without_parent_ids = vect_difference(&min_keys, &self.parent_ids);
            for p_id in &self.parent_ids {
                let index = min_keys.iter().position(|x| *x == *p_id);
                if index.is_some() {
                    min_keys.remove(index.unwrap());
                }
            }
            //     todo check bugfree?
            min_keys.truncate(self.params.mu - new_parent_ids.len());
            new_parent_ids.extend(min_keys);

            // if enough parents are generated, break
            if new_parent_ids.len() >= self.params.mu {
                break;
            }
        }
        self.parent_ids = new_parent_ids;
    }

    fn mutate_chromosomes(&mut self) {
        // mutate new chromosomes; do not mutate parents
        for i in 0..(self.params.mu + self.params.lambda) {
            if self.parent_ids.contains(&i) {
                continue;
            }

            let rand_parent_id = self.parent_ids.choose(&mut rand::thread_rng()).unwrap();
            self.chromosomes[i] = self.chromosomes[*rand_parent_id].clone();
            self.chromosomes[i].mutate_single();
            assert_ne!(i, *rand_parent_id);
        }
    }

    fn eval_chromosomes(&mut self) {
        for i in 0..(self.params.mu + self.params.lambda) {
            if !self.parent_ids.contains(&i) {
                let fitness = self.chromosomes[i].evaluate(&self.data, &self.label);
                self.fitness_vals[i] = fitness;

                // // TODO CHECK HERE
                // self.chromosomes[i].reorder();
                // let fitness = self.chromosomes[i].evaluate(&self.data, &self.label);
                // assert_eq!(self.fitness_vals[i], fitness);
            }
        }

        let mut best_fitnesses_sorted = self.fitness_vals.clone();
        best_fitnesses_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        self.fitness_vals_sorted = best_fitnesses_sorted;
    }

    pub fn get_best_fitness(&self) -> &[f32] {
        return &self.fitness_vals_sorted[..self.params.mu];
        // return &self.fitness_vals_sorted;
    }

    // fn crossover(&self) {
    //     for i in 0..(self.params.mu + self.params.lambda) {}
    // }

    // fn one_point_crossover
}

fn get_argmins_of_value(vecs: &Vec<f32>, res: &mut Vec<usize>, comp_value: f32) {
    vecs.iter().enumerate().for_each(|(i, v)| {
        if float_eq!(*v, comp_value, abs <= 0.000_1) {
            res.push(i);
        }
    });
}
