use cgp::datasets::*;
use cgp::global_params::CgpParameters;
use cgp::utils::runner::Runner;
use clap::Parser;
use float_eq::float_eq;
use std::fs;
use std::fs::File;
use std::io::Write;
use plotly::common::Mode;
use plotly::{Plot, Scatter};
use plotly::color::NamedColor;
use rand::thread_rng;
use rand::seq::SliceRandom;

#[derive(Parser)]
#[clap(author, version, about, name = "testname")]
struct Args {
    #[arg(long, default_value_t = 0)]
    run_id: usize,

    #[arg(long, default_value_t = 0)]
    dataset: usize,

    #[arg(long, default_value_t = 50)]
    nbr_nodes: usize,

    #[arg(long, default_value_t = 2)]
    cgp_type: usize,

    // 0: single
    // 1: prob
    #[arg(long, default_value_t = 0)]
    mutation_type: usize,

    #[arg(long, default_value_t = -1.)]
    mutation_prob: f32,

    #[arg(long, default_value_t = 3)]
    ant_type:usize,

    #[arg(long, default_value_t = 20)]
    population_size:usize,

    #[arg(long, default_value_t = 1.)]
    tau_0:f32,

    #[arg(long, default_value_t = 1.)]
    alpha:f32,

    #[arg(long, default_value_t = 1.)]
    beta:f32,

    #[arg(long, default_value_t = 1.)]
    roh:f32,

    #[arg(long, default_value_t = 1)]
    use_global_best:i32,

    #[arg(long, default_value_t = 2)]
    distance_function:i32,

    #[arg(long, default_value_t = 0)]
    one_table:i32,

    #[arg(long, default_value_t = 1)]
    mu:usize,

    #[arg(long, default_value_t = 4)]
    lambda:usize,

    #[arg(long, default_value_t = 0)]
    elitism_type:usize,
}

fn main() {
    let args = Args::parse();

    if args.mutation_type == 1 {
        if float_eq!(args.mutation_prob, -1., abs <= 0.01) {
            panic!("Mutation prob not listed");
        }
    }

    let cgp_type = match args.cgp_type {
        0 => "Vanilla",
        1 => "mu, lambda",
        2 => "ant",
        _ => panic!(),
    };

    let _ant_type = match args.ant_type {
        0 => "acs",
        1 => "as",
        2 => "mmas",
        3 => "aslbt",
        _ => panic!(),
    };

    let (base_data, base_label) = match args.dataset {
        0 => koza::get_dataset(),
        1 => pagie::get_dataset(),
        2 => nguyen::get_dataset(),
        3 => pollution::get_dataset(),
        4 => analcatdata::get_dataset(),
        5 => abalone::get_dataset(),
        6 => bike_sharing_day::get_dataset(),
        7 => bike_sharing_hour::get_dataset(),
        8 => cal_housing::get_dataset(),
        9 => diabetes::get_dataset(),
        10 => energydata::get_dataset(),
        11 => forestfires::get_dataset(),
        12 => winequality_white::get_dataset(),
        13 => winequality_red::get_dataset(),
        14 => adult::get_dataset(),
        15 => bach::get_dataset(),
        16 => car::get_dataset(),
        17 => chronic_kidney_disease::get_dataset(),
        18 => forest::get_dataset(),
        19 => human::get_dataset(),
        20 => iris::get_dataset(),
        21 => wall24::get_dataset(),
        22 => wine_quality::get_dataset(),
        _ => panic!("Wrong dataset"),
    };

    let (mut data, mut label, mut test_data, mut test_label) = train_test_split(base_data.clone(), base_label.clone());

    let dataset_string = match args.dataset {
        0 => "koza",
        1 => "pagie",
        2 => "nguyen",
        3 => "pollution",
        4 => "analcatdata",
        5 => "abalone",
        6 => "bike_sharing_day",
        7 => "bike_sharing_hour",
        8 => "cal_housing",
        9 => "diabetes",
        10 => "energydata",
        11 => "forestfires",
        12 => "winequality_white",
        13 => "winequality_red",
        14 => "adult",
        15 => "bach",
        16 => "car",
        17 => "chronic_kidney_disease",
        18 => "forest",
        19 => "human",
        20 => "iris",
        21 => "wall24",
        22 => "wine_quality",
        _ => panic!("Wrong dataset"),
    };

    let mut graph_dir:String = String::new();
    if args.dataset < 14{
        if args.cgp_type == 1{
            graph_dir = "dataset_type_regression/dataset_id_".to_string() + &args.dataset.to_string() + "/cgp_type_" + cgp_type + "/" + &args.mu.to_string() + "," + &args.lambda.to_string()
        }else if args.cgp_type == 0{
            graph_dir = "dataset_type_classification/dataset_id_".to_string() + &args.dataset.to_string() + "/cgp_type_" + cgp_type + "/" + &args.mu.to_string() + "+" + &args.lambda.to_string()
        }else {
            if args.ant_type == 0 {
                graph_dir = "dataset_type_regression/dataset_id_".to_string() + &args.dataset.to_string() + "/cgp_type_" + cgp_type + "/ant_type_" + _ant_type + "/one_table_" + &args.one_table.to_string() + "/global_best_" + &args.use_global_best.to_string() + "/distance_function_" + &args.distance_function.to_string() + "/nbr_nodes_" + &args.nbr_nodes.to_string() + "_popsize_" + &args.population_size.to_string() + "_tau_0_" + &args.tau_0.to_string() + "_alpha_" + &args.alpha.to_string() + "_beta_" + &args.beta.to_string() + "_roh_" + &args.roh.to_string();
            } else if args.ant_type == 2 {
                graph_dir = "dataset_type_regression/dataset_id_".to_string() + &args.dataset.to_string() + "/cgp_type_" + cgp_type + "/ant_type_" + _ant_type + "/one_table_" + &args.one_table.to_string() + "/global_best_" + &args.use_global_best.to_string() + "/distance_function_" + &args.distance_function.to_string() + "/nbr_nodes_" + &args.nbr_nodes.to_string() + "_popsize_" + &args.population_size.to_string() + "beta_" + &args.beta.to_string() + "_roh_" + &args.roh.to_string();
            } else {
                graph_dir = "dataset_type_regression/dataset_id_".to_string() + &args.dataset.to_string() + "/cgp_type_" + cgp_type + "/ant_type_" + _ant_type + "/one_table_" + &args.one_table.to_string() + "/global_best_" + &args.use_global_best.to_string() + "/distance_function_" + &args.distance_function.to_string() + "/nbr_nodes_" + &args.nbr_nodes.to_string() + "_popsize_" + &args.population_size.to_string() + "tau_0_" + &args.tau_0.to_string() + "_beta_" + &args.beta.to_string() + "_roh_" + &args.roh.to_string();
            }
        }
    }else{
        if args.cgp_type == 1{
            graph_dir = "dataset_type_classification/dataset_id_".to_string() + &args.dataset.to_string() + "/cgp_type_" + cgp_type + "/" + &args.mu.to_string() + "," + &args.lambda.to_string()
        }else if args.cgp_type == 0{
            graph_dir = "dataset_type_classification/dataset_id_".to_string() + &args.dataset.to_string() + "/cgp_type_" + cgp_type + "/" + &args.mu.to_string() + "+" + &args.lambda.to_string()
        }else{
            if args.ant_type == 0{
                graph_dir = "dataset_type_classification/dataset_id_".to_string() + &args.dataset.to_string() + "/cgp_type_" + cgp_type + "/ant_type_" + _ant_type + "/one_table_" + &args.one_table.to_string() + "/global_best_" + &args.use_global_best.to_string() + "/distance_function_" + &args.distance_function.to_string() + "/nbr_nodes_" + &args.nbr_nodes.to_string() + "_popsize_" + &args.population_size.to_string() + "tau_0_" + &args.tau_0.to_string() + "_alpha_" + &args.alpha.to_string() + "_beta_" + &args.beta.to_string() + "_roh_" + &args.roh.to_string();
            }else if args.ant_type == 2{
                graph_dir = "dataset_type_classification/dataset_id_".to_string() + &args.dataset.to_string() + "/cgp_type_" + cgp_type + "/ant_type_" + _ant_type + "/one_table_" + &args.one_table.to_string() + "/global_best_" + &args.use_global_best.to_string() + "/distance_function_" + &args.distance_function.to_string() + "/nbr_nodes_" + &args.nbr_nodes.to_string() + "_popsize_" + &args.population_size.to_string() + "beta_" + &args.beta.to_string() + "_roh_" + &args.roh.to_string();
            }else{
                graph_dir = "dataset_type_classification/dataset_id_".to_string() + &args.dataset.to_string() + "/cgp_type_" + cgp_type + "/ant_type_" + _ant_type + "/one_table_" + &args.one_table.to_string() + "/global_best_" + &args.use_global_best.to_string() + "/distance_function_" + &args.distance_function.to_string() + "/nbr_nodes_" + &args.nbr_nodes.to_string() + "_popsize_" + &args.population_size.to_string() + "tau_0_" + &args.tau_0.to_string() + "_beta_" + &args.beta.to_string() + "_roh_" + &args.roh.to_string();
            }
        }
    }


    fs::create_dir_all(graph_dir.clone()).expect("cannot create dir");
    let mut params = CgpParameters::default();

    params.tau_0 = args.tau_0;
    if args.ant_type == 2{
        params.tau_0 = 100000.;
    }

    params.alpha = args.alpha;
    params.beta = args.beta;
    params.roh = args.roh;
    params.distance_function = args.distance_function;

    if args.cgp_type == 2 {
        params.lambda = args.population_size - 1;
        params.mu = 1;
    }else{
        params.mu = args.mu;
        params.lambda = args.lambda;
    }


    params.graph_width = args.nbr_nodes;

    let nbr_inputs = data[0].len();
    let nbr_outputs = label[0].len();

    params.nbr_inputs = nbr_inputs;
    params.nbr_outputs = nbr_outputs;

    let mut use_global_best_ant = true;
    if args.use_global_best == 0{
        use_global_best_ant = false;
    }

    // let stdout = std::io::stdout();
    // let mut lock = stdout.lock();

    let mut replace_parents = false;
    if args.cgp_type == 1 {
        replace_parents = true;
    }
    let mut fitness_vals = Vec::new();
    let mut mae_vals = Vec::new();
    let mut best_fitness_vals = Vec::new();
    let mut best_mae_vals = Vec::new();
    let mut active_nodes = Vec::new();
    let mut positional_bias:Vec<f32> = vec![0.;params.nbr_inputs + params.graph_width];
    let mut end_at = Vec::new();
    let mut iterations_till_best = Vec::new();
    let mut total_func_evals_till_best:Vec<f32> = Vec::new();
    let mut final_fitnesses = Vec::new();
    let mut output = File::create(graph_dir.clone() + "/run_" + &args.run_id.to_string() +  "_fitness_and_func_evals").expect("cannot create file");

    for z in 0..10 {
        (data, label, test_data, test_label) = train_test_split(base_data.clone(), base_label.clone());
        let mut i = 0;
        // let mut prev_it: Vec<f32> = Vec::from(runner.get_best_fitness());
        let mut func_evals = 0;
        let mut pos_best_fitness = 0;
        let mut best_fitness = 0.;
        let mut best_mae = 0.;
        let mut last_fitness_change = 0;
        let mut runner = Runner::new(
            params.clone(),
            data.clone(),
            label.clone(),
            args.mutation_type,
            args.mutation_prob,
            use_global_best_ant,
            args.elitism_type
        );
        let mut func_evals_best = 0;
        if args.cgp_type != 2 {
            runner = Runner::new(
                params.clone(),
                data.clone(),
                label.clone(),
                args.mutation_type,
                args.mutation_prob,
                use_global_best_ant,
                args.elitism_type,
            );
            runner.new_parents_by_method(replace_parents);
            loop {
                func_evals += params.lambda;
                i += 1;
                let fitness = runner.learn_step(replace_parents);
                if i == 1 {
                    best_fitness = fitness;
                    if args.dataset < 14 {
                        best_mae = runner.get_best_mae();
                    }
                }
                if fitness < best_fitness {
                    best_fitness = fitness;
                    pos_best_fitness = i;
                    func_evals_best = func_evals;
                    if args.dataset < 14 {
                        best_mae = runner.get_best_mae();
                    }
                }

                let current_num_active_nodes = runner.get_num_active_nodes();
                for j in 0..current_num_active_nodes.len(){
                    active_nodes.push(current_num_active_nodes[j]);
                }
                let current_active_nodes = runner.get_all_active_nodes();
                for k in 0..current_active_nodes.len() {
                    if current_active_nodes[k] < params.nbr_inputs + params.graph_width {
                        positional_bias[current_active_nodes[k]] += 1.;
                    }
                }
                best_fitness_vals.push(best_fitness);
                let mut current_maes = Vec::new();
                if args.dataset < 14 {
                    best_mae_vals.push(best_mae);
                    current_maes = runner.get_maes();
                }
                let current_fitnesses = runner.get_fitnesses();
                for j in 0..(params.mu + params.lambda){
                    fitness_vals.push(current_fitnesses[j]);
                    if args.dataset < 14{
                        mae_vals.push(current_maes[j]);
                    }
                }


                // if float_eq!(runner.get_best_fitness()[0], 0., abs <= 0.000_1) {  // for multiple parents
                if float_eq!(runner.get_best_fitness(), 0., abs <= 0.000_1) {
                    // for single parent
                    final_fitnesses.push(runner.evaluate_chromosomes_cgp(test_data.clone(), test_label.clone()));
                    iterations_till_best.push(pos_best_fitness as f32);
                    total_func_evals_till_best.push(func_evals_best as f32);
                    end_at.push(i);
                    writeln!(output, "Fitness_{}: {}", z, final_fitnesses[z]).expect("unable to write");
                    writeln!(output, "Iterations_{}: {}", z, iterations_till_best[z]).expect("unable to write");
                    break;
                }

                if i > 100000 {
                    final_fitnesses.push(runner.evaluate_chromosomes_cgp(test_data.clone(), test_label.clone()));
                    iterations_till_best.push(pos_best_fitness as f32);
                    total_func_evals_till_best.push(func_evals_best as f32);
                    end_at.push(i);
                    writeln!(output, "Fitness_{}: {}", z, final_fitnesses[z]).expect("unable to write");
                    writeln!(output, "Iterations_{}: {}", z, iterations_till_best[z]).expect("unable to write");
                    break;
                }
            }
        } else {
            runner = Runner::new(
                params.clone(),
                data.clone(),
                label.clone(),
                args.mutation_type,
                args.mutation_prob,
                use_global_best_ant,
                args.elitism_type,
            );
            if args.one_table == 0 {
                let mut pheromone_table_functions = vec![vec![params.tau_0; 14]; params.graph_width];
                let mut pheromone_table_connections = vec![vec![0.; (params.nbr_inputs + params.graph_width).pow(2)]; params.graph_width + params.nbr_outputs];
                runner.initialize_pheromone_table(&mut pheromone_table_connections);
                if args.ant_type == 2 {
                    runner.initialize_mmas(&mut pheromone_table_connections);
                }
                loop {
                    i += 1;
                    func_evals += params.mu + params.lambda;
                    let current_fitness = runner.ant_learn_two_tables(&mut pheromone_table_functions, &mut pheromone_table_connections, args.ant_type);
                    if i == 1 {
                        best_fitness = current_fitness;
                        if args.dataset < 14 {
                            best_mae = runner.get_best_mae();
                        }
                    }
                    if best_fitness > current_fitness {
                        best_fitness = current_fitness;
                        func_evals_best = func_evals;
                        if args.dataset < 14 {
                            best_mae = runner.get_best_mae();
                        }
                        pos_best_fitness = i;
                        last_fitness_change = 0;
                    }
                    let current_num_active_nodes = runner.get_num_active_nodes();
                    for j in 0..current_num_active_nodes.len(){
                        active_nodes.push(current_num_active_nodes[j]);
                    }
                    let current_active_nodes = runner.get_all_active_nodes();
                    for k in 0..current_active_nodes.len() {
                        if current_active_nodes[k] < params.nbr_inputs + params.graph_width {
                            positional_bias[current_active_nodes[k]] += 1.;
                        }
                    }
                    best_fitness_vals.push(best_fitness);
                    let mut current_maes = Vec::new();
                    if args.dataset < 14 {
                        best_mae_vals.push(best_mae);
                        current_maes = runner.get_maes();
                    }
                    let current_fitnesses = runner.get_fitnesses();
                    for j in 0..(params.mu + params.lambda){
                        fitness_vals.push(current_fitnesses[j]);
                        if args.dataset < 14{
                            mae_vals.push(current_maes[j]);
                        }
                    }
                    if (i > 10000) || (last_fitness_change > 500) {
                        final_fitnesses.push(runner.evaluate_chromosomes_cgp(test_data.clone(), test_label.clone()));
                        /*println!("{}", final_fitness);*/
                        end_at.push(i);
                        iterations_till_best.push(pos_best_fitness as f32);
                        total_func_evals_till_best.push(func_evals_best as f32);
                        writeln!(output, "Fitness_{}: {}", z, final_fitnesses[z]).expect("unable to write");
                        writeln!(output, "Iterations_{}: {}", z, iterations_till_best[z]).expect("unable to write");
                        break;
                    }
                    if float_eq!(runner.get_best_fitness(), 0., abs <= 0.000_1) {
                        final_fitnesses.push(runner.evaluate_chromosomes_cgp(test_data.clone(), test_label.clone()));
                        /*println!("{}", final_fitness);*/
                        end_at.push(i);
                        iterations_till_best.push(pos_best_fitness as f32);
                        total_func_evals_till_best.push(func_evals_best as f32);
                        writeln!(output, "Fitness_{}: {}", z, final_fitnesses[z]).expect("unable to write");
                        writeln!(output, "Iterations_{}: {}", z, iterations_till_best[z]).expect("unable to write");
                        break;
                    }
                    last_fitness_change += 1;
                }
            } else {
                let mut pheromone_table = vec![vec![0.; (params.nbr_inputs + params.graph_width).pow(2) * 4 + (params.nbr_inputs + params.graph_width) * 10]; params.graph_width + params.nbr_outputs];
                runner.init_one_pheromone_table(&mut pheromone_table);
                loop {
                    i += 1;
                    func_evals += params.mu + params.lambda;
                    let current_fitness = runner.ant_learn_one_table(&mut pheromone_table, args.ant_type);
                    if i == 1 {
                        best_fitness = current_fitness;
                        if args.dataset < 14 {
                            best_mae = runner.get_best_mae();
                        }
                    }
                    if best_fitness > current_fitness {
                        best_fitness = current_fitness;
                        if args.dataset < 14 {
                            best_mae = runner.get_best_mae();
                        }
                        last_fitness_change = 0;
                    }
                    let current_num_active_nodes = runner.get_num_active_nodes();
                    for j in 0..current_num_active_nodes.len(){
                        active_nodes.push(current_num_active_nodes[j]);
                    }
                    let current_active_nodes = runner.get_all_active_nodes();
                    for k in 0..current_active_nodes.len() {
                        if current_active_nodes[k] < params.nbr_inputs + params.graph_width {
                            positional_bias[current_active_nodes[k]] += 1.;
                        }
                    }
                    best_fitness_vals.push(best_fitness);
                    let mut current_maes = Vec::new();
                    if args.dataset < 14 {
                        best_mae_vals.push(best_mae);
                        current_maes = runner.get_maes();
                    }
                    let current_fitnesses = runner.get_fitnesses();
                    for j in 0..(params.mu + params.lambda){
                        fitness_vals.push(current_fitnesses[j]);
                        if args.dataset < 14{
                            mae_vals.push(current_maes[j]);
                        }
                    }
                    if (i > 100000) || (last_fitness_change > 500) {
                        println!("{}", runner.best_ant_one_table(&pheromone_table, test_data.clone(), test_label.clone()));
                        end_at.push(i);
                        iterations_till_best.push(pos_best_fitness as f32);
                        total_func_evals_till_best.push(func_evals_best as f32);
                        writeln!(output, "Fitness_{}: {}", z, final_fitnesses[z]).expect("unable to write");
                        writeln!(output, "Iterations_{}: {}", z, iterations_till_best[z]).expect("unable to write");
                        break;
                    }
                    if float_eq!(runner.get_best_fitness(), 0., abs <= 0.000_1) {
                        println!("{}", runner.best_ant_one_table(&pheromone_table, test_data.clone(), test_label.clone()));
                        end_at.push(i);
                        iterations_till_best.push(pos_best_fitness as f32);
                        total_func_evals_till_best.push(func_evals_best as f32);
                        writeln!(output, "Fitness_{}: {}", z, final_fitnesses[z]).expect("unable to write");
                        writeln!(output, "Iterations_{}: {}", z, iterations_till_best[z]).expect("unable to write");
                        break;
                    }
                    last_fitness_change += 1;
                }
            }
        }
    }

    let mut min_iters = 10000000000;
    for j in 0..end_at.len(){
        if end_at[j] < min_iters{
            min_iters = end_at[j];
        }
    }
    let end_at_clone = end_at.clone();
    for j in 0..end_at.len()-1{
        for i in j+1..end_at.len(){
            end_at[i] = end_at[i] + end_at_clone[j];
        }
    }


    let mut avg_fitness:Vec<f32> = Vec::new();
    let mut std_dev_fitnesses:Vec<f32> = Vec::new();
    let mut avg_mae:Vec<f32> = Vec::new();
    let mut std_dev_maes:Vec<f32> = Vec::new();
    let mut avg_active_nodes_all:Vec<f32> = Vec::new();
    let mut std_dev_active_nodes_all:Vec<f32> = Vec::new();
    let mut best_fitness_iteration:Vec<f32> = Vec::new();
    let mut best_global_fitnesses:Vec<f32> = Vec::new();
    let popsize = params.mu + params.lambda;
    if args.dataset < 14 {
        for j in 0..min_iters {
            let mut fitness: Vec<f32> = Vec::new();
            let mut mae: Vec<f32> = Vec::new();
            let mut nodes: Vec<f32> = Vec::new();
            let mut best_fitness: Vec<f32> = Vec::new();
            for k in 0..end_at.len() {
                if k > 0{
                    best_fitness.push(best_fitness_vals[end_at[k-1] + j]);
                }else{
                    best_fitness.push(best_fitness_vals[j])
                }
                for l in 0..popsize {
                    if k > 0 {
                        fitness.push(fitness_vals[end_at[k - 1] * popsize + j * popsize + l]);
                        mae.push(mae_vals[end_at[k - 1] * popsize + j * popsize + l]);
                        nodes.push(active_nodes[end_at[k - 1] * popsize + j * popsize + l] as f32);
                    } else {
                        fitness.push(fitness_vals[j * popsize + l]);
                        mae.push(mae_vals[j * popsize + l]);
                        nodes.push(active_nodes[j * popsize + l] as f32);
                    }
                }
            }
            let avg_best_fitness: f32 = best_fitness.iter().sum();
            best_global_fitnesses.push(avg_best_fitness / best_fitness.len() as f32);
            fitness.retain(|&x| x != f32::INFINITY);
            mae.retain(|&x| x != f32::INFINITY);
            if fitness.len() == 0 {
                fitness.push(0.);
            }
            if mae.len() == 0 {
                mae.push(0.);
            }
            best_fitness_iteration.push(*fitness.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap());
            let avg_fit: f32 = fitness.iter().sum();
            let curr_avg_mae: f32 = mae.iter().sum();
            let avg_node: f32 = nodes.iter().sum();
            avg_fitness.push(avg_fit / fitness.len() as f32);
            avg_mae.push(curr_avg_mae / mae.len() as f32);
            avg_active_nodes_all.push(avg_node / nodes.len() as f32);

            let mut std_dev_fitness: f32 = 0.;
            let mut std_dev_mae: f32 = 0.;
            let mut std_dev_node: f32 = 0.;
            for k in 0..fitness.len() {
                std_dev_fitness += (fitness[k] - avg_fitness[j]).powi(2);
                std_dev_mae += (mae[k] - avg_mae[j]).powi(2);
                std_dev_node += (nodes[k] - avg_active_nodes_all[j]).powi(2);
            }
            std_dev_fitness /= fitness.len() as f32;
            std_dev_mae /= mae.len() as f32;
            std_dev_node /= nodes.len() as f32;

            std_dev_fitness = std_dev_fitness.powf(0.5);
            std_dev_mae = std_dev_mae.powf(0.5);
            std_dev_node = std_dev_node.powf(0.5);

            std_dev_fitnesses.push(std_dev_fitness);
            std_dev_maes.push(std_dev_mae);
            std_dev_active_nodes_all.push(std_dev_node);
        }
    } else {
        for j in 0..min_iters {
            let mut fitness: Vec<f32> = Vec::new();
            let mut nodes: Vec<f32> = Vec::new();
            let mut best_fitness: Vec<f32> = Vec::new();
            for k in 0..end_at.len() {
                if k > 0{
                    best_fitness.push(best_fitness_vals[end_at[k-1] + j]);
                }else{
                    best_fitness.push(best_fitness_vals[j])
                }
                for l in 0..popsize {
                    if k > 0 {
                        fitness.push(fitness_vals[end_at[k - 1] * popsize + j * popsize + l]);
                        nodes.push(active_nodes[end_at[k - 1] * popsize + j * popsize + l] as f32);
                    } else {
                        fitness.push(fitness_vals[j * popsize + l]);
                        nodes.push(active_nodes[j * popsize + l] as f32);
                    }
                }
            }
            let avg_best_fitness: f32 = best_fitness.iter().sum();
            best_global_fitnesses.push(avg_best_fitness / best_fitness.len() as f32);
            fitness.retain(|&x| x != f32::INFINITY);
            if fitness.len() == 0 {
                fitness.push(0.);
            }
            let avg_fit: f32 = fitness.iter().sum();
            let avg_node: f32 = nodes.iter().sum();
            avg_fitness.push(avg_fit / fitness.len() as f32);
            avg_active_nodes_all.push(avg_node / nodes.len() as f32);

            let mut std_dev_fitness: f32 = 0.;
            let mut std_dev_node: f32 = 0.;
            for k in 0..fitness.len() {
                std_dev_fitness += (fitness[k] - avg_fitness[j]).powi(2);
                std_dev_node += (nodes[k] - avg_active_nodes_all[j]).powi(2);
            }
            std_dev_fitness /= fitness.len() as f32;
            std_dev_node /= nodes.len() as f32;

            std_dev_fitness = std_dev_fitness.powf(0.5);
            std_dev_node = std_dev_node.powf(0.5);

            std_dev_fitnesses.push(std_dev_fitness);
            std_dev_active_nodes_all.push(std_dev_node);
        }
    }

    let mut color = NamedColor::Blue;
    if args.cgp_type == 0{
        if params.mu != 1 || params.lambda != 4{
            color = NamedColor::Orange;
        }
    }else if args.cgp_type == 1 {
        color = NamedColor::Green;
    }else{
        if args.ant_type == 0{
            color = NamedColor::Black;
        }else if args.ant_type == 1{
            color = NamedColor::Yellow;
        }else if args.ant_type == 2{
            color = NamedColor::Purple;
        }else if args.ant_type == 3{
            color = NamedColor::Red;
        }
    }

    if args.dataset < 14{
        let mut plot = Plot::new();
        let trace = Scatter::new(Vec::from_iter(0..avg_mae.len()), avg_mae)
            .mode(Mode::Lines).name("average mae").line(plotly::common::Line::new().color(color));
        plot.add_trace(trace);

        let dir = graph_dir.clone() + "/run_" + &args.run_id.to_string() +  "_avg_mae.html";
        plot.write_html(dir);

        let mut plot = Plot::new();
        let trace = Scatter::new(Vec::from_iter(0..std_dev_maes.len()), std_dev_maes)
            .mode(Mode::Lines).name("std_dev_mae").line(plotly::common::Line::new().color(color));
        plot.add_trace(trace);

        let dir = graph_dir.clone() + "/run_" + &args.run_id.to_string() +  "_std_dev_mae.html";
        plot.write_html(dir);
    }

    let trace = Scatter::new(Vec::from_iter(0..best_fitness_iteration.len()), best_fitness_iteration)
        .mode(Mode::Lines).name("iteration best fitness").line(plotly::common::Line::new().color(color));
    let mut plot = Plot::new();
    plot.add_trace(trace);
    let dir = graph_dir.clone() + "/run_" + &args.run_id.to_string() +  "_iteration_best_fitness.html";
    plot.write_html(dir);

    let trace = Scatter::new(Vec::from_iter(0..best_global_fitnesses.len()), best_global_fitnesses)
        .mode(Mode::Lines).name("global best fitness").line(plotly::common::Line::new().color(color));
    let mut plot = Plot::new();
    plot.add_trace(trace);
    let dir = graph_dir.clone() + "/run_" + &args.run_id.to_string() +  "_global_best_fitness.html";
    plot.write_html(dir);

    let mut plot = Plot::new();
    let trace = Scatter::new(Vec::from_iter(0..avg_fitness.len()), avg_fitness)
        .mode(Mode::Lines).name("average fitness").line(plotly::common::Line::new().color(color));
    plot.add_trace(trace);
    let dir = graph_dir.clone() + "/run_" + &args.run_id.to_string() +  "_avg_fitness.html";
    plot.write_html(dir);

    let mut plot = Plot::new();
    let trace = Scatter::new(Vec::from_iter(0..std_dev_fitnesses.len()), std_dev_fitnesses)
        .mode(Mode::Lines).name("std_dev fitness").line(plotly::common::Line::new().color(color));
    plot.add_trace(trace);
    let dir = graph_dir.clone() + "/run_" + &args.run_id.to_string() +  "_std_dev_fitness.html";
    plot.write_html(dir);

    let trace = Scatter::new(Vec::from_iter(0..avg_active_nodes_all.len()), avg_active_nodes_all)
        .mode(Mode::Lines).name("average active nodes").line(plotly::common::Line::new().color(color));
    let mut plot = Plot::new();
    plot.add_trace(trace);
    let dir = graph_dir.clone() + "/run_" + &args.run_id.to_string() +  "_avg_active_nodes.html";
    plot.write_html(dir);

    let mut max:usize = end_at.iter().sum();
    max *= popsize;
    positional_bias = positional_bias.iter().map(|x| x / max as f32).collect();
    let trace = Scatter::new(Vec::from_iter(0..positional_bias.len()), positional_bias)
        .mode(Mode::Lines).name("positional bias").line(plotly::common::Line::new().color(color));
    let mut plot = Plot::new();
    plot.add_trace(trace);
    plot.write_html(graph_dir.clone() + "/run_" + &args.run_id.to_string() +  "_positional_bias.html");

    let mut mean_fitness:f32 = final_fitnesses.iter().sum();
    mean_fitness /= final_fitnesses.len() as f32;

    let mut mean_iterations:f32 = iterations_till_best.iter().sum();
    mean_iterations /= iterations_till_best.len() as f32;

    let mut mean_evals:f32 = total_func_evals_till_best.iter().sum();
    mean_evals /= total_func_evals_till_best.len() as f32;

    let mut std_dev_fitness:f32 = 0.;
    let mut std_dev_iter:f32 = 0.;
    let mut std_dev_eval:f32 = 0.;
    for j in 0..final_fitnesses.len(){
        std_dev_fitness += (final_fitnesses[j] - mean_fitness).powi(2);
        std_dev_iter += (iterations_till_best[j] - mean_iterations).powi(2);
        std_dev_eval += (total_func_evals_till_best[j] - mean_evals).powi(2);
    }
    std_dev_fitness /= final_fitnesses.len() as f32;
    std_dev_iter /= iterations_till_best.len() as f32;
    std_dev_eval /= total_func_evals_till_best.len() as f32;

    std_dev_fitness = std_dev_fitness.powf(0.5);
    std_dev_iter = std_dev_iter.powf(0.5);
    std_dev_eval = std_dev_eval.powf(0.5);

    writeln!(output, "Fitness: {}", mean_fitness).expect("unable to write");
    writeln!(output, "Fitness_std_dev: {}", std_dev_fitness).expect("unable to write");
    writeln!(output, "Iterations: {}", mean_iterations).expect("unable to write");
    writeln!(output, "Iterations_std_dev: {}", std_dev_iter).expect("unable to write");
    writeln!(output, "Func_evals: {}", mean_evals).expect("unable to write");
    writeln!(output, "Func_evals_std_dev: {}", std_dev_eval).expect("unable to write");
    writeln!(output, "Fastest fund solution: {}", min_iters).expect("unable to write");
}


pub fn train_test_split(data: Vec<Vec<f32>>, label: Vec<Vec<f32>>) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>){
    let train_size = (data.len() as f32 * 0.8).round() as usize;
    let test_size = data.len() - train_size;

    let mut train_data = vec![vec![0.; data[0].len()]; train_size];
    let mut train_labels = vec![vec![0.; label[0].len()]; train_size];
    let mut test_data = vec![vec![0.; data[0].len()]; test_size];
    let mut test_labels = vec![vec![0.; label[0].len()]; test_size];

    let mut random_split:Vec<_> = (0..label.len()).collect();
    random_split.shuffle(&mut thread_rng());
    for i in 0..train_size{
        train_data[i] = data[random_split[i]].clone();
        train_labels[i] = label[random_split[i]].clone();
    }
    for i in 0..test_size{
        test_data[i] = data[random_split[train_size + i]].clone();
        test_labels[i] = label[random_split[train_size + i]].clone();
    }
    return (train_data, train_labels, test_data, test_labels);
}