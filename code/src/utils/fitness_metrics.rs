use ndarray::prelude::*;
// use ndarray::Zip;

pub fn fitness_boolean(output: &ArrayView2<bool>, labels: &Array2<bool>) -> f32 {
    let mut fitness: i32 = 0;
    // Zip::from(output).and(labels).for_each(|x, y| { if x == y { fitness += 1 } });
    let nbr_correct = output ^ labels;
    let nbr_correct = !nbr_correct;
    nbr_correct.map(|x| {
        if *x == true {
            fitness += 1
        }
    });

    let fitness = 1. - (fitness as f32 / labels.len() as f32) as f32;
    return fitness;
}
