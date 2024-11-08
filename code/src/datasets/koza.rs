use rand::{thread_rng, Rng};

pub fn get_dataset() -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let data = [(); 20].map(|_| thread_rng().gen_range(-1.0..1.0) as f32).to_vec();
    let mut labels = Vec::new();
    for i in 0..data.len(){
        labels.push([data[i].powi(6) - 2. * data[i].powi(4) + data[i].powi(2)].to_vec())
    }
    let mut final_data = Vec::new();
    for i in 0..data.len(){
        final_data.push([data[i]].to_vec())
    }
    return (final_data, labels)
}