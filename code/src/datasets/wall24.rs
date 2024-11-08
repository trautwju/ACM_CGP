use std::fs::File;
use std::io::{self, BufRead};

pub fn get_dataset() -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let file = File::open("/data/oc-compute03/trautwju/Masterarbeit/src/datasets/Data/Klassifikation/wall24.arff").expect("Couldn't open input");
    let mut csv_file = csv::Reader::from_reader(file);
    let data_len = csv_file.records().count();
    let mut data = vec![vec![0.; 24];  data_len - 26];
    let mut labels = vec![vec![0.; 4];  data_len - 26];
    let mut mins = vec![100000000000.; 24];
    let mut maxs = vec![-10000000000.; 24];
    let mut i = 0;
    let file = File::open("/data/oc-compute03/trautwju/Masterarbeit/src/datasets/Data/Klassifikation/wall24.arff").expect("Couldn't open input");
    let reader = io::BufReader::new(file);
    for line in reader.lines() {
        let record = line.unwrap();
        if record.starts_with("@"){
            continue;
        }
        let vector:Vec<&str> = record.split(",").collect();
        for j in 0..24{
            data[i][j] = vector[j].trim().parse::<f32>().unwrap();
            if data[i][j] > maxs[j]{
                maxs[j] = data[i][j]
            }
            if data[i][j] < mins[j]{
                mins[j] = data[i][j]
            }
        }

        let label = match vector[24].trim() {
            "'Move-Forward'" => 0,
            "'Sharp-Right-Turn'" => 1,
            "'Slight-Left-Turn'" => 2,
            "'Slight-Right-Turn'" => 3,
            _ => 100,
        };
        labels[i][label] = 1.;
        i += 1;
    }
    for i in 0..data.len(){
        for j in 0..data[0].len(){
            data[i][j] = (data[i][j] - mins[j]) / (maxs[j] - mins[j])
        }
    }
    return (data, labels);
}