use std::fs::File;
use std::io::{self, BufRead};

pub fn get_dataset() -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let file = File::open("/data/oc-compute03/trautwju/Masterarbeit/src/datasets/Data/Klassifikation/wine_quality.arff").expect("Couldn't open input");
    let mut csv_file = csv::Reader::from_reader(file);
    let data_len = csv_file.records().count();
    let mut data = vec![vec![0.; 12];  data_len - 14];
    let mut labels = vec![vec![0.; 7];  data_len - 14];
    let mut mins = vec![100000000000.; 12];
    let mut maxs = vec![-10000000000.; 12];
    let mut i = 0;
    let file = File::open("/data/oc-compute03/trautwju/Masterarbeit/src/datasets/Data/Klassifikation/wine_quality.arff").expect("Couldn't open input");
    let reader = io::BufReader::new(file);
    for line in reader.lines() {
        let record = line.unwrap();
        if record.starts_with("@"){
            continue;
        }
        let vector:Vec<&str> = record.split(",").collect();

        data[i][0] = match vector[0] {
            "'red'" => 0.,
            "'white'" => 1.,
            _ => -1.,
        };
        for j in 1..12{
            data[i][j] = vector[j].trim().parse::<f32>().unwrap();
        }

        let label = match vector[12].trim() {
            "'3'" => 0,
            "'4'" => 1,
            "'5'" => 2,
            "'6'" => 3,
            "'7'" => 4,
            "'8'" => 5,
            "'9'" => 6,
            _ => 100,
        };
        labels[i][label] = 1.;
        for j in 0..12{
            if data[i][j] > maxs[j]{
                maxs[j] = data[i][j]
            }
            if data[i][j] < mins[j]{
                mins[j] = data[i][j]
            }
        }
        i += 1;
    }
    for i in 0..data.len(){
        for j in 0..data[0].len(){
            data[i][j] = (data[i][j] - mins[j]) / (maxs[j] - mins[j])
        }
    }
    return (data, labels);
}