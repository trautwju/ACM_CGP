use std::fs::File;
use std::io::{self, BufRead};

pub fn get_dataset() -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let file = File::open("/data/oc-compute03/trautwju/Masterarbeit/src/datasets/Data/Klassifikation/car.data").expect("Couldn't open input");
    let mut csv_file = csv::Reader::from_reader(file);
    let data_len = csv_file.records().count();
    let mut data = vec![vec![0.; 6];  data_len + 1];
    let mut mins = vec![100000000000.; 6];
    let mut maxs = vec![-10000000000.; 6];
    let mut labels = vec![vec![0.; 4];  data_len + 1];
    let mut i = 0;
    let file = File::open("/data/oc-compute03/trautwju/Masterarbeit/src/datasets/Data/Klassifikation/car.data").expect("Couldn't open input");
    let reader = io::BufReader::new(file);
    for line in reader.lines() {
        let record = line.unwrap();
        let vector:Vec<&str> = record.split(",").collect();
        data[i][0] = match vector[0].trim(){
            "vhigh" => 0.0,
            "high" => 1.0,
            "med" => 2.0,
            "low" => 3.0,
            _ => -1.,
        };

        data[i][1] = match vector[1].trim(){
            "vhigh" => 0.0,
            "high" => 1.0,
            "med" => 2.0,
            "low" => 3.0,
            _ => -1.,
        };

        data[i][2] = match vector[2].trim(){
            "5more" => 5.0,
            _ => vector[2].trim().parse::<f32>().unwrap(),
        };

        data[i][3] = match vector[3].trim(){
            "more" => 6.0,
            _ => vector[3].trim().parse::<f32>().unwrap(),
        };

        data[i][4] = match vector[4].trim(){
            "small" => 0.0,
            "med" => 1.0,
            "big" => 2.0,
            _ => -1.,
        };

        data[i][5] = match vector[5].trim(){
            "low" => 0.0,
            "med" => 1.0,
            "high" => 2.0,
            _ => -1.,
        };

        let label = match vector[6].trim(){
            "unacc" => 0,
            "acc" => 1,
            "good" => 2,
            "vgood" => 3,
            _ => 100,
        };
        labels[i][label] = 1.0;

        for j in 0..6{
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