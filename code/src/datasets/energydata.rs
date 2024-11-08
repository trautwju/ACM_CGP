use std::fs::File;
use std::io::{self, BufRead};

pub fn get_dataset() -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let file = File::open("/data/oc-compute03/trautwju/Masterarbeit/src/datasets/Data/Regression/energydata_complete.csv").expect("Couldn't open input");
    let mut csv_file = csv::Reader::from_reader(file);
    let data_len = csv_file.records().count();
    let mut data = vec![vec![0.; 28]; data_len];
    let mut labels = vec![vec![0.; 1]; data_len];
    let mut standardisation_labels:Vec<f32> = vec![0.; data_len + 1];
    let mut mins = vec![100000000000.; 28];
    let mut maxs = vec![-10000000000.; 28];
    let mut i = 0;
    let file = File::open("/data/oc-compute03/trautwju/Masterarbeit/src/datasets/Data/Regression/energydata_complete.csv").expect("Couldn't open input");
    let reader = io::BufReader::new(file);
    for line in reader.lines() {
        let record = line.unwrap();
        let vector:Vec<&str> = record.split(",").collect();
        if vector[0] == "\"date\""{
            continue;
        }
        let date:Vec<_> = vector[0].split("-").collect();
        let time:Vec<_>= date[2].split(":").collect();
        let month = date[1].parse::<f32>().unwrap();
        let day_str:Vec<_> = time[0].split(" ").collect();
        let mut day = day_str[0].parse::<f32>().unwrap() - 11.;
        let mut hour = day_str[1].parse::<f32>().unwrap();
        let minute = time[1].parse::<f32>().unwrap();
        if month > 1.{
            day += 31.;
        }
        if month > 2.{
            day += 29.;
        }
        if month > 3.{
            day += 31.;
        }
        if month > 4.{
            day += 30.;
        }
        hour = hour + (minute / 60.);
        hour = hour / 24.;
        day += hour;
        data[i][0] = day;
        if data[i][0] > maxs[0]{
            maxs[0] = data[i][0]
        }
        if data[i][0] < mins[0]{
            mins[0] = data[i][0]
        }
        for j in 2..29{
            let mut chars = vector[j].chars();
            chars.next();
            chars.next_back();
            let string = chars.as_str();
            data[i][j-1] = string.trim().parse::<f32>().unwrap();
            if data[i][j-1] > maxs[j-1]{
                maxs[j] = data[i][j-1]
            }
            if data[i][j-1] < mins[j-1]{
                mins[j-1] = data[i][j-1]
            }
        }
        let mut chars = vector[1].chars();
        chars.next();
        chars.next_back();
        let string = chars.as_str();
        labels[i][0] = string.trim().parse::<f32>().unwrap();
        standardisation_labels[i] = labels[i][0];
        i += 1;
    }

    let mean:f32 = standardisation_labels.iter().sum::<f32>() / standardisation_labels.len() as f32;
    let mut deviation:f32 = standardisation_labels
        .iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f32>();
    deviation /= standardisation_labels.len() as f32;
    deviation = deviation.sqrt();

    for i in 0..data.len(){
        for j in 0..data[0].len(){
            data[i][j] = (data[i][j] - mins[j]) / (maxs[j] - mins[j])
        }
        labels[i][0] = (labels[i][0] - mean) / deviation;
    }

    return (data, labels);
}