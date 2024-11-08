use std::fs::File;
use std::io::{self, BufRead};

pub fn get_dataset() -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let file = File::open("/data/oc-compute03/trautwju/Masterarbeit/src/datasets/Data/Klassifikation/chronic_kidney_disease_full.arff").expect("Couldn't open input");
    let mut csv_file = csv::Reader::from_reader(file);
    let data_len = csv_file.records().count();
    let mut data = vec![vec![0.; 24];  data_len - 26];
    let mut labels = vec![vec![0.; 2];  data_len - 26];
    let mut mins = vec![100000000000.; 24];
    let mut maxs = vec![-10000000000.; 24];
    let mut i = 0;
    let file = File::open("/data/oc-compute03/trautwju/Masterarbeit/src/datasets/Data/Klassifikation/chronic_kidney_disease_full.arff").expect("Couldn't open input");
    let reader = io::BufReader::new(file);
    for line in reader.lines() {
        let record = line.unwrap();
        if record.starts_with("@"){
            continue;
        }
        let vector:Vec<&str> = record.split(",").collect();

        data[i][0] = match vector[0].trim(){
            "?" => -1.,
            _ => vector[0].trim().parse::<f32>().unwrap(),
        };

        data[i][1] = match vector[1].trim(){
            "?" => -1.,
            _ => vector[1].trim().parse::<f32>().unwrap(),
        };

        data[i][2] = match vector[2].trim(){
            "?" => -1.,
            _ => vector[2].trim().parse::<f32>().unwrap(),
        };

        data[i][3] = match vector[3].trim(){
            "?" => -1.,
            _ => vector[3].trim().parse::<f32>().unwrap(),
        };

        data[i][4] = match vector[4].trim(){
            "?" => -1.,
            _ => vector[4].trim().parse::<f32>().unwrap(),
        };

        data[i][5] = match vector[5].trim(){
            "'abnormal'" => 0.0,
            "'normal'" => 1.0,
            _ => -1.
        };

        data[i][6] = match vector[6].trim(){
            "'abnormal'" => 0.0,
            "'normal'" => 1.0,
            _ => -1.
        };

        data[i][7] = match vector[7].trim(){
            "'notpresent'" => 0.0,
            "'present'" => 1.0,
            _ => -1.
        };

        data[i][8] = match vector[8].trim(){
            "'notpresent'" => 0.0,
            "'present'" => 1.0,
            _ => -1.
        };

        data[i][9] = match vector[9].trim(){
            "?" => -1.,
            _ => vector[9].trim().parse::<f32>().unwrap(),
        };

        data[i][10] = match vector[10].trim(){
            "?" => -1.,
            _ => vector[10].trim().parse::<f32>().unwrap(),
        };

        data[i][11] = match vector[11].trim(){
            "?" => -1.,
            _ => vector[11].trim().parse::<f32>().unwrap(),
        };

        data[i][12] = match vector[12].trim(){
            "?" => -1.,
            _ => vector[12].trim().parse::<f32>().unwrap(),
        };

        data[i][13] = match vector[13].trim(){
            "?" => -1.,
            _ => vector[13].trim().parse::<f32>().unwrap(),
        };

        data[i][14] = match vector[14].trim(){
            "?" => -1.,
            _ => vector[14].trim().parse::<f32>().unwrap(),
        };

        data[i][15] = match vector[15].trim(){
            "?" => -1.,
            _ => vector[15].trim().parse::<f32>().unwrap(),
        };

        data[i][16] = match vector[16].trim(){
            "?" => -1.,
            _ => vector[16].trim().parse::<f32>().unwrap(),
        };

        data[i][17] = match vector[17].trim(){
            "?" => -1.,
            _ => vector[17].trim().parse::<f32>().unwrap(),
        };

        data[i][18] = match vector[18].trim(){
            "'no'" => 0.0,
            "'yes'" => 1.0,
            _ => -1.
        };

        data[i][19] = match vector[19].trim(){
            "'no'" => 0.0,
            "'yes'" => 1.0,
            "' yes'" => 2.0,
            _ => -1.
        };

        data[i][20] = match vector[20].trim(){
            "'no'" => 0.0,
            "'yes'" => 1.0,
            _ => -1.
        };

        data[i][21] = match vector[21].trim(){
            "'good'" => 0.0,
            "'poor'" => 1.0,
            _ => -1.
        };

        data[i][22] = match vector[22].trim(){
            "'no'" => 0.0,
            "'yes'" => 1.0,
            _ => -1.
        };

        data[i][23] = match vector[23].trim(){
            "'no'" => 0.0,
            "'yes'" => 1.0,
            _ => -1.
        };

        let label = match vector[24].trim(){
            "'ckd'" => 0,
            "'notckd'" => 1,
            _ => 100,
        };

        labels[i][label] = 1.;
        for j in 0..24{
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