use std::fs::File;
use std::io::{self, BufRead};

pub fn get_dataset() -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let file = File::open("/data/oc-compute03/trautwju/Masterarbeit/src/datasets/Data/Klassifikation/bach.arff").expect("Couldn't open input");
    let mut csv_file = csv::Reader::from_reader(file);
    let data_len = csv_file.records().count();
    let mut data = vec![vec![0.; 15];  data_len - 17];
    let mut labels = vec![vec![0.; 102];  data_len - 17];
    let mut mins = vec![100000000000.; 15];
    let mut maxs = vec![-10000000000.; 15];
    let mut i = 0;
    let file = File::open("/data/oc-compute03/trautwju/Masterarbeit/src/datasets/Data/Klassifikation/bach.arff").expect("Couldn't open input");
    let reader = io::BufReader::new(file);
    for line in reader.lines() {
        let record = line.unwrap();
        if record.starts_with("@"){
            continue;
        }
        let vector:Vec<&str> = record.split(",").collect();

        data[i][0] = vector[0].trim().parse::<f32>().unwrap();

        data[i][1] = match vector[1].trim() {
            "' NO'" => 0.0,
            "'YES'" => 1.0,
            _ => -1.,
        };

        data[i][2] = match vector[2].trim() {
            "' NO'" => 0.0,
            "'YES'" => 1.0,
            _ => -1.,
        };
        data[i][3] = match vector[5].trim() {
            "' NO'" => 0.0,
            "'YES'" => 1.0,
            _ => -1.,
        };
        data[i][4] = match vector[4].trim() {
            "' NO'" => 0.0,
            "'YES'" => 1.0,
            _ => -1.,
        };
        data[i][5] = match vector[5].trim() {
            "' NO'" => 0.0,
            "'YES'" => 1.0,
            _ => -1.,
        };
        data[i][6] = match vector[6].trim() {
            "' NO'" => 0.0,
            "'YES'" => 1.0,
            _ => -1.,
        };
        data[i][7] = match vector[7].trim() {
            "' NO'" => 0.0,
            "'YES'" => 1.0,
            _ => -1.,
        };
        data[i][8] = match vector[8].trim() {
            "' NO'" => 0.0,
            "'YES'" => 1.0,
            _ => -1.,
        };
        data[i][9] = match vector[9].trim() {
            "' NO'" => 0.0,
            "'YES'" => 1.0,
            _ => -1.,
        };
        data[i][10] = match vector[10].trim() {
            "' NO'" => 0.0,
            "'YES'" => 1.0,
            _ => -1.,
        };
        data[i][11] = match vector[11].trim() {
            "' NO'" => 0.0,
            "'YES'" => 1.0,
            _ => -1.,
        };
        data[i][12] = match vector[12].trim() {
            "' NO'" => 0.0,
            "'YES'" => 1.0,
            _ => -1.,
        };

        data[i][13] = match vector[13].trim() {
            "'A'" => 0.0,
            "'A#'" => 1.0,
            "'Ab'" => 2.0,
            "'B'" => 3.0,
            "'Bb'" => 4.0,
            "'C'" => 5.0,
            "'C#'" => 6.0,
            "'D'" => 7.0,
            "'D#'" => 8.0,
            "'Db'" => 9.0,
            "'E'" => 10.0,
            "'Eb'" => 11.0,
            "'F'" => 12.0,
            "'F#'" => 13.0,
            "'G'" => 14.0,
            "'G#'" => 15.0,
            _ => -1.,
        };

        data[i][14] = vector[14].trim().parse::<f32>().unwrap();

        let label = match vector[15].trim() {
            "' A#d'" => 0,
            "' A#d7'" => 1,
            "' A_d'" => 2,
            "' A_m'" => 3,
            "' A_M'" => 4,
            "' A_m4'" => 5,
            "' A_M4'" => 6,
            "' A_m6'" => 7,
            "' A_M6'" => 8,
            "' A_m7'" => 9,
            "' A_M7'" => 10,
            "' Abd'" => 11,
            "' Abm'" => 12,
            "' AbM'" => 13,
            "' B_d'" => 14,
            "' B_d7'" => 15,
            "' B_m'" => 16,
            "' B_M'" => 17,
            "' B_M4'" => 18,
            "' B_m6'" => 19,
            "' B_m7'" => 20,
            "' B_M7'" => 21,
            "' Bbd'" => 22,
            "' Bbm'" => 23,
            "' BbM'" => 24,
            "' Bbm6'" => 25,
            "' BbM7'" => 26,
            "' C#d'" => 27,
            "' C#d6'" => 28,
            "' C#d7'" => 29,
            "' C#m'" => 30,
            "' C#M'" => 31,
            "' C#M4'" => 32,
            "' C#m7'" => 33,
            "' C#M7'" => 34,
            "' C_d6'" => 35,
            "' C_d7'" => 36,
            "' C_m'" => 37,
            "' C_M'" => 38,
            "' C_M4'" => 39,
            "' C_m6'" => 40,
            "' C_M6'" => 41,
            "' C_m7'" => 42,
            "' C_M7'" => 43,
            "' D#d'" => 44,
            "' D#d6'" => 45,
            "' D#d7'" => 46,
            "' D#m'" => 47,
            "' D#M'" => 48,
            "' D_d7'" => 49,
            "' D_m'" => 50,
            "' D_M'" => 51,
            "' D_M4'" => 52,
            "' D_m6'" => 53,
            "' D_M6'" => 54,
            "' D_m7'" => 55,
            "' D_M7'" => 56,
            "' Dbd'" => 57,
            "' Dbd7'" => 58,
            "' Dbm'" => 59,
            "' DbM'" => 60,
            "' Dbm7'" => 61,
            "' DbM7'" => 62,
            "' E_d'" => 63,
            "' E_m'" => 64,
            "' E_M'" => 65,
            "' E_M4'" => 66,
            "' E_m6'" => 67,
            "' E_m7'" => 68,
            "' E_M7'" => 69,
            "' Ebd'" => 70,
            "' EbM'" => 71,
            "' EbM7'" => 72,
            "' F#d'" => 73,
            "' F#d7'" => 74,
            "' F#m'" => 75,
            "' F#M'" => 76,
            "' F#M4'" => 77,
            "' F#m6'" => 78,
            "' F#m7'" => 79,
            "' F#M7'" => 80,
            "' F_d'" => 81,
            "' F_d7'" => 82,
            "' F_m'" => 83,
            "' F_M'" => 84,
            "' F_M4'" => 85,
            "' F_m6'" => 86,
            "' F_M6'" => 87,
            "' F_m7'" => 88,
            "' F_M7'" => 89,
            "' G#d'" =>  90,
            "' G#d7'" => 91,
            "' G#m'" => 92,
            "' G#M'" => 93,
            "' G_d'" => 94,
            "' G_m'" => 95,
            "' G_M'" => 96,
            "' G_M4'" => 97,
            "' G_m6'" => 98,
            "' G_M6'" => 99,
            "' G_m7'" => 100,
            "' G_M7'" => 101,
            _ => 200,
        };

        labels[i][label] = 1.;
        for j in 0..15{
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