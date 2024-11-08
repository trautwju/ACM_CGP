use std::fs::File;
use std::io::{self, BufRead};

pub fn get_dataset() -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let file = File::open("/data/oc-compute03/trautwju/Masterarbeit/src/datasets/Data/Klassifikation/adult.data").expect("Couldn't open input");
    let mut csv_file = csv::Reader::from_reader(file);
    let data_len = csv_file.records().count();
    let mut data = vec![vec![0.; 14];  data_len + 1];
    let mut labels = vec![vec![0.; 2];  data_len + 1];
    let mut mins = vec![100000000000.; 14];
    let mut maxs = vec![-10000000000.; 14];
    let mut i = 0;
    let file = File::open("/data/oc-compute03/trautwju/Masterarbeit/src/datasets/Data/Klassifikation/adult.data").expect("Couldn't open input");
    let reader = io::BufReader::new(file);
    for line in reader.lines() {
        let record = line.unwrap();
        let vector:Vec<&str> = record.split(",").collect();
        data[i][0] = match vector[0].trim(){
            "?" => -1.,
            _ => vector[0].trim().parse::<f32>().unwrap(),
        };

        data[i][1] = match vector[1].trim() {
            "Private" => 0.0,
            "Self-emp-not-inc" => 1.0,
            "Self-emp-inc" => 2.0,
            "Federal-gov" => 3.0,
            "Local-gov" => 4.0,
            "State-gov" => 5.0,
            "Without-pay" => 6.0,
            "Never-worked" => 7.0,
            _ => -1.,
        };

        data[i][2] = match vector[2].trim(){
            "?" => -1.,
            _ => vector[2].trim().parse::<f32>().unwrap(),
        };

        data[i][3] = match vector[3].trim() {
            "Bachelors" => 0.0,
            "Some-college" => 1.0,
            "11th" => 2.0,
            "HS-grad" => 3.0,
            "Prof-school" => 4.0,
            "Assoc-acdm" => 5.0,
            "Assoc-voc" => 6.0,
            "9th" => 7.0,
            "7th-8th" => 8.0,
            "12th" => 9.0,
            "Masters" => 10.0,
            "1st-4th" => 11.0,
            "10th" => 12.0,
            "Doctorate" => 13.0,
            "5th-6th" => 14.0,
            "Preschool" => 15.0,
            _ => -1.,
        };

        data[i][4] = match vector[4].trim(){
            "?" => -1.,
            _ => vector[4].trim().parse::<f32>().unwrap(),
        };

        data[i][5] = match vector[5].trim() {
            "Married-civ-spouse" => 0.0,
            "Divorced" => 1.0,
            "Never-married" => 2.0,
            "Separated" => 3.0,
            "Widowed" => 4.0,
            "Married-spouse-absent" => 5.0,
            "Married-AF-spouse" => 6.0,
            _ => -1.,
        };

        data[i][6] = match vector[6].trim() {
            "Tech-support" => 0.0,
            "Craft-repair" => 1.0,
            "Other-service" => 2.0,
            "Sales" => 3.0,
            "Exec-managerial" => 4.0,
            "Prof-specialty" => 5.0,
            "Handlers-cleaners" => 6.0,
            "Machine-op-inspct" => 7.0,
            "Adm-clerical" => 8.0,
            "Farming-fishing" => 9.0,
            "Transport-moving" => 10.0,
            "Priv-house-serv" => 11.0,
            "Protective-serv" => 12.0,
            "Armed-Forces" => 13.0,
            _ => -1.,
        };

        data[i][7] = match vector[7].trim() {
            "Wife" => 0.0,
            "Own-child" => 1.0,
            "Husband" => 2.0,
            "Not-in-family" => 3.0,
            "Other-relative" => 4.0,
            "Unmarried" => 5.0,
            _ => -1.,
        };

        data[i][8] = match vector[8].trim() {
            "White" => 0.0,
            "Asian-Pac-Islander" => 1.0,
            "Amer-Indian-Eskimo" => 2.0,
            "Other" => 3.0,
            "Black" => 4.0,
            _ => -1.,
        };

        data[i][9] = match vector[9].trim() {
            "Female" => 0.0,
            "Male" => 1.0,
            _ => -1.,
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

        data[i][13] = match vector[13].trim() {
            "United-States" => 0.0,
            "Cambodia" => 1.0,
            "England" => 2.0,
            "Puerto-Rico" => 3.0,
            "Canada" => 4.0,
            "Germany" => 5.0,
            "Outlying-US(Guam-USVI-etc)" => 6.0,
            "India" => 7.0,
            "Japan" => 8.0,
            "Greece" => 9.0,
            "South" => 10.0,
            "China" => 11.0,
            "Cuba" => 12.0,
            "Iran" => 13.0,
            "Honduras" => 14.0,
            "Philippines" => 15.0,
            "Italy" => 16.0,
            "Poland" => 17.0,
            "Jamaica" => 18.0,
            "Vietnam" => 19.0,
            "Mexico" => 20.0,
            "Portugal" => 21.0,
            "Ireland" => 22.0,
            "France" => 23.0,
            "Dominican-Republic" => 24.0,
            "Laos" => 25.0,
            "Ecuador" => 26.0,
            "Taiwan" => 27.0,
            "Haiti" => 28.0,
            "Columbia" => 29.0,
            "Hungary" => 30.0,
            "Guatemala" => 31.0,
            "Nicaragua" => 32.0,
            "Scotland" => 33.0,
            "Thailand" => 34.0,
            "Yugoslavia" => 35.0,
            "El-Salvador" => 36.0,
            "Trinadad&Tobago" => 37.0,
            "Peru" => 38.0,
            "Hong" => 39.0,
            "Holand-Netherlands" => 40.0,
            _ => -1.,
        };

        let label = match vector[14].trim() {
            "<=50K" => 0,
            ">50K" => 1,
            _ => 2,
        };
        labels[i][label] = 1.;

        for j in 0..14{
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