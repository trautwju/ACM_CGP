pub fn get_dataset() -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut prel_data= vec![vec![0.; 1]; (10. / 0.4 + 1.) as usize];
    let mut data = vec![vec![0.; 2];  ((10. / 0.4 + 1.) * (10. / 0.4 + 1.)) as usize];
    let mut labels = vec![vec![0.; 1]; ((10. / 0.4 + 1.) * (10. / 0.4 + 1.)) as usize];
    for i in 0..prel_data.len(){
        prel_data[i][0] = ((-5. + (i as f32) * 0.4) * 100. ).round() / 100.;
    }
    let mut counter = 0;
    for i in 0..prel_data.len(){
        for j in 0..prel_data.len(){
            data[counter][0] = prel_data[i][0];
            data[counter][1] = prel_data[j][0];
            labels[counter][0] = (1. / (1. + data[counter][0].powf(-4.))) + (1. / (1. + data[counter][1].powf(-4.)));
            counter += 1;
        }
    }
    return (data, labels)
}