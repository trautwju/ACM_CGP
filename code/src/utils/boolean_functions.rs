use ndarray::prelude::*;

pub fn and(con1: &ArrayView1<bool>, con2: &ArrayView1<bool>) -> Array1<bool> {
    return con1 & con2;
}

pub fn or(con1: &ArrayView1<bool>, con2: &ArrayView1<bool>) -> Array1<bool> {
    return con1 | con2;
}

pub fn nand(con1: &ArrayView1<bool>, con2: &ArrayView1<bool>) -> Array1<bool> {
    return !and(&con1, &con2);
}

pub fn nor(con1: &ArrayView1<bool>, con2: &ArrayView1<bool>) -> Array1<bool> {
    return !or(&con1, &con2);
}

// pub fn and(con1: &Vec<bool>, con2: &Vec<bool>) -> Vec<bool> {
//     return con1
//         .iter()
//         .zip(con2.iter())
//         .map(|(&a, &b)| a & b)
//         .collect();
// }
//
// pub fn or(con1: &Vec<bool>, con2: &Vec<bool>) -> Vec<bool> {
//     return con1
//         .iter()
//         .zip(con2.iter())
//         .map(|(&a, &b)| a | b)
//         .collect();
// }
//
// pub fn nand(con1: &Vec<bool>, con2: &Vec<bool>) -> Vec<bool> {
//     return and(con1, con2).iter().map(|&a| !a).collect();
// }
//
// pub fn nor(con1: &Vec<bool>, con2: &Vec<bool>) -> Vec<bool> {
//     return or(con1, con2).iter().map(|&a| !a).collect();
// }
