use rand::Rng;

// TODO: implement random from scratch
fn cost(data: &Vec<(f64, f64)>, w: &f64, b: &f64) -> f64 {
    // mse
    data.iter()
        .map(|(x, y)| {
            let y_pred = w * x + b;
            let err = y_pred - y;
            err * err
        })
        .sum::<f64>()
        / data.len() as f64
}

fn main() {
    let data = vec![(0.0, 0.0), (1.0, 2.0), (2.0, 4.0), (3.0, 6.0), (4.0, 8.0)];
    let mut rng = rand::thread_rng();
    // y = x * w
    let mut w: f64 = rng.gen_range(0.0..10.0);
    let mut b: f64 = rng.gen_range(0.0..5.0);
    let eps = 1e-3;
    let lr = 1e-3;
    // finite difference to calculate cost distance (usually done with derrivative)
    println!("cost: {};", cost(&data, &w, &b));
    for _ in 0..500 {
        let c = cost(&data, &w, &b);
        let dw = (cost(&data, &(w + eps), &b) - c) / eps;
        let db = (cost(&data, &w, &(b + eps)) - c) / eps;
        w -= dw * lr;
        b -= db * lr;
        println!("cost: {}; w: {}; b: {};", c, w, b);
    }
}
