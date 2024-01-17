use rand::Rng;

fn sigmoid(x: &f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// TODO: implement random from scratch
fn mse(data: &Vec<(f64, f64, f64)>, w1: &f64, w2: &f64, b: &f64) -> f64 {
    // mse
    data.iter()
        .map(|(x1, x2, y)| {
            let y_pred = sigmoid(&(w1 * x1 + w2 * x2 + b));
            let err = y_pred - y;
            err * err
        })
        .sum::<f64>()
        / data.len() as f64
}

fn train(data: &Vec<(f64, f64, f64)>) {
    let mut rng = rand::thread_rng();
    // y = x * w
    let mut w1: f64 = rng.gen_range(0.0..1.0);
    let mut w2: f64 = rng.gen_range(0.0..1.0);
    let mut b: f64 = rng.gen_range(0.0..5.0);
    let eps = 1e-1;
    let lr = 1e-1;
    let mut cost = mse(data, &w1, &w2, &b);
    // finite difference to calculate cost distance (usually done with derrivative)
    println!("initial cost: {};", mse(data, &w1, &w2, &b));
    for _ in 0..10000 {
        cost = mse(data, &w1, &w2, &b);
        let dw1 = (mse(data, &(w1 + eps), &w2, &b) - cost) / eps;
        let dw2 = (mse(data, &w1, &(w2 + eps), &b) - cost) / eps;
        let db = (mse(data, &w1, &w2, &(b + eps)) - cost) / eps;
        w1 -= dw1 * lr;
        w2 -= dw2 * lr;
        b -= db * lr;
    }
    println!(
        "cost: {:.2}; w1: {:.2}; w2: {:.2}; b: {:.2};",
        cost, w1, w2, b
    );

    data.iter().for_each(|(x1, x2, _y)| {
        println!("{} | {} = {}", x1, x2, sigmoid(&(w1 * x1 + w2 * x2 + b)))
    })
}

fn main() {
    println!("Single neuron");
    println!("OR gate");
    let mut data = vec![
        (1.0, 1.0, 1.0),
        (1.0, 0.0, 1.0),
        (0.0, 1.0, 1.0),
        (0.0, 0.0, 0.0),
    ];
    train(&data);

    println!("AND gate");
    data = vec![
        (1.0, 1.0, 1.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0),
    ];
    train(&data);

    println!("NAND gate");
    data = vec![
        (1.0, 1.0, 0.0),
        (1.0, 0.0, 1.0),
        (0.0, 1.0, 1.0),
        (0.0, 0.0, 1.0),
    ];
    train(&data);

    // (x|y) & ~(x&y)
    println!("XOR gate");
    data = vec![
        (1.0, 1.0, 0.0),
        (1.0, 0.0, 1.0),
        (0.0, 1.0, 1.0),
        (0.0, 0.0, 0.0),
    ];
    train(&data);
}
