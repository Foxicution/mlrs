use rand::Rng;

fn sigmoid(x: &f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[derive(Debug, Default)]
struct Xor {
    or_w1: f64,
    or_w2: f64,
    or_b: f64,
    nand_w1: f64,
    nand_w2: f64,
    nand_b: f64,
    and_w1: f64,
    and_w2: f64,
    and_b: f64,
}

impl Xor {
    fn new() -> Xor {
        // TODO: implement random from scratch
        let mut rng = rand::thread_rng();
        Xor {
            or_w1: rng.gen_range(0.0..1.0),
            or_w2: rng.gen_range(0.0..1.0),
            or_b: rng.gen_range(0.0..5.0),
            nand_w1: rng.gen_range(0.0..1.0),
            nand_w2: rng.gen_range(0.0..1.0),
            nand_b: rng.gen_range(0.0..5.0),
            and_w1: rng.gen_range(0.0..1.0),
            and_w2: rng.gen_range(0.0..1.0),
            and_b: rng.gen_range(0.0..5.0),
        }
    }
}

fn forward(m: &Xor, x1: &f64, x2: &f64) -> f64 {
    let a = sigmoid(&(m.or_w1 * x1 + m.or_w2 * x2 + m.or_b));
    let b = sigmoid(&(m.nand_w1 * x1 + m.nand_w2 * x2 + m.nand_b));
    sigmoid(&(m.and_w1 * a + m.and_w2 * b + m.and_b))
}

fn mse(data: &Vec<(f64, f64, f64)>, m: &Xor) -> f64 {
    data.iter()
        .map(|(x1, x2, y)| {
            let y_pred = forward(m, x1, x2);
            let err = y_pred - y;
            err * err
        })
        .sum::<f64>()
        / data.len() as f64
}

// calc gradient
fn finite_diff(data: &Vec<(f64, f64, f64)>, m: &mut Xor, eps: &f64) -> Xor {
    let mut g = Xor::default();
    let cost = mse(data, m);

    let mut saved = m.or_w1;
    m.or_w1 += eps;
    g.or_w1 = (mse(data, m) - cost) / eps;
    m.or_w1 = saved;

    saved = m.or_w2;
    m.or_w2 += eps;
    g.or_w2 = (mse(data, m) - cost) / eps;
    m.or_w2 = saved;

    saved = m.or_b;
    m.or_b += eps;
    g.or_b = (mse(data, m) - cost) / eps;
    m.or_b = saved;

    saved = m.nand_w1;
    m.nand_w1 += eps;
    g.nand_w1 = (mse(data, m) - cost) / eps;
    m.nand_w1 = saved;

    saved = m.nand_w2;
    m.nand_w2 += eps;
    g.nand_w2 = (mse(data, m) - cost) / eps;
    m.nand_w2 = saved;

    saved = m.nand_b;
    m.nand_b += eps;
    g.nand_b = (mse(data, m) - cost) / eps;
    m.nand_b = saved;

    saved = m.and_w1;
    m.and_w1 += eps;
    g.and_w1 = (mse(data, m) - cost) / eps;
    m.and_w1 = saved;

    saved = m.and_w2;
    m.and_w2 += eps;
    g.and_w2 = (mse(data, m) - cost) / eps;
    m.and_w2 = saved;

    saved = m.and_b;
    m.and_b += eps;
    g.and_b = (mse(data, m) - cost) / eps;
    m.and_b = saved;

    g
}

// gradient descend
fn apply_diff(m: &mut Xor, g: &Xor, lr: &f64) {
    m.or_w1 -= g.or_w1 * lr;
    m.or_w2 -= g.or_w2 * lr;
    m.or_b -= g.or_b * lr;

    m.nand_w1 -= g.nand_w1 * lr;
    m.nand_w2 -= g.nand_w2 * lr;
    m.nand_b -= g.nand_b * lr;

    m.and_w1 -= g.and_w1 * lr;
    m.and_w2 -= g.and_w2 * lr;
    m.and_b -= g.and_b * lr;
}

fn train(data: &Vec<(f64, f64, f64)>) {
    let mut m = Xor::new();
    // finite difference to calculate cost distance (usually done with derrivative)
    let lr = 1e-1;
    let eps = 1e-3;

    println!("Initial cost: {}", mse(data, &m));
    for _ in 0..100_000 {
        let g = finite_diff(data, &mut m, &eps);
        apply_diff(&mut m, &g, &lr);
    }
    println!("New cost: {}", mse(data, &m));

    println!("\nModel");
    data.iter()
        .for_each(|(x1, x2, _y)| println!("{} | {} = {}", x1, x2, forward(&m, x1, x2).round()));

    println!("\nLayer 1, neuron 1");
    data.iter().for_each(|(x1, x2, _y)| {
        println!(
            "{} | {} = {}",
            x1,
            x2,
            sigmoid(&(m.or_w1 * x1 + m.or_w2 * x2 + m.or_b)).round()
        )
    });

    println!("\nLayer 1, neuron 2");
    data.iter().for_each(|(x1, x2, _y)| {
        println!(
            "{} | {} = {}",
            x1,
            x2,
            sigmoid(&(m.nand_w1 * x1 + m.nand_w2 * x2 + m.nand_b)).round()
        )
    });

    println!("\nLayer 2, neuron 1");
    data.iter().for_each(|(x1, x2, _y)| {
        println!(
            "{} | {} = {}",
            x1,
            x2,
            sigmoid(&(m.and_w1 * x1 + m.and_w2 * x2 + m.and_b)).round()
        )
    });
}

fn main() {
    println!("-----------------------");
    println!("OR gate");
    let mut data = vec![
        (1.0, 1.0, 1.0),
        (1.0, 0.0, 1.0),
        (0.0, 1.0, 1.0),
        (0.0, 0.0, 0.0),
    ];
    train(&data);

    println!("-----------------------");
    println!("AND gate");
    data = vec![
        (1.0, 1.0, 1.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0),
    ];
    train(&data);

    println!("-----------------------");
    println!("NAND gate");
    data = vec![
        (1.0, 1.0, 0.0),
        (1.0, 0.0, 1.0),
        (0.0, 1.0, 1.0),
        (0.0, 0.0, 1.0),
    ];
    train(&data);

    // (x|y) & ~(x&y)
    println!("-----------------------");
    println!("XOR gate");
    data = vec![
        (1.0, 1.0, 0.0),
        (1.0, 0.0, 1.0),
        (0.0, 1.0, 1.0),
        (0.0, 0.0, 0.0),
    ];
    train(&data);
}
