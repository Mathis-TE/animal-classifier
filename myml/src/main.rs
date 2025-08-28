
//XOR
//0, 0 -> 0
//0, 1 -> 1
//1, 0 -> 1
//1, 1 -> 0

use myml::{network::Network};

fn main(){
    let inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];

    let targets = vec![
        vec![0.0],
        vec![1.0],
        vec![1.0],
        vec![0.0],
    ];

    let mut network = Network::new(vec![2, 3, 1], 0.2, Some(2));

    network.train(inputs.clone(), targets.clone(), 10000);

    println!("0 and 0: {:?}", network.feed_forward(vec![0.0, 0.0]));
    println!("0 and 1: {:?}", network.feed_forward(vec![0.0, 1.0]));
    println!("1 and 0: {:?}", network.feed_forward(vec![1.0, 0.0]));
    println!("1 and 1: {:?}", network.feed_forward(vec![1.0, 1.0]));

    let epochs = 2_000;
    let hist = network.train_return_mse(inputs.clone(), targets.clone(), epochs);

    // Affichages utiles
    println!("Epochs: {}", hist.len());
    println!("MSE début : {:.12e}", hist[0]);
    println!("MSE milieu: {:.12e}", hist[hist.len()/2]);
    println!("MSE fin   : {:.12e}", hist[hist.len()-1]);

    // (Optionnel) échantillonner l'historique pour le log
    let step = (hist.len() / 10).max(1);
    for (i, mse) in hist.iter().enumerate().step_by(step) {
        println!("epoch {:>5} -> MSE {:.12e}", i + 1, mse);
    }

    // Vérif des prédictions
    for x in &inputs {
        println!("{x:?} -> {:?}", network.feed_forward(x.clone()));
    }


}