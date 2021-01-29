import train_policy
import racer
import argparse
import os

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
    parser.add_argument("--n_epochs", type=int, help="number of epochs", default=50)
    parser.add_argument("--batch_size", type=int, help="batch_size", default=256)
    parser.add_argument("--n_steering_classes", type=int, help="number of steering classes", default=20)
    parser.add_argument("--train_dir", help="directory of training data", default='./dataset/train')
    parser.add_argument("--validation_dir", help="directory of validation data", default='./dataset/val')
    parser.add_argument("--weights_out_file", help="where to save the weights of the network e.g. ./weights/learner_0.weights", default='')
    parser.add_argument("--dagger_iterations", help="", default=10) 
    args = parser.parse_args()

    ####
    # Enter your DAgger code here
    # Reuse functions in racer.py and train_policy.py
    # Save the learner weights of the i-th DAgger iteration in ./weights/learner_i.weights where 
    ####
   
    iteration = 0 
    # For the first run train the model on the provided dataset. This will act as our initial expert demonstration. 
    print ('TRAINING LEARNER ON INITIAL DATASET')
    os.system('python train_policy.py --n_epochs=50 --batch_size=256 --weights_out_file=./weights/learner_{}.weights --train_dir=./dataset/train/ --weighted_loss=False'.format(iteration))

    # run for dagger_iterations
    while iteration < args.dagger_iterations:
        
        # rollout the policy learnt by network in the previous iteration, but get feedback from the expert and store to dataset(aggregation).
        print ('GETTING EXPERT DEMONSTRATIONS')
        os.system('python racer.py --out_dir=./dataset/train --save_expert_actions=True --expert_drives=False --run_id={} --timesteps=100000 --learner_weights=./weights/learner_{}.weights --n_steering_classes=20'.format(iteration, iteration))

        # get new policy by retraining model from scratch on the new aggregated dataset. these weights will be used for rollout in next iteration.
        print ('RETRAINING LEARNER ON AGGREGATED DATASET')
        os.system('python train_policy.py --n_epochs=50 --batch_size=256 --weights_out_file=./weights/learner_{}.weights --train_dir=./dataset/train/ --weighted_loss=False'.format(iteration+1))
        
        iteration = iteration + 1

    # plot the cross track errors saved in log file
    errors = []
    iterations = list(range(0, args.dagger_iterations))
    f = open('logs.txt', 'r')
    for line in f:
        errors.append(float(line))
    
    import matplotlib.pyplot as plt 
    
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(iterations, errors)
    ax.set_xlabel('Dagger Iterations')
    ax.set_ylabel('Cumulative CrossTrack Error (Error dist.)') # note that the cumulative crosstrack errors have been averaged over the number of timesteps traversed by agent during rollout.

    plt.show()