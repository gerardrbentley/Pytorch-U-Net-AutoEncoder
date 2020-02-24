import mlflow
import mlflow.pytorch
import os
from datetime import datetime
from contextlib import nullcontext

def get_run_manager(args):
    """
        Returns nullcontext manager if not args.log_mlflow. Else returns mlflow.start_run() with run configured
    """
    if not args.log_mlflow:
        return nullcontext()
    BASE_DIR = '/faim/mlflow'
    
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    try:
        if args.checkpoint == '':
            checkpoint = 'none'
        else:
            checkpoint = args.checkpoint
            checkpoint = str(checkpoint).replace(os.sep, '_')
            # Get artifact uri
        mlflow.set_experiment(args.experiment)
        dir_string = os.path.join(BASE_DIR, args.experiment)
        # mlflow.set_tracking_uri('file:/' + dir_string)
        
        run_string = f'm_{args.model}_d_{args.dataset}_e_{args.epochs}_b_{args.batch_size}_cp_{checkpoint}_{current_time}_{args.comment}'
        print_string = f'Started Training for {args.model} (initialized at checkpoint: {checkpoint}) on dataset {args.dataset} for {args.epochs} epochs, batch size {args.batch_size}. Comment: {args.comment}. DT: {current_time}'
    except:
        print(f'Could not Parse Project Args and start mlflow tracking run')
        return nullcontext()
    # print(f'mlflow dir_string: {dir_string}')
    print(f"mlflow run_string: {run_string}")
    return mlflow.start_run(run_name=run_string)
        