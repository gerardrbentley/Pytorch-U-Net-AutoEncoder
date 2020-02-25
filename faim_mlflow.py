import mlflow
import mlflow.pytorch
import os
from datetime import datetime
from contextlib import nullcontext

def get_run_manager(args):
    """
        Returns nullcontext manager if not args.log_mlflow. Returns current run if launched from MLFlow, Else returns mlflow.start_run() with new run configured
    """
    if not args.log_mlflow:
        return nullcontext()
    
    # Unsure when this will hit
    current_run = mlflow.active_run()
    if current_run is not None:
        print('Active run found: ', type(current_run))
        return current_run
    else:
        print('No MLFlow Run active')
    
    # When called via `mlflow run --experiment-name exp . ` MLFLOW_RUN_ID will be in environment. 
    if 'MLFLOW_RUN_ID' in os.environ:
        print(f"MLFLOW RUN IN ENV: {os.environ['MLFLOW_RUN_ID']}")
        return mlflow.start_run()
    else:
        print('No MLFlow Run in ENV')

    # Set experiment for this run (affects python run calls)
    mlflow.set_experiment(args.experiment)

    BASE_DIR = '/faim/mlflow'
    
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    try:
        if args.checkpoint == '':
            checkpoint = 'none'
        else:
            checkpoint = args.checkpoint
            checkpoint = str(checkpoint).replace(os.sep, '_')
            # Get artifact uri
        dir_string = os.path.join(BASE_DIR, args.experiment)
        # mlflow.set_tracking_uri('file:/' + dir_string)
        
        run_string = f'model_{args.model}_data_{args.dataset}_{args.comment}'
        print_string = f'Started Training for {args.model} (initialized at checkpoint: {checkpoint}) on dataset {args.dataset} for {args.epochs} epochs, batch size {args.batch_size}. Comment: {args.comment}. DT: {current_time}'
    except:
        print(f'Could not Parse Project Args and start mlflow tracking run')
        return nullcontext()
    # print(f'mlflow dir_string: {dir_string}')
    print(f"mlflow run_string: {run_string}")
    return mlflow.start_run(run_name=run_string)
        