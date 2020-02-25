import mlflow

if __name__ == "__main__":
    models = ['unet256', 'unet512', 'unet1024', 'smallu128', 'smallu256', 'smallu512', 'smallu1024']
    for name in models:
        for bsize in [32, 16, 8, 4]:
            for learningrate in [0.01, 0.03, 0.001, 0.005]:
                params = {
                    "model": name,
                    "batch_size": bsize,
                    "epochs": 10,
                    "lr_start": learningrate,
                    "experiment": 'grid_search'
                }
                mlflow.projects.run('.', entry_point='noaug', parameters=params, experiment_name='grid_search')
