if True:
    import pandas as pd
    import numpy as np
    import sys
    sys.path.append('/mydata/watres/quentin/code/FLOW/hourly_analysis/')
    import GAMCR
    
    station = 'Basel'
    mode = 'notflashy'
    model_ghost = GAMCR.model.GAMCR(lam=0.1)
    save_folder = './{0}_{1}/data/'.format(station, mode)
    X, matJ, y, timeyear, dates = model_ghost.load_data(save_folder, max_files=96)
            
    
    ls_lambs = [0.000001 * (10**i) for i in range(2,7)]
    ls_global_lambs = [0.000001 * (10**i) for i in range(3,10)]
    
    lam = ls_lambs[0]
    global_lam = ls_global_lambs[3] 
    model = GAMCR.model.GAMCR(lam=lam)
    model.load_model('./{0}_{1}/data/params.pkl'.format(station, mode), lam=lam)
    save_folder = './{0}_{1}/'.format(station, mode)
    name_model = '{0}_{1}_best_model'.format(station, mode)
    loss = model.train(X, matJ, y, dates=dates, lr=1e-1, max_iter=10000, warm_start=False, save_folder=save_folder, name_model=name_model, normalization_loss=1, lam_global = global_lam)


else:
    if __name__ == '__main__':

        # To launch the code:  python CV_model.py --ls_stations Pully Basel Lugano --ls_modes flashy notflashy

        
        
        import pandas as pd
        import numpy as np
        import sys
        from joblib import Parallel, delayed
        sys.path.append('/mydata/watres/quentin/code/FLOW/hourly_analysis/')
        import GAMCR
        
        # ls_stations = ['Pully', 'Lugano', 'Basel']
        # ls_modes = ['flashy', 'notflashy']
        import argparse
        parser = argparse.ArgumentParser(description="Getting sites to train.")
    
        # Add arguments
        parser.add_argument('--ls_stations', nargs='+', required=True, help="List of station names")
        parser.add_argument('--ls_modes', nargs='+', help="List of modes")
        # Parse the arguments
        args = parser.parse_args()
    
        for station in args.ls_stations:
            for mode in args.ls_modes:    
                model_ghost = GAMCR.model.GAMCR(lam=0.1)
                save_folder = './{0}_{1}/data/'.format(station, mode)
                X, matJ, y, timeyear, dates = model_ghost.load_data(save_folder, max_files=25)
                
                ls_lambs = [0.000001 * (10**i) for i in range(2,7)]
                ls_global_lambs = [0.000001 * (10**i) for i in range(3,10)]
                
                # Define the function for model training
                def train_model(idx_lam, lam, idx_global_lam, global_lam):
                    model = GAMCR.model.GAMCR(lam=lam)
                    model.load_model('./{0}_{1}/data/params.pkl'.format(station, mode), lam=lam)
                    save_folder = './{0}_{1}/'.format(station, mode)
                    name_model = '{0}_{1}_trained_model_CV_{2}_{3}'.format(station, mode, idx_lam, idx_global_lam)
                    loss = model.train(X, matJ, y, dates=dates, lr=1e-1, max_iter=6000, warm_start=False, 
                                       save_folder=save_folder, name_model=name_model, normalization_loss=1, 
                                       lam_global=global_lam)
                    return loss
                
                # Parallelize the nested loops using joblib
                results = Parallel(n_jobs=4)(delayed(train_model)(idx_lam, lam, idx_global_lam, global_lam)
                                              for idx_lam, lam in enumerate(ls_lambs)
                                              for idx_global_lam, global_lam in enumerate(ls_global_lambs))
                
