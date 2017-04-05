
SVC_params = {
    'C':[1.0,100,0.001], 
    'cache_size':[200], 
    'class_weight':[None], 
    'coef0':[0.0],
    'decision_function_shape':[None], 
    'degree':[3], 
    'gamma':['auto'], 
    'kernel':['rbf','linear'],
    'max_iter':[-1], 
    'probability':[False], 
    'random_state':[None], 
    'shrinking':[True],
    'tol':[0.001], 
    'verbose':[False]
}
    
LinearSVC_params = {
    'C':[1.0,100,0.001],
    'class_weight':[None], 
    'dual':[True], 
    'fit_intercept':[True],
    'intercept_scaling':[1], 
    'loss':['squared_hinge'], 
    'max_iter':[1000,2000],
    'multi_class':['ovr'], 
    'penalty':['l2'], 
    'random_state':[None], 
    'tol':[0.0001],
    'verbose':[0]
}

NuSVC_params = {
    'cache_size':[200], 
    'class_weight':[None], 
    'coef0':[0.0],
    'decision_function_shape':[None], 
    'degree':[3], 
    'gamma':['auto'], 
    'kernel':['rbf','linear'],
    'max_iter':[-1], 
    'nu':[0.5], 
    'probability':[False, True], 
    'random_state':[None],
    'shrinking':[True], 
    'tol':[0.001], 
    'verbose':[False]
}
   
LogisticRegression_params = {
    'C':[1.0,100,0.001], 
    'class_weight':[None], 
    'dual':[False], 
    'fit_intercept':[True],
    'intercept_scaling':[1], 
    'max_iter':[100,1000],
    'multi_class':['ovr'], 
    'n_jobs':[1],
    'penalty':['l1','l2'], 
    'random_state':[None], 
    'solver':['liblinear'], 
    'tol':[0.0001],
    'verbose':[0], 
    'warm_start':[False]
}
          
SGDClassifier_params = {
    'alpha':[0.0001, 0.00001, 0.000001], 
    'average':[False], 
    'class_weight':[None], 
    'epsilon':[0.1],
    'eta0':[0.0], 
    'fit_intercept':[True], 
    'l1_ratio':[0.15],
    'learning_rate':['optimal'], 
    'loss':['hinge'], 
    'n_iter':[10, 80], 
    'n_jobs':[1],
    'penalty':['l2', 'elasticnet'], 
    'power_t':[0.5], 
    'random_state':[None], 
    'shuffle':[True],
    'verbose':[0], 
    'warm_start':[False]
}
       
MultinomialNB_params = {
    'alpha':[1.0, 100, 0.001], 
    'class_prior':[None], 
    'fit_prior':[True]
}

BernoulliNB_params = {
    'alpha':[1.0,100,0.001], 
    'binarize':[0.0], 
    'class_prior':[None], 
    'fit_prior':[True]
}
