from sklearn.linear_model import LinearRegression

def regress_covariates_from_behavior(behaviors, covariates):
    # Regress covariates out of behavioral scores.
    if covariates is None:
        print('NO covariates')
        return behaviors
    print("Regressing covariates out of behavioral scores...")
    lr = LinearRegression()
    lr.fit(covariates, behaviors)
    residuals = behaviors - lr.predict(covariates)
    print('\nResiduals:\n', residuals,"\n")
    return residuals
