def first():
    import numpy as np
    from sklearn.linear_model import LinearRegression

    hours = np.array([1,2,3,4,5,6])
    scores = np.array([35,42,50,58,65,72])

    model = LinearRegression()
    model.fit(hours.reshape(-1,1), scores)

    prediction = model.predict([[10]])

    print(prediction)

    print("Intercept:", model.intercept_)
    print("Slope:", model.coef_[0])

