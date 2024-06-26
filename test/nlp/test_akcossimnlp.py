from akrule.nlp import AKCosSimNLP

import pandas as pd

def test_akcossimnlp():
    values = [
        "The running back broke through the football defense and scored a 20-yard football touchdown.",
        "The point guard executed a perfect pass, leading to a three-point basketball shot.",
        "The football coach praised the football players for their dedication and hard work during football practice.",
        "The basketball coach emphasized the importance of teamwork and communication on the basketball court.",
        "During the final quarter, the basketball crowd cheered as the underdog team took the lead.",
        "The football quarterback threw a touchdown pass that secured the win for his football team.",
        "The football defense managed to intercept the ball, turning the football game around.",
        "The basketball game ended with an impressive slam dunk by the star basketball player.",
        "The team’s defense was tight, preventing the opponents from scoring any easy basketball baskets.",
        "The football match was intense, with both football teams giving their best until the final football whistle."
    ]
    X_test = pd.DataFrame(values, columns=["TEXT"])
    
    model = AKCosSimNLP(data=X_test)
    _ = model.fit_transform(X_test["TEXT"])
    
    text = "The football match was intense, with both football teams giving their best until the final football whistle."
    X_pred = model.predict(text)
    
    error_msg = "Incorret shape X_pred:{X_pred.shape[0]}, X_test:{X_test.shape[0]}!"
    assert X_pred.shape[0]==X_test.shape[0], error_msg
    
    error_msg = "COS_SIMILARITY not in X_pred.columns:{X_pred.columns}!"
    assert "COS_SIMILARITY" in X_pred.columns, error_msg
    
    error_msg = "First score is not 1, score:{X_pred['COS_SIMILARITY'].values[0]}!"
    assert X_pred["COS_SIMILARITY"].values[0]==1, error_msg
    